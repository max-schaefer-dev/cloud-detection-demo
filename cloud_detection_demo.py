import glob
import os.path
import yaml
import cv2
import urllib.request
import gdown
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from PIL import Image
import streamlit as st
import rasterio
import xarray
import xrspatial.multispectral as ms
import pytorch_lightning as pl
import torch

import streamlit_utils as utl
from streamlit_image_comparison import image_comparison
from cloud_model import CloudModel
from utils.config import dict2cfg
from utils.visualize import display_chip_bands, get_xarray, true_color_img, plot_pred_and_true_label

# Constant variables
BANDS = ['B02', 'B03', 'B04', 'B08']
DATA_DIR = Path('./data/')
CFG_DIR = Path('./configs/')

# Train settings
MODEL_NAMES = ['Resnet34-Unet', 'EfficientNetB1-Unet', 'Resnext50_32x4d-Unet']
AVAILABLE_SAMPLES = sorted([os.path.split(chip_id)[1] for chip_id in DATA_DIR.glob('*')])
AVAILABLE_TTA = [0, 1, -1]
TTA_SETTINGS = [0,1,2,3]
POSTPROCESS_SETTINGS = ['None', 'Remove small areas', 'Morphological Close', 'Morphological Dilation']

st.set_page_config(layout="wide")
utl.local_css("./css/streamlit.css")

def initialize_model(model_name):
    # Read config file
    cfg_path = CFG_DIR / f'{model_name}-512.yaml'
    cfg_dict  = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)
    MODEL_CFG = dict2cfg(cfg_dict) # dict to class
    MODEL_WEIGHTS = Path(f'./weights/{model_name}-512x512.pt')

    # Download weights
    if not MODEL_WEIGHTS.is_file():
        # Check if folder for weights exists
        if not os.path.isdir('weights'):
            os.mkdir('weights')

        gdown_id = MODEL_CFG.gdown_id # google drive id for model weights
        output = f'weights/{model_name}-512x512.pt'

        # Downloading waits and displaying a massage
        with st.spinner(f'Please wait. Downloading {model_name} weights... ({MODEL_CFG.weight_size})'):
            gdown.download(id=gdown_id, output=output, quiet=False)

    
    # Initialize model
    cloud_model = CloudModel(bands=BANDS, hparams=cfg_dict)
    cloud_model.load_state_dict(torch.load(MODEL_WEIGHTS))
    cloud_model.eval()

    st.success(f'{model_name} model initialized!')

    return cloud_model

def model_prediction(cloud_model, image_arr):
    # Prediction
    logits = cloud_model.forward(image_arr) # Get raw logits
    pred = torch.softmax(logits, dim=1)[:, 1] # Scale logits between [0,1]
    pred = pred.detach().numpy()
    pred = np.squeeze(pred, axis=0) # drop batch dim.
    
    return pred

def prediction(cloud_model, image_arr, tta_option):
    if tta_option != 0:
        stacked_pred = []

        for tta in AVAILABLE_TTA[:tta_option]:
            # Augment image (flip)
            flipped_image_arr = cv2.flip(image_arr, flipCode=tta)
            prep_image_arr = prep_image_dims(flipped_image_arr)

            flipped_pred = model_prediction(cloud_model, prep_image_arr)

            # Reverse Augmentation (flip back)
            pred = cv2.flip(flipped_pred, flipCode=tta)

            stacked_pred.append(pred)

        prep_image_arr = prep_image_dims(image_arr)
        pred = model_prediction(cloud_model, prep_image_arr)
        stacked_pred.append(pred)

        stacked_pred = np.stack(stacked_pred)
        pred = np.mean(stacked_pred, axis=0)
    else:
        prep_image_arr = prep_image_dims(image_arr)
        pred = model_prediction(cloud_model, prep_image_arr)

    return pred

def stack_chip_bands(chip_id):
    # Prepare image
    band_arrs = []
    for band in BANDS:
        band_path = DATA_DIR / chip_id / f'{band}.tif'
        with rasterio.open(band_path) as b:
            band_arr = b.read(1).astype('float32')
        band_arrs.append(band_arr)
    image_arr = np.stack(band_arrs, axis=-1)

    return image_arr

def prep_image_dims(image_arr):
    image_arr = image_arr[None, :] # add batch dimension
    image_arr = torch.from_numpy(image_arr) # numpy to torch tensor
    image_arr = torch.permute(image_arr,(0,3,1,2)).float() # Add batch dim.

    return image_arr

def run_inference(model_choice, chip_id, tta_option):

    image_arr = stack_chip_bands(chip_id)

    if len(model_choice) == 1:
        model_name = model_choice[0]
        cloud_model = initialize_model(model_name)

        pred = prediction(cloud_model, image_arr, tta_option)
                
        pred = (pred > 0.5).astype('uint8') # Round values > 0.5
        pred_binary_image = pred*255 # Scale [0,1] to [0,255] to visualize

    elif len(model_choice) > 1:

        # Initialize model and store in list
        models = [initialize_model(model_name) for model_name in model_choice]

        # Stack all predictions
        stacked_pred = []

        with st.spinner(f'Predicting...'):
            for model in models:
                
                pred_binary_image = prediction(model, image_arr, tta_option)

                stacked_pred.append(pred_binary_image)

            stacked_pred = np.stack(stacked_pred)
            pred = np.mean(stacked_pred, axis=0) # take mean of all predictions

            pred = (pred > 0.5).astype('uint8') # Round values for creating a binary image
            pred_binary_image = pred*255 # Scale [0,1] to [0,255] to visualize
    else:
        pass 

    
    fig, difference = plot_pred_and_true_label(pred_binary_image, chip_id, tta_option, model_choice)
    st.pyplot(fig=fig)

    return pred_binary_image, difference

# Section: Select sample
st.title('Cloud Model Demo')
st.subheader('Select sample', anchor=None)
chip_id = st.selectbox(label='Select sample', options=AVAILABLE_SAMPLES)
figure = display_chip_bands(chip_id)
st.pyplot(fig=figure)   

# Section: Select Model
st.subheader('Select Model & Settings', anchor=None)
col1, col2 = st.columns([1,1])
with col1:
    model_choice = st.multiselect(label='Select model/s', options=MODEL_NAMES, default='Resnet34-Unet')
with col2:
    tta_option = st.selectbox(label='Select TTA (Test-Time-Augmentations)', options=TTA_SETTINGS)

# Section: Inference
st.subheader('Inference', anchor=None)
btn_click = st.button('Start Inference')

if btn_click:

    assert model_choice, 'No model has been selected.'

    pred_binary_image, difference = run_inference(model_choice, chip_id, tta_option)

    renderer = plt.gcf().canvas.get_renderer()
    diff_image = difference.make_image(renderer, unsampled=True)[0]

    true_color = true_color_img(chip_id).to_numpy()
    true_color_o = np.asarray(Image.fromarray(true_color).convert('RGB'))

    diff_image = np.asarray(Image.fromarray(diff_image).convert('RGB'))

    dst = cv2.addWeighted(true_color_o, 0.6, diff_image, 0.4, 0)

    st.caption('<div style="text-align:center;"><h3>True color with FP-vs-FN Overlay</h3></div>', unsafe_allow_html=True)

    image_comparison(
        img1=Image.fromarray(true_color).convert('RGB'),
        label1='True color',
        img2=Image.fromarray(dst),
        label2='FP-vs-FN'
    )