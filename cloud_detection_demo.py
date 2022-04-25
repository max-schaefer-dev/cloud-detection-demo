import glob
import os.path
import yaml
import cv2
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
from cloud_model import CloudModel
from utils.config import dict2cfg
from utils.metrics import JaccardIndex

BANDS = ['B02', 'B03', 'B04', 'B08']
DATA_DIR = Path('./data/')
CFG_DIR = Path('./configs/')
AVAILABLE_MODELS = ['Resnet34-Unet', 'EfficientNetB1-Unet', 'Resnext50_32x4d-Unet']
AVAILABLE_SAMPLES = DATA_DIR.glob('*')
AVAILABLE_SAMPLES = [os.path.split(chip_id)[1] for chip_id in AVAILABLE_SAMPLES]
AVAILABLE_TTA = [0, 1, -1]
TTA_SETTINGS = [0,1,2,3]
POSTPROCESS_SETTINGS = ['None', 'Remove small areas', 'Morphological Close', 'Morphological Dilation']

st.set_page_config(layout="wide")
utl.local_css("./css/streamlit.css")

def get_xarray(filepath):
    """Put images in xarray.DataArray format"""
    im_arr = np.array(Image.open(filepath))
    return xarray.DataArray(im_arr, dims=["y", "x"])

def true_color_img(chip_id):
    """Given the path to the directory of Sentinel-2 chip feature images,
    plots the true color image"""
    chip_dir = DATA_DIR / chip_id
    red = get_xarray(chip_dir / "B04.tif")
    green = get_xarray(chip_dir / "B03.tif")
    blue = get_xarray(chip_dir / "B02.tif")

    return ms.true_color(r=red, g=green, b=blue)

# @st.cache(allow_output_mutation=True)
def display_chip_bands(chip_id='none'):
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.5)) #figsize=(20, 10)

    true_color = true_color_img(chip_id)
    plt.suptitle(f'chip_id: {chip_id}', fontsize=16)
    ax[0].imshow(true_color)
    ax[0].set_title('True color')

    for i, band in enumerate(BANDS, 1):
        datarray = get_xarray(f'./data/{chip_id}/{band}.tif')
        ax[i].imshow(datarray)
        ax[i].set_title(band)

    return fig


def initialize_model(model_name):
    # Read config file
    cfg_path = CFG_DIR / f'{model_name}-512.yaml'
    cfg_dict  = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)
    MODEL_CFG = dict2cfg(cfg_dict) # dict to class
    MODEL_WEIGHTS = Path(f'./weights/{model_name}-512x512.pt')
    
    # Initialize model
    cloud_model = CloudModel(bands=BANDS, hparams=cfg_dict)
    cloud_model.load_state_dict(torch.load(MODEL_WEIGHTS))
    cloud_model.eval()

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


def IOU(true, pred):
    # Intersection and union totals
    intersection = np.logical_and(true, pred)
    union = np.logical_or(true, pred)
    iou = intersection.sum() / union.sum()

    return round(iou, 4)

# Visualize prediction
def plot_pred_and_true_label(pred_binary_image):
    fig, ax = plt.subplots(1,3, figsize=(10,4))
    
    true_label = Image.open(DATA_DIR / chip_id / 'label.tif')
    iou_score = IOU(true_label, pred_binary_image)

    true_color = true_color_img(chip_id)

    plt.suptitle(f'IoU Score: {iou_score}', fontsize=16)

    ax[0].imshow(true_color)
    ax[0].set_title('True color')
    
    ax[1].imshow(pred_binary_image)
    ax[1].set_title('Prediction')

    ax[2].imshow(true_label)
    ax[2].set_title('True label')

    return fig

def stack_chip_bands(chip_id):
    # Prepare image
    band_arrs = []
    for band in BANDS:
        band_path = DATA_DIR / chip_id / f'{band}.tif'
        with rasterio.open(band_path) as b:
            band_arr = b.read(1).astype("float32")
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
                
        pred = (pred > 0.5).astype("uint8") # Round values > 0.5
        pred_binary_image = pred*255 # Scale [0,1] to [0,255] to visualize

    elif len(model_choice) > 1:

        # Stack all predictions
        stacked_pred = []
        for model_name in model_choice:
            
            cloud_model = initialize_model(model_name)
            pred_binary_image = prediction(cloud_model, image_arr, tta_option)
            # pred_binary_image = model_prediction(cloud_model, image_arr)

            stacked_pred.append(pred_binary_image)

        stacked_pred = np.stack(stacked_pred)
        pred = np.mean(stacked_pred, axis=0) # take mean of all predictions

        pred = (pred > 0.5).astype("uint8") # Round values for creating a binary image
        pred_binary_image = pred*255 # Scale [0,1] to [0,255] to visualize
    else:
        pass 


    fig = plot_pred_and_true_label(pred_binary_image)
    st.pyplot(fig=fig)   
    # st.image(image=binary_image, width=250)

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
    model_choice = st.multiselect(label='Select model/s', options=AVAILABLE_MODELS)
with col2:
    tta_option = st.selectbox(label='Select TTA (Test-Time-Augmentations)', options=TTA_SETTINGS)
    
# postprocess_option = st.selectbox(label='Select Post-Processing Technique', options=POSTPROCESS_SETTINGS)

# Section: Inference
st.subheader('Inference', anchor=None)
btn_click = st.button('Start Inference')

if btn_click:
    run_inference(model_choice, chip_id, tta_option)