import os.path
import yaml
import cv2
import urllib.request
import gdown
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from utils import streamlit_utils as utl
from utils.config import dict2cfg
from utils.image_comparison import image_comparison
from utils.inference import inference
from utils.metrics import calculate_scores
from utils.postprocessing import postprocessing
from utils.visualize import display_chip_bands, true_color_img, plot_pred_and_true_label

# Load app_settings
cfg_dict = yaml.load(open('app_settings.yaml', 'r'), Loader=yaml.FullLoader)
APP_CFG  = dict2cfg(cfg_dict)

# Constant variables
DATA_DIR = Path('./data/')
CFG_DIR = Path('./configs/')

st.set_page_config(layout="wide")
utl.local_css("./css/streamlit.css")

# Section: Select sample
st.title('Cloud Model Demo')
st.subheader('Select sample', anchor=None)
chip_id = st.selectbox(label='Select sample', options=APP_CFG.available_samples)
figure = display_chip_bands(chip_id)
st.pyplot(fig=figure)   

# Section: Select Model
st.subheader('Select Model & Settings', anchor=None)
m_col1, m_col2, m_col3 = st.columns([2,1,1])
with m_col1:
    # Choose model/s for inference 
    model_option = st.multiselect(label='Select model/s', options=APP_CFG.model_names, default='Resnet34-Unet')
with m_col2:
    # Select TTA option
    tta_option = st.number_input(
        label='Select TTA',
        min_value=0,
        max_value=3,
        step=1,
        help='Test-Time-Augmentation. 1 = average of 2 predictions (raw pred. & pred. on augmented/flipped image)')
with m_col3:
    # Select a treshold
    threshold_option = st.number_input(label='Select Inference Threshold', min_value=0.0, max_value=1.0, value=0.5, help='Default: 0.50. Values > Treshold get rounded up to 1 (Cloud). Values < Treshold get rounded to 0 (non-cloud)')

# Postprocessing
pp_col1, pp_col2, pp_col3 = st.columns([2,1,1])

with pp_col1:
    pp_option = st.selectbox(label='Select PP technique', options=APP_CFG.postprocess_settings, index=0, help='Choose post-processing technique which gets applied to the image after the prediciton.')
with pp_col2:

    if pp_option == 'Morphological Dilation':
        active = False
    else:
        active = True

    pp_iter_option = st.number_input(label='Iterations', min_value=1, max_value=7, value=1, help='How often should this option be applied? Only used with morph. Dilation.', disabled=active)
with pp_col3:

    if pp_option != 'None':
        active = False
    else:
        active = True

    pp_kernel_option = st.number_input(label='Kernel size', min_value=2, max_value=7, value=3, help='How big should the kernel be? Default (3,3).', disabled=active)

# Section: Inference
st.subheader('Inference', anchor=None)
start_inference = st.button('Start Inference')

if start_inference:

    assert model_option, 'No model has been selected.'

    # Inference pipeline
    pred_binary_image = inference(model_option, chip_id, tta_option, threshold_option)

    # Postprocessing if selected
    if pp_option != 'None':
        pred_binary_image = postprocessing(pred_binary_image, pp_option, pp_iter_option, pp_kernel_option)

    # Metric scores table
    true_label = Image.open(DATA_DIR / chip_id / 'label.tif')
    y_true = np.array(true_label).ravel()
    y_pred = (pred_binary_image/255).ravel()

    # Display dataframe with scores
    st.caption('<div style="text-align:center;"><h3>Metric Scores</h3></div>', unsafe_allow_html=True)
    score_df = calculate_scores(y_true, y_pred, chip_id, tta_option, model_option)
    st.table(data=score_df.head())

    # Plot true_color, prediction, label & FP vs. FN
    fig, difference = plot_pred_and_true_label(pred_binary_image, chip_id, true_label)
    st.pyplot(fig=fig)

    # Convert difference fig into array to plot it
    renderer = plt.gcf().canvas.get_renderer()
    diff_image = difference.make_image(renderer, unsampled=True)[0]

    # Plot FP vs. FN
    true_color = true_color_img(chip_id).to_numpy()
    true_color_o = np.asarray(Image.fromarray(true_color).convert('RGB'))
    diff_image = np.asarray(Image.fromarray(diff_image).convert('RGB'))
    dst = cv2.addWeighted(true_color_o, 0.6, diff_image, 0.4, 0)

    # True color with FP-vs-FN Overlay
    st.caption('<div style="text-align:center;"><h3>True color with FP-vs-FN Overlay</h3></div>', unsafe_allow_html=True)

    image_comparison(
        img1=Image.fromarray(true_color).convert('RGB'),
        label1='True color',
        img2=Image.fromarray(dst),
        label2='FP-vs-FN'
    )