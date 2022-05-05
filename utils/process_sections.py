import cv2
import numpy as np
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from utils.image_comparison import image_comparison
from utils.inference import inference
from utils.metrics import calculate_scores
from utils.postprocessing import postprocessing
from utils.visualize import display_chip_bands, plot_pred_and_true_label, true_color_img


# Constant variables
DATA_DIR = Path('./data/')
CFG_DIR = Path('./configs/')

def process_section_1(available_samples: list) -> str:
    '''Displays section title, displays a selectbox for chip_id and returns the selected chip_id'''

    st.subheader('1. Select chip', anchor=None)

    chip_id = st.selectbox(label='Select chip', options=available_samples)
    display_chip_bands(chip_id)

    return chip_id


def process_section_2(model_names: list, postprocess_settings: list) -> dict:
    '''Displays section title, selection boxes for model settings and returns a Dictionary containing all of the selected settings'''

    st.subheader('2. Select Model & Settings', anchor=None)

    row1_col1, row1_col2, row1_col3 = st.columns([2,1,1]) # Model Options
    row2_col1, row2_col2, row2_col3 = st.columns([2,1,1]) # Postprocessing options

    # Choose model/s for inference 
    with row1_col1:
        model_option = st.multiselect(label='Select model/s', options=model_names, default='Resnet34-Unet')

    # Select TTA option
    with row1_col2:
        tta_option = st.number_input(
            label='Select TTA',
            min_value=0,
            max_value=3,
            step=1,
            help='Test-Time-Augmentation. 1 = average of 2 predictions (raw pred. & pred. on augmented/flipped image)')

    # Select a treshold
    with row1_col3:
        threshold_option = st.number_input(label='Select Inference threshold', min_value=0.0, max_value=1.0, value=0.5, help='Default: 0.50. Values > Treshold get rounded up to 1 (Cloud). Values < Treshold get rounded to 0 (non-cloud)')

    # Select postprocessing technique
    with row2_col1:
        pp_option = st.selectbox(label='Select Post-Processing technique', options=postprocess_settings, index=0, help='Choose post-processing technique which gets applied to the image after the prediciton.')

    # Select postprocessing num. of iterations
    with row2_col2:
        if pp_option == 'Morphological Dilation':
            active = False
        else:
            active = True

        pp_iter_option = st.number_input(label='Iterations', min_value=1, max_value=7, value=1, help='How often should this option be applied? Only used with morph. Dilation.', disabled=active)

    # Select postprocessing kernel size
    with row2_col3:
        if pp_option != 'None':
            active = False
        else:
            active = True

        pp_kernel_option = st.number_input(label='Kernel size', min_value=2, max_value=7, value=3, help='How big should the kernel be? Default (3,3).', disabled=active)

    model_options = {
        'model_option': model_option,
        'tta_option': tta_option,
        'threshold_option': threshold_option,
        'pp_option': pp_option,
        'pp_iter_option': pp_iter_option,
        'pp_kernel_option': pp_kernel_option

    }

    return model_options


def process_section_3(chip_id: str, m_options: dict) -> None:
    '''Displays section header & the Metric Score table. Plots 4 images (true color image, predicted binary mask, true label (bin. mask) & FP vs. FN). Finally the image comparison (True color with FP-vs-FN Overlay) gets displayed.'''

    st.subheader('3. Inference & Analytics', anchor=None)
    start_inference = st.button('Start Inference')

    if start_inference:

        assert m_options['model_option'], 'No model has been selected.'

        # Inference pipeline
        pred_binary_image = inference(m_options['model_option'], chip_id, m_options['tta_option'], m_options['threshold_option'])

        # Postprocessing if selected
        if m_options['pp_option'] != 'None':
            pred_binary_image = postprocessing(pred_binary_image, m_options['pp_option'], m_options['pp_iter_option'], m_options['pp_kernel_option'])


        ## Metric scores table
        st.caption('<div style="text-align:center;"><h3>Metric Scores</h3></div>', unsafe_allow_html=True)

        # Prepare prediction and label to compute scores
        true_label = Image.open(DATA_DIR / chip_id / 'label.tif')
        y_true = np.array(true_label).ravel()
        y_pred = (pred_binary_image/255).ravel()

        score_df = calculate_scores(y_true, y_pred, chip_id, m_options)
        st.table(data=score_df.head())

        ## Plot true_color, prediction, label & FP vs. FN
        fig, difference = plot_pred_and_true_label(pred_binary_image, true_label, chip_id)
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