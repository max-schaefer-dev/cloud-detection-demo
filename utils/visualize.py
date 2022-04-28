import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import xarray
import xrspatial.multispectral as ms
from pathlib import Path
import streamlit as st
from utils.metrics import calculate_scores

BANDS=['B02', 'B03', 'B04', 'B08']
BAND_NAMES=['Blue visible light','Green visible light', 'Red visible light', 'Near infrared light']
DATA_DIR = Path('./data/')


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

def display_chip_bands(chip_id='none'):
    """Given a chip_id and the spectral bands, plots the true color image and additionally all 4 bands"""
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.5))

    true_color = true_color_img(chip_id)

    ax[0].imshow(true_color)
    ax[0].set_title('True color (B02+B03+B04)', fontsize=11)

    for i, band in enumerate(BANDS, 1):
        datarray = get_xarray(f'./data/{chip_id}/{band}.tif')
        ax[i].imshow(datarray)
        ax[i].set_title(f'{BAND_NAMES[i-1]} ({band})', fontsize=11)

    return fig

# Visualize prediction
def plot_pred_and_true_label(pred_binary_image, chip_id, tta_option, model_name):
    fig, ax = plt.subplots(1,4, figsize=(14,7))
    
    true_label = Image.open(DATA_DIR / chip_id / 'label.tif')
    y_true = np.array(true_label).ravel()
    y_pred = (pred_binary_image/255).ravel()

    # Display dataframe with scores
    st.caption('<div style="text-align:center;"><h3>Metric Scores</h3></div>', unsafe_allow_html=True)
    score_df = calculate_scores(y_true, y_pred, chip_id, tta_option, model_name)
    st.table(data=score_df.head())

    # Setup 1st subplot
    true_color = true_color_img(chip_id)
    ax[0].imshow(true_color)
    ax[0].set_title('True color')

    # Setup 2nd subplot
    ax[1].imshow(pred_binary_image)
    ax[1].set_title('Prediction')

    # Setup 3rd subplot
    ax[2].imshow(true_label)
    ax[2].set_title('True label')

    # Setup 4rd subplot
    viridis = cm.get_cmap('viridis', 3)
    newcolors = viridis(np.linspace(0, 1, 3))
    red = np.array([256/256, 0/256, 0/256, 1])
    blue = np.array([256/256, 256/256, 256/256, 1])
    green = np.array([0/256, 256/256, 0/256, 1])
    newcolors[0] = red
    newcolors[1] = blue
    newcolors[2] = green
    newcmp = ListedColormap(newcolors)
    difference = np.array(true_label)-(pred_binary_image/255)
    diff_im = ax[3].imshow(difference, cmap=newcmp)

    # Setup title with colored rectangles
    red_rect = mpatches.Rectangle(
    (0.18, 1.035),
    width=0.05,
    height=0.05,
    color='red',
    transform=ax[3].transAxes,
    clip_on=False)

    green_rect = mpatches.Rectangle(
    (0.57, 1.035),
    width=0.05,
    height=0.05,
    color='green',
    transform=ax[3].transAxes,
    clip_on=False)

    ax[3].set_title(f'FP   vs.      FN',)
    ax[3].add_patch(red_rect)
    ax[3].add_patch(green_rect)

    return fig, diff_im