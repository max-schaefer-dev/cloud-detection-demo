import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import rasterio
import xarray
import xrspatial.multispectral as ms
import torch
import yaml
from pathlib import Path
import streamlit as st
from utils.config import dict2cfg

# Load app_settings
cfg_dict = yaml.load(open('app_settings.yaml', 'r'), Loader=yaml.FullLoader)
APP_CFG  = dict2cfg(cfg_dict)

# Constant variables
DATA_DIR = Path('./data/')

def get_xarray(filepath):
    '''
    Put images in xarray.DataArray format.

    Args:
        filepath (str): path to file

    Returns:
        array (xarray.DataArray): image as xarray.DataArray object
    '''
    im_arr = np.array(Image.open(filepath))

    return xarray.DataArray(im_arr, dims=["y", "x"])

def true_color_img(chip_id):
    '''
    Given the path to the directory of Sentinel-2 chip feature images,
    plots the true color image

    Args:
        chip_id (str): chip_id

    Returns:
        array (ms.true_color): stacked xarray objects
    '''
    chip_dir = DATA_DIR / chip_id.lower()
    red = get_xarray(chip_dir / "B04.tif")
    green = get_xarray(chip_dir / "B03.tif")
    blue = get_xarray(chip_dir / "B02.tif")

    return ms.true_color(r=red, g=green, b=blue)

def stack_chip_bands(chip_id):
    '''
    Stack 4 frequenz bands belonging to chip_id. B02, B03, B04, B08

    Args:
        chip_id (str): chip_id

    Returns:
        array (np.array): stacked bands
    '''
    # Prepare image
    band_arrs = []
    for band in APP_CFG.bands:
        band_path = DATA_DIR / chip_id / f'{band}.tif'
        with rasterio.open(band_path) as b:
            band_arr = b.read(1).astype('float32')
        band_arrs.append(band_arr)
    image_arr = np.stack(band_arrs, axis=-1)

    return image_arr

def prep_image_dims(image_arr):
    '''
    Prepare image_arr to inference.
    Add batch dim., convert to tensor and rearrange dims.

    Args:
        image_arr (np.array): chip_id

    Returns:
        image_arr (torch.tensor): prepared torch tensor
    '''
    image_arr = image_arr[None, :] # add batch dimension
    image_arr = torch.from_numpy(image_arr) # numpy to torch tensor
    image_arr = torch.permute(image_arr,(0,3,1,2)).float() # rearrange dim.

    return image_arr

def display_chip_bands(chip_id='none'):
    '''
    Plot True Color & all 4 bands in a subplot

    Args:
        chip_id (str): chip_id

    Returns:
        fig (plt.Figure): Matplotlib Figure object
    '''
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.5))

    true_color = true_color_img(chip_id)

    ax[0].imshow(true_color)
    ax[0].set_title('True color (B02+B03+B04)', fontsize=11)

    for i, band in enumerate(APP_CFG.bands, 1):
        datarray = get_xarray(f'./data/{chip_id}/{band}.tif')
        ax[i].imshow(datarray)
        ax[i].set_title(f'{APP_CFG.band_names[i-1]} ({band})', fontsize=11)

    return fig

# Visualize prediction
def plot_pred_and_true_label(pred_binary_image, chip_id, true_label):
    '''
    Plot True Color & all 4 bands in a subplot

    Args:
        pred_binary_image (np.array): predicted binary mask
        chip_id (str): chip_id
        true_label (np.array): true binary mask (label)

    Returns:
        fig (plt.Figure): Matplotlib Figure object
        diff_im (np.array): Array displaying the diff. between label & prediction
    '''
    fig, ax = plt.subplots(1,4, figsize=(14,7))

    st.caption('<div style="text-align:center;"><h3>Compare prediction & label</h3></div>', unsafe_allow_html=True)
    # Setup 1st subplot
    true_color = true_color_img(chip_id)
    ax[0].imshow(true_color)
    ax[0].set_title('True color')

    # Setup 2nd subplot
    ax[1].imshow(pred_binary_image)
    ax[1].set_title('Prediction')

    # Setup 3rd subplot
    ax[2].imshow(true_label)
    ax[2].set_title('Label')

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