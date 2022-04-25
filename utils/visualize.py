import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import xarray
import xrspatial.multispectral as ms
from pathlib import Path

BANDS=['B02', 'B03', 'B04', 'B08']
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
    plt.suptitle(f'chip_id: {chip_id}', fontsize=16)
    ax[0].imshow(true_color)
    ax[0].set_title('True color')

    for i, band in enumerate(BANDS, 1):
        datarray = get_xarray(f'./data/{chip_id}/{band}.tif')
        ax[i].imshow(datarray)
        ax[i].set_title(band)

    return fig