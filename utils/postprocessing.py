'''Documentation for techniques used: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html'''

import cv2
import torch
import numpy as np

def postprocessing(image: np.array, option: str, iterations: int, k_size: int):

    assert option != 'None', 'Postprocessing triggered, although option "None" selected!'

    if option == 'Morphological Dilation':
        processed_image = morph_dilation(image=image, iterations=iterations, kernel_size=k_size)

    elif option == 'Morphological Closing':
        processed_image = morph_closing(image=image, kernel_size=k_size)

    elif option == 'Morphological Opening':
        processed_image = morph_opening(image=image, kernel_size=k_size)
    
    return processed_image


def morph_dilation(image: np.array, iterations: int = 1, kernel_size: int = 3):
    'Increases the region/size of foreground object.'

    kernel = np.ones(kernel_size,np.uint8)

    processed_image = cv2.dilate(image.astype(np.uint8), kernel, iterations = iterations)
    
    return processed_image


def morph_opening(image: np.array, iteration: int = 1, kernel_size: int = 3):
    'Removes noise within the image. Morpological erosion followed by dilation'

    kernel = np.ones(kernel_size,np.uint8)

    processed_image = cv2.morphologyEx(image.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=iteration)
    
    return processed_image


def morph_closing(image: np.array, iteration: int = 1, kernel_size: int = 3):
    'Closes small holes inside the foreground object. Morpological dilation followed by erosion.'

    kernel = np.ones(kernel_size,np.uint8)

    processed_image = cv2.morphologyEx(image.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=iteration)
    
    return processed_image