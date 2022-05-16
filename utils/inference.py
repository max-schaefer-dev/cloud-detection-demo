import torch
import numpy as np
import cv2
import yaml
import streamlit as st
from model.cloud_model import CloudModel
from utils.config import dict2cfg
from utils.prepare_model import prepare_model
from utils.visualize import prep_image_dims, stack_chip_bands

# Load app_settings
cfg_dict = yaml.load(open('app_settings.yaml', 'r'), Loader=yaml.FullLoader)
APP_CFG  = dict2cfg(cfg_dict)


def prediction(cloud_model: CloudModel, image_arr: np.array) -> np.array:
    '''Predict binary mask for given image array'''

    logits = cloud_model.forward(image_arr) # Get raw logits
    pred = torch.softmax(logits, dim=1)[:, 1] # Scale logits between [0,1]
    pred = pred.detach().numpy()
    pred = np.squeeze(pred, axis=0) # drop batch dim.
    
    return pred


def prepare_prediction(cloud_model: CloudModel, image_arr: np.array, tta_option: list) -> np.array:
    '''Prediction pipeline. Stacks predictions and takes average'''

    if tta_option != 0: 
        stacked_pred = []

        for tta in APP_CFG.available_tta[:tta_option]:
            # Augment image (flip)
            flipped_image_arr = cv2.flip(image_arr, flipCode=tta)
            prep_image_arr = prep_image_dims(flipped_image_arr)

            flipped_pred = prediction(cloud_model, prep_image_arr)

            # Reverse Augmentation (flip back)
            pred = cv2.flip(flipped_pred, flipCode=tta)

            stacked_pred.append(pred)

        prep_image_arr = prep_image_dims(image_arr)
        pred = prediction(cloud_model, prep_image_arr)
        stacked_pred.append(pred)

        stacked_pred = np.stack(stacked_pred)
        pred = np.mean(stacked_pred, axis=0)
    else:
        prep_image_arr = prep_image_dims(image_arr)
        pred = prediction(cloud_model, prep_image_arr)

    return pred


def inference(model_choice: list, chip_id: str, tta_option: list, threshold: float) -> np.array:

    image_arr = stack_chip_bands(chip_id)

    if len(model_choice) == 1:
        model_name = model_choice[0]
        cloud_model = prepare_model(model_name)

        pred = prepare_prediction(cloud_model, image_arr, tta_option)
                
        pred = (pred > threshold).astype('uint8') # Round values > threshold
        pred_binary_image = pred*255 # Scale [0,1] to [0,255] to visualize

    elif len(model_choice) > 1:

        # Initialize model and store in list
        models = [prepare_model(model_name) for model_name in model_choice]

        # Stack all predictions
        stacked_pred = []

        with st.spinner(f'Predicting...'):
            for model in models:
                
                pred_binary_image = prepare_prediction(model, image_arr, tta_option)

                stacked_pred.append(pred_binary_image)

            stacked_pred = np.stack(stacked_pred)
            pred = np.mean(stacked_pred, axis=0) # take mean of all predictions

            pred = (pred > threshold).astype('uint8') # Round values for creating a binary image
            pred_binary_image = pred*255 # Scale [0,1] to [0,255] to visualize
    else:
        pass 

    return pred_binary_image