import yaml
import os
import torch
import gdown
from pathlib import Path
import streamlit as st
from model.cloud_model import CloudModel
from utils.config import dict2cfg

# Load app_settings
cfg_dict = yaml.load(open('app_settings.yaml', 'r'), Loader=yaml.FullLoader)
APP_CFG  = dict2cfg(cfg_dict)

# Constant variables
CFG_DIR = Path('./configs/')

def prepare_model(model_name: str) -> CloudModel:
    '''Creates a CloudModel object with provided CFG and dataframe.'''
    
    # Read config file
    cfg_path = CFG_DIR / f'{model_name}-512.yaml'
    cfg_dict  = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)
    MODEL_CFG = dict2cfg(cfg_dict) # dict to class
    MODEL_WEIGHTS_P = Path(f'./temp/weights/{model_name}-512x512.pt')

    # Download weights
    if not MODEL_WEIGHTS_P.is_file():
        # Check if folder for weights exists
        if not os.path.isdir('temp/weights'):
            os.makedirs('temp/weights')

        gdown_id = MODEL_CFG.gdown_id # google drive id for model weights
        output = f'temp/weights/{model_name}-512x512.pt'

        # Downloading waits and displaying a massage
        with st.spinner(f'Please wait. Downloading {model_name} weights... ({MODEL_CFG.weight_size})'):
            gdown.download(id=gdown_id, output=output, quiet=False)

    
    # Initialize model
    cloud_model = CloudModel(bands=APP_CFG.bands, hparams=cfg_dict)
    cloud_model.load_state_dict(torch.load(MODEL_WEIGHTS_P))
    cloud_model.eval()

    st.success(f'{model_name} model initialized!')

    return cloud_model