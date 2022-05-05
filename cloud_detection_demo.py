import os.path
import yaml
from pathlib import Path
import streamlit as st

from utils import streamlit_utils as utl
from utils.config import dict2cfg
from utils.postprocessing import postprocessing
from utils.process_sections import process_section_1, process_section_2, process_section_3

# Load app_settings
cfg_dict = yaml.load(open('app_settings.yaml', 'r'), Loader=yaml.FullLoader)
APP_CFG  = dict2cfg(cfg_dict)

st.set_page_config(layout="wide")
st.title('Cloud Model Demo')
utl.local_css("./css/streamlit.css")

# Section 1: Select chip_id
chip_id = process_section_1(APP_CFG.available_chip_id)

# Section 2: Select Model settings
m_options = process_section_2(APP_CFG.model_names, APP_CFG.postprocess_settings)

# Section 3: Inference & Analytics
process_section_3(chip_id, m_options)