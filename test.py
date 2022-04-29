import yaml
from utils.config import cfg2dict, dict2cfg

# Load app_settings
cfg_dict = yaml.load(open('app_settings.yaml', 'r'), Loader=yaml.FullLoader)
CFG      = dict2cfg(cfg_dict)
print(CFG.model_names)
# app_config = cfg2dict('app_settings.yaml')