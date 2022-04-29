class Config:
    '''Object used as an dict. for easy access'''
    def __init__(self, data):
        self.__dict__.update(**data)

def dict2cfg(cfg_dict: dict) -> Config:
    '''Create config from dictionary.

    Args:
        cfg_dict (dict): dictionary with configs to be converted to config.

    Returns:
        cfg: python class object as config
    '''
    
    return Config(cfg_dict) # dict to cfg


def cfg2dict(cfg: Config) -> dict:
    '''Create dictionary from config.
    
    Args:
        cfg (config): python class object as config.
    
    Returns:
        cfg_dict (dict): dictionary with configs.
    '''
    return {k:v for k,v in dict(vars(cfg)).items() if '__' not in k}