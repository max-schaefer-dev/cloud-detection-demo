import torch

def get_optimizer(model_params: torch.Generator, lr: float, opt_name: str) -> torch.optim:
    '''Calls and returns chosen optimizer from the pytorch library.'''
    
    if opt_name=='Adam':
        opt = torch.optim.Adam(params=model_params, lr=lr)
    elif opt_name=='AdamW':
        opt = torch.optim.AdamW(params=model_params, lr=lr)
    elif opt_name=='RAdam':
        opt = torch.optim.RAdam(params=model_params, lr=lr)
    else:
        raise ValueError("Wrong optimizer name")
    return opt