import torch

def get_lr_scheduler(opt: torch.optim, scheduler_name: str) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    '''Returns the learning rate scheduler out of the pytorch library'''
    
    if scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    else:
        raise NotImplemented

    return scheduler