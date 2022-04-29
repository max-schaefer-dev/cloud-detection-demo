from typing import Optional, Union, Mapping

import ttach as tta
import torch
import torch.nn as nn
from ttach.wrappers import SegmentationTTAWrapper
from utils.config import Config
from cloud_model import CloudModel

def prepare_tta(model: CloudModel, CFG: Config) -> CloudModel:
    '''Wraps pytorch model into a SegmentationTTAWrapper'''

    if CFG.tta == 1:
        augmentations = [tta.HorizontalFlip()]
    elif CFG.tta == 2:
        augmentations = [tta.HorizontalFlip(), tta.VerticalFlip()]
    elif CFG.tta == 3:
        augmentations = [tta.HorizontalFlip(), tta.VerticalFlip(), tta.Rotate90(angles=[0, 90])]
    else:
        return model

    transforms = tta.Compose(augmentations)

    model = SegmentationTTAWrapper(model, transforms, merge_mode='mean')

    return model
