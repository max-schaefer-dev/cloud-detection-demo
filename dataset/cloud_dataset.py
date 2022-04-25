import numpy as np
import pandas as pd
import rasterio
import torch
from typing import Optional, List

class CloudDataset(torch.utils.data.Dataset):
    """Reads in images, transforms pixel values, and serves a
    dictionary containing chip ids, image tensors, and
    label masks (where available).
    """

    def __init__(
        self,
        x_paths: pd.DataFrame,
        y_paths: Optional[pd.DataFrame] = None,
        bands: List[str] = ['B02','B03','B04','B08'],
        transforms: Optional[list] = None,
        LGJ: bool = False,
    ):
        """
        Instantiate the CloudDataset class.

        Args:
            x_paths (pd.DataFrame): a dataframe with a row for each chip. There must be a column for chip_id,
                and a column with the path to the TIF for each of bands
            bands (list[str]): list of the bands included in the data
            y_paths (pd.DataFrame, optional): a dataframe with a for each chip and columns for chip_id
                and the path to the label TIF with ground truth cloud cover
            transforms (list, optional): list of transforms to apply to the feature data (eg augmentations)
        """
        self.data = x_paths
        self.label = y_paths
        self.transforms = transforms
        self.bands = bands


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        # Loads an n-channel image from a chip-level dataframe
        img = self.data.loc[idx]
        band_arrs = []
        for band in self.bands:
            with rasterio.open(img[f"{band}_path"]) as b:
                band_arr = b.read(1).astype("float32")
            band_arrs.append(band_arr)
        x_arr = np.stack(band_arrs, axis=-1)

        # Prepare dictionary for item
        item = {"chip_id": img.chip_id, "chip": x_arr}

        if self.label is not None:
            label_path = self.label.loc[idx].label_path
            with rasterio.open(label_path) as lp:
                y_arr = lp.read(1).astype("float32")
            # Apply same data augmentations to the label
            if self.transforms:
                transformed = self.transforms(image=x_arr, mask=y_arr)
                y_arr = transformed['mask']
                x_arr = transformed['image']

            item["label"] = y_arr.astype("float32")
            x_arr = np.transpose(x_arr, [2, 0, 1])
            item["chip"] = x_arr.astype("float32")
        else:
            if self.transforms:
                x_arr = self.transforms(image=x_arr)["image"]

            x_arr = np.transpose(x_arr, [2, 0, 1])
            item["chip"] = x_arr

        return item