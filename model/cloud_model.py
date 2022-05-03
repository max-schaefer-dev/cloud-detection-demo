from typing import List

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch

class CloudModel(pl.LightningModule):
    def __init__(
        self,
        bands: List[str],
        hparams: dict = {},
    ):
        """
        Instantiate the CloudModel class based on the pl.LightningModule.

        Args:
            bands (list[str]): Names of the bands provided for each chip
            hparams (dict, optional): Dictionary of additional modeling parameters.
        """
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()

        # required
        self.bands = bands
        self.encoder = self.hparams.get('encoder', 'resnet34')
        self.weights = self.hparams.get('weights', 'imagenet')
        self.gpu = self.hparams.get('gpu', False)

        self.model = self._prepare_model()


    def forward(self, image: torch.Tensor):
        # Forward pass
        embedding = self.model(image)
        return embedding

    def _prepare_model(self):

        # Instantiate U-Net model
        unet_model = smp.Unet(
            encoder_name=self.encoder,
            encoder_weights=self.weights,
            in_channels=4,
            classes=2
        )

        if self.gpu:
            unet_model.cuda()

        return unet_model