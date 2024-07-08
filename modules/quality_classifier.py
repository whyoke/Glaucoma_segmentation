import os.path as op
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from typing import Dict, Union, List, Tuple
import PIL

import torchvision.models as models
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule
import timm
import albumentations as A

from .utils.data_fields import ImageQualityPredictionType 


DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_ID2LABEL_MAP = {
    0: ImageQualityPredictionType.good.value,
    1: ImageQualityPredictionType.acceptable.value,
    2: ImageQualityPredictionType.poor.value,
}


class FundusImageQualityClassifier:
    """Fundus Image Quality Classifier for Inference."""

    def __init__(
        self,
        model: Union[nn.Module, str],
        image_transforms=None,
        id2label_map: Dict[int, str] = DEFAULT_ID2LABEL_MAP,
        device: str = "cpu",
    ):
        if isinstance(model, str):
            self.transforms = A.load(op.join(model, "transform.json"))
            self.model = ImageQualityModel.from_configs(model_path=model, device=device)
        elif isinstance(model, nn.Module):
            self.transforms = image_transforms
            if self.transforms is None:
                raise ValueError("image_transforms must be provided if 'model' is not a string path")
            self.model = model

        self.model.to(device)
        self.model.eval()
        self.device = device

        self.id2label_map = id2label_map

    @torch.no_grad()
    def predict(self, image: Union[str, PIL.Image.Image]) -> Dict[List[float], str]:
        if isinstance(image, str):
            image = PIL.Image.open(image)
        image = image.convert("RGB")
        image = self.transforms(image=np.array(image))["image"]
        image = image.unsqueeze(0)

        logits = self.model(image.to(self.device))
        predicted_probabilities = F.softmax(logits, dim=1)
        predicted_quality = predicted_probabilities.argmax(dim=1).tolist()[0]
        predicted_quality = self.id2label_map[predicted_quality]
        predicted_probabilities = predicted_probabilities.tolist()[0]
        predicted_probabilities = [round(p, 2) for p in predicted_probabilities]
        predicted_probability_dict = {
            "good": predicted_probabilities[0],
            "acceptable": predicted_probabilities[1],
            "poor": predicted_probabilities[2],
        }
        return {
            "probability": predicted_probability_dict,
            "prediction": predicted_quality,
        }


class ImageQualityModel(LightningModule):
    """Fundus Image Quality Classifier for Training."""

    def __init__(
        self,
        in_channels: int = 3,
        learning_rate: float = 1e-3,
        pretrained: bool = False,
        backbone_name: str = "resnet34",
        latent_dim: int = 512,
        bias_head: bool = False,
        device: str = "cpu",
        configs: dict = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.learning_rate = learning_rate
        self.pretrained = pretrained
        self.backbone_name = backbone_name
        self.latent_dim = latent_dim
        self.bias_head = bias_head

        self.accuracy = Accuracy()
        self.loss = nn.CrossEntropyLoss()

        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=latent_dim,
        )

        num_in_features = self.model.get_classifier().in_features

        self.model.fc = nn.Linear(in_features=num_in_features, out_features=latent_dim, bias=False)
        self.head = nn.Linear(latent_dim, 3, bias=bias_head)

        self.configs = {
            "in_channels": in_channels,
            "learning_rate": learning_rate,
            "pretrained": pretrained,
            "backbone_name": backbone_name,
            "latent_dim": latent_dim,
            "bias_head": bias_head,
        }

        if configs:
            if isinstance(configs, dict):
                self.configs.update(configs)
            else:
                raise ValueError("configs must be a dict")

        # for wandb logging
        self.save_hyperparameters()

    def forward(self, x):
        embeddings = self.model(x)
        out = self.head(embeddings)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.2, patience=3, min_lr=1e-6, verbose=True
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "monitor": "val_loss",
        }

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def save_configs(self, save_path: str):
        json.dump(
            self.configs,
            open(op.join(save_path, "configs.json"), "w", encoding="utf-8"),
        )

    @classmethod
    def from_configs(
        cls,
        model_path: str,
        device: str = "cpu",
    ):
        if isinstance(model_path, str):
            configs = json.load(open(op.join(model_path, "configs.json"), "r", encoding="utf-8"))
        else:
            raise ValueError("configs must be a string path to a folder containing configs.json and model.ckpt")
        configs["device"] = device
        return cls(**configs).load_from_checkpoint(op.join(model_path, "model.ckpt"), map_location = device)
