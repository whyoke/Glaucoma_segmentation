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

from .utils.data_fields import DRPredictionType

DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_ID2LABEL_MAP = {
    0: DRPredictionType.no_dr.value,
    1: DRPredictionType.dr.value,
}
PRETRAINED_MODEL_LAST_LINEAR_CONFIG = {
    "in_features": 2048,
    "out_features": 1,
    "bias": True,
    }


# !git clone https://github.com/Cadene/pretrained-models.pytorch.git
# import sys
# sys.path.append("../input/pretrained-models/pretrained-models/pretrained-models.pytorch-master/")
# import pretrainedmodels

import sys

# sys.path.append("pretrained-models/pretrained-models.pytorch-master/")
import pretrainedmodels


def get_pretrained_model(
    model_name="resnet18",
    num_outputs=None,
    pretrained=True,
    freeze_bn=False,
    dropout_p=0,
    linear_bias=True,
    **kwargs
):
    """
    A function to get a pretrained model from 'pretrainedmodels'.
    Usage:
        !git clone https://github.com/Cadene/pretrained-models.pytorch.git
        import sys
        sys.path.append("../input/pretrained-models/pretrained-models/pretrained-models.pytorch-master/")
        import pretrainedmodels

        BACKBONE_NAME = "se_resnext50_32x4d"
        # create model
        pretrained_model = get_pretrained_model(model_name=BACKBONE_NAME,
                            num_outputs=1,
                            pretrained=False,
                            freeze_bn=True,
                            dropout_p=0,
                            )
        pretrained_path = f"../pretrained_{BACKBONE_NAME}.pth"
        pretrained_model.load_state_dict(torch.load(pretrained_path))
        pretrained_model.last_linear = nn.Linear(in_features = 2048, out_features = 2, bias = True)
    """
    pretrained = "imagenet" if pretrained else None
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=pretrained)

    in_features = model.last_linear.in_features

    if "dpn" in model_name:
        in_channels = model.last_linear.in_channels
        model.last_linear = nn.Conv2d(in_channels, num_outputs, kernel_size=1, bias=True)
    else:
        if "resnet" in model_name:
            model.avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            model.avg_pool = nn.AdaptiveAvgPool2d(1)
        if dropout_p == 0:
            model.last_linear = nn.Linear(in_features, num_outputs)
        else:
            model.last_linear = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_outputs),
            )
    if num_outputs:
        model.last_linear = nn.Linear(in_features, num_outputs, bias=linear_bias)

    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    return model


class DRModel(LightningModule):
    """Diabetic Retinopathy Model for Training with PyTorch Lightning"""

    def __init__(
        self,
        backbone_name: str = "se_resnext50_32x4d",
        pretrained_weight_path: str = "../trained_models/pretrained_se_resnext50_32x4d.pth",
        learning_rate: float = 1e-3,
        device: str = "cpu",
        init_pretrained: bool = True,
        configs=None,
        **kwargs,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.accuracy = Accuracy()
        self.loss = nn.CrossEntropyLoss()

        # Get pretrained DR model.
        self.model = get_pretrained_model(
            model_name=backbone_name,
            num_outputs=2,
            pretrained=False,
            freeze_bn=True,
            dropout_p=0,
            linear_bias=True,
            )

        if init_pretrained:
            self.init_pretrained_model(pretrained_weight_path, device=device)

        self.configs = {
            "learning_rate": learning_rate,
        }
        if configs:
            if isinstance(configs, dict):
                self.configs.update(configs)
            else:
                raise ValueError("configs must be a dict")

        # for wandb logging
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

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
        json.dump(self.configs, open(save_path, "w"))

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
        configs["init_pretrained"] = True
        configs["pretrained_weight_path"] = op.join(model_path, "pretrained_se_resnext.ckpt")
        return cls.load_from_checkpoint(op.join(model_path, "model.ckpt"), map_location = device, **configs)

    def init_pretrained_model(self, path: str, device: str) -> None:
        # Keep our last linear layer.
        true_last_linear = self.model.last_linear
        # Revert last linear layer to original weights' configs to match the weight.
        self.model.last_linear = nn.Linear(**PRETRAINED_MODEL_LAST_LINEAR_CONFIG)
        self.model.load_state_dict(torch.load(path, map_location=torch.device(device)))

        # Revert last linear layer to our last linear layer.
        self.model.last_linear = true_last_linear
        print(f"{self.__class__.__name__} is on the device: ", next(self.model.parameters()).device)


class DRClassifier:
    """Diabetic Retinopathy Classifier for Inference"""

    def __init__(
        self,
        model: Union[nn.Module, str],
        image_transforms=None,
        id2label_map: Dict[int, str] = DEFAULT_ID2LABEL_MAP,
        device: str = "cpu",
    ):
        if isinstance(model, str):
            self.transforms = A.load(op.join(model, "transform.json"))
            self.model = DRModel.from_configs(model_path=model, device=device)
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
    def predict(self, image: Union[str, PIL.Image.Image]) -> Tuple[str, float]:
        if isinstance(image, str):
            image = PIL.Image.open(image)
        image = image.convert("RGB")
        image = self.transforms(image=np.array(image))["image"]
        image = image.unsqueeze(0)

        logits = self.model(image.to(self.device))
        predicted_probabilities = F.softmax(logits, dim=1)
        predicted_dr = predicted_probabilities.argmax(dim=1).tolist()[0]
        predicted_dr = self.id2label_map[predicted_dr]
        predicted_probabilities = predicted_probabilities.tolist()[0]
        predicted_probabilities = [round(p, 2) for p in predicted_probabilities]
        predicted_probability_dict = {
            "negative": predicted_probabilities[0],
            "positive": predicted_probabilities[1],
        }
        return {
            "probability": predicted_probability_dict,
            "prediction": predicted_dr,
        }
