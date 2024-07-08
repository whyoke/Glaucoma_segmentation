import torch
from fastai.vision import *
from fastai.vision.all import *
from pathlib import Path
from functools import partial


DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 16
DEFAULT_VALIDATION_SPLIT = 0.1
SEED = 42


def get_images(path):
    all_files = get_image_files(path)
    images = [i for i in all_files if "mask" not in str(i)]
    return images


def get_masks(o: Path, mask_folder_path: Path):
    return mask_folder_path / f"{o.stem}{o.suffix}"


def get_data(
    image_folder_path: Path,
    mask_folder_path: Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    tfms=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    codes: list = [""],
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    num_workers: int = 0,
):
    get_msk = partial(get_masks, mask_folder_path=mask_folder_path)
    if not tfms:
        tfms = [IntToFloatTensor(div_mask=255), *aug_transforms(), Normalize.from_stats(*imagenet_stats)]
    db = DataBlock(
        blocks=(ImageBlock(), MaskBlock(codes=codes)),
        splitter=RandomSplitter(valid_pct=validation_split, seed=SEED),
        batch_tfms=tfms,
        item_tfms=[Resize(image_size)],
        get_items=get_image_files,
        get_y=get_msk,
    )
    return db.dataloaders(source=image_folder_path, batch_size=batch_size, pin_memory=True, num_workers=num_workers)


def dice(inputs, targs, iou=False, eps=1e-8, sz=256):
    # threshold for the number of predicted pixels
    noise_th = 75.0 * (sz / 128.0) ** 2
    best_thr0 = 0.2
    n = targs.shape[0]
    # inputs = torch.softmax(inputs, dim=1)[:,1,...].view(n,-1)
    inputs = torch.sigmoid(inputs).view(n, -1)
    inputs = (inputs > best_thr0).long()
    inputs[inputs.sum(-1) < noise_th, ...] = 0.0
    targs = targs.view(n, -1)
    intersect = (inputs * targs).sum(-1).float()
    union = (inputs + targs).sum(-1).float()
    if not iou:
        return ((2.0 * intersect + eps) / (union + eps)).mean()
    else:
        return ((intersect + eps) / (union - intersect + eps)).mean()


def unet_splitter(m):
    """Custom splitter to apply Discriminative learning"""
    return L(m.encoder, m.decoder).map(params)
