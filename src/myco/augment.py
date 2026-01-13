"""Augmentation policies aligned with LEMON MoCo v3 training."""
from __future__ import annotations

import random

from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .utils import center_crop


class RotationCrop40:
    """Rotate without black corners using a larger crop then center-crop."""

    def __init__(self, big_size: int = 60, out_size: int = 40, degrees: float = 360.0) -> None:
        self.big_size = big_size
        self.out_size = out_size
        self.degrees = degrees

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.size != (self.big_size, self.big_size):
            img = img.resize((self.big_size, self.big_size), resample=Image.BICUBIC)
        angle = random.random() * self.degrees
        img = TF.rotate(img, angle=angle, interpolation=TF.InterpolationMode.BILINEAR, expand=False)
        return center_crop(img, self.out_size)


def build_lemon_a1_gray_transform(img_size: int = 40) -> T.Compose:
    """Return the LEMON a1 + gray augmentation pipeline."""
    rrc = T.RandomResizedCrop(
        size=img_size,
        scale=(0.32, 1.0),
        ratio=(3 / 4, 4 / 3),
        interpolation=T.InterpolationMode.BILINEAR,
    )
    cj = T.ColorJitter(brightness=0.6, contrast=0.7, saturation=0.5, hue=0.2)
    blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    re = T.RandomErasing(p=0.3, scale=(0.1, 0.3), ratio=(0.8, 1.2), value="random")

    return T.Compose(
        [
            rrc,
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([cj], p=0.8),
            T.RandomGrayscale(p=0.2),
            blur,
            T.ToTensor(),
            re,
        ]
    )
