"""Utility helpers for reproducibility and image processing."""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def center_crop(img: Image.Image, size: int) -> Image.Image:
    """Return the centered square crop of ``size`` pixels from ``img``."""
    width, height = img.size
    left = int((width - size) / 2)
    top = int((height - size) / 2)
    return img.crop((left, top, left + size, top + size))


def read_patch(slide, center: Tuple[float, float], size: int) -> Image.Image:
    """Read a square patch from an OpenSlide object around ``center``."""
    cx, cy = center
    half = size // 2
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    return slide.read_region((x0, y0), 0, (size, size)).convert("RGB")
