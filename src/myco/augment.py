"""Augmentation policies aligned with LEMON MoCo v3 training."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .utils import center_crop


class RotationCrop40:
    """Rotate without black corners using a larger crop then center-crop."""

    def __init__(
        self, big_size: int = 60, out_size: int = 40, degrees: float = 360.0
    ) -> None:
        assert big_size > 0, "big_size must be positive."
        assert out_size > 0, "out_size must be positive."
        assert degrees > 0, "degrees must be positive."
        self.big_size = big_size
        self.out_size = out_size
        self.degrees = degrees

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.size != (self.big_size, self.big_size):
            img = img.resize((self.big_size, self.big_size), resample=Image.BICUBIC)
        angle = random.random() * self.degrees
        img = TF.rotate(
            img, angle=angle, interpolation=TF.InterpolationMode.BILINEAR, expand=False
        )
        return center_crop(img, self.out_size)


def build_lemon_a1_gray_transform(img_size: int = 40) -> T.Compose:
    """Return the LEMON a1 + gray augmentation pipeline."""
    assert img_size > 0, "img_size must be positive."
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


def _apply_color_jitter(
    img: Image.Image,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
    fn_idx: Tuple[int, ...],
) -> Image.Image:
    for fn_id in fn_idx:
        if fn_id == 0 and brightness is not None:
            img = TF.adjust_brightness(img, brightness)
        elif fn_id == 1 and contrast is not None:
            img = TF.adjust_contrast(img, contrast)
        elif fn_id == 2 and saturation is not None:
            img = TF.adjust_saturation(img, saturation)
        elif fn_id == 3 and hue is not None:
            img = TF.adjust_hue(img, hue)
    return img


def _sample_gaussian_sigma(sigma_min: float = 0.1, sigma_max: float = 2.0) -> float:
    """Sample a Gaussian blur sigma using the active torch RNG state.

    Parameters
    ----------
    sigma_min : float
        Lower bound for sigma in pixel units.
    sigma_max : float
        Upper bound for sigma in pixel units.

    Returns
    -------
    float
        A sampled sigma value in ``[sigma_min, sigma_max]``.
    """
    assert sigma_min > 0.0, "sigma_min must be positive."
    assert sigma_max >= sigma_min, "sigma_max must be >= sigma_min."
    return float(torch.empty(1).uniform_(sigma_min, sigma_max).item())


def apply_lemon_a1_gray_with_params(
    img: Image.Image,
    img_size: int,
    seed: int,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Apply LEMON a1 + gray augmentations with deterministic parameters.

    Parameters
    ----------
    img : Image.Image
        Input PIL image. Expected size is (img_size, img_size).
    img_size : int
        Output spatial size (H=W=img_size).
    seed : int
        Seed for deterministic augmentation parameters.

    Returns
    -------
    tensor : torch.Tensor
        Augmented tensor with shape (3, img_size, img_size) in [0, 1].
    params : list[dict]
        Per-step parameters describing the augmentation pipeline.
    """
    assert img_size > 0, "img_size must be positive."
    assert isinstance(seed, int), "seed must be an integer."
    assert img.size == (img_size, img_size), (
        f"Expected img size {(img_size, img_size)}, got {img.size}."
    )
    random.seed(seed)
    torch.manual_seed(seed)

    params: List[Dict[str, Any]] = []

    i, j, h, w = T.RandomResizedCrop.get_params(
        img, scale=(0.32, 1.0), ratio=(3 / 4, 4 / 3)
    )
    img = TF.resized_crop(
        img,
        i,
        j,
        h,
        w,
        size=[img_size, img_size],
        interpolation=TF.InterpolationMode.BILINEAR,
    )
    params.append(
        {
            "name": "random_resized_crop",
            "params": {
                "i": int(i),
                "j": int(j),
                "h": int(h),
                "w": int(w),
                "size": img_size,
            },
        }
    )

    flip_p = 0.5
    do_flip = random.random() < flip_p
    if do_flip:
        img = TF.hflip(img)
    params.append(
        {
            "name": "random_horizontal_flip",
            "params": {"p": flip_p, "applied": do_flip},
        }
    )

    apply_jitter = random.random() < 0.8
    if apply_jitter:
        fn_idx, b, c, s, h = T.ColorJitter.get_params(
            brightness=(0.4, 1.6),
            contrast=(0.3, 1.7),
            saturation=(0.5, 1.5),
            hue=(-0.2, 0.2),
        )
        img = _apply_color_jitter(img, b, c, s, h, fn_idx)
        params.append(
            {
                "name": "color_jitter",
                "params": {
                    "applied": True,
                    "fn_idx": [int(idx) for idx in fn_idx],
                    "brightness": float(b),
                    "contrast": float(c),
                    "saturation": float(s),
                    "hue": float(h),
                },
            }
        )
    else:
        params.append({"name": "color_jitter", "params": {"applied": False}})

    gray_p = 0.2
    do_gray = random.random() < gray_p
    if do_gray:
        img = TF.rgb_to_grayscale(img, num_output_channels=3)
    params.append(
        {"name": "random_grayscale", "params": {"p": gray_p, "applied": do_gray}}
    )

    sigma = _sample_gaussian_sigma(0.1, 2.0)
    img = TF.gaussian_blur(img, kernel_size=[3, 3], sigma=sigma)
    params.append({"name": "gaussian_blur", "params": {"sigma": float(sigma)}})

    tensor = TF.to_tensor(img)
    params.append({"name": "to_tensor", "params": {"dtype": str(tensor.dtype)}})

    erase_p = 0.3
    do_erase = random.random() < erase_p
    if do_erase:
        i, j, h, w, v = T.RandomErasing.get_params(
            tensor, scale=(0.1, 0.3), ratio=(0.8, 1.2), value="random"
        )
        tensor = TF.erase(tensor, i, j, h, w, v, inplace=False)
        params.append(
            {
                "name": "random_erasing",
                "params": {
                    "p": erase_p,
                    "applied": True,
                    "i": int(i),
                    "j": int(j),
                    "h": int(h),
                    "w": int(w),
                },
            }
        )
    else:
        params.append(
            {"name": "random_erasing", "params": {"p": erase_p, "applied": False}}
        )

    return tensor, params


def save_augmentation_examples(
    img: Image.Image,
    output_dir: Path,
    img_size: int,
    seed: int,
) -> None:
    """Save per-augmentation examples and parameters for inspection.

    Saves an input image, each augmentation output, and a params.json describing
    the parameters used. Uses fixed seeds for reproducibility.
    """
    assert isinstance(output_dir, Path), "output_dir must be a Path."
    assert img_size > 0, "img_size must be positive."
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = output_dir / "input.png"
    img.save(input_path)

    params: List[Dict[str, Any]] = []
    before_path = input_path

    def _save_step(name: str, img_out: Image.Image) -> Path:
        path = output_dir / f"{name}.png"
        img_out.save(path)
        return path

    random.seed(seed)
    torch.manual_seed(seed)

    i, j, h, w = T.RandomResizedCrop.get_params(
        img, scale=(0.32, 1.0), ratio=(3 / 4, 4 / 3)
    )
    img = TF.resized_crop(
        img,
        i,
        j,
        h,
        w,
        size=[img_size, img_size],
        interpolation=TF.InterpolationMode.BILINEAR,
    )
    after_path = _save_step("random_resized_crop", img)
    params.append(
        {
            "name": "random_resized_crop",
            "before": str(before_path),
            "after": str(after_path),
            "params": {
                "i": int(i),
                "j": int(j),
                "h": int(h),
                "w": int(w),
                "size": img_size,
            },
        }
    )
    before_path = after_path

    flip_p = 0.5
    do_flip = random.random() < flip_p
    if do_flip:
        img = TF.hflip(img)
    after_path = _save_step("random_horizontal_flip", img)
    params.append(
        {
            "name": "random_horizontal_flip",
            "before": str(before_path),
            "after": str(after_path),
            "params": {"p": flip_p, "applied": do_flip},
        }
    )
    before_path = after_path

    apply_jitter = random.random() < 0.8
    if apply_jitter:
        fn_idx, b, c, s, h = T.ColorJitter.get_params(
            brightness=(0.4, 1.6),
            contrast=(0.3, 1.7),
            saturation=(0.5, 1.5),
            hue=(-0.2, 0.2),
        )
        img = _apply_color_jitter(img, b, c, s, h, fn_idx)
        jitter_params: Dict[str, Any] = {
            "applied": True,
            "fn_idx": [int(idx) for idx in fn_idx],
            "brightness": float(b),
            "contrast": float(c),
            "saturation": float(s),
            "hue": float(h),
        }
    else:
        jitter_params = {"applied": False}
    after_path = _save_step("color_jitter", img)
    params.append(
        {
            "name": "color_jitter",
            "before": str(before_path),
            "after": str(after_path),
            "params": jitter_params,
        }
    )
    before_path = after_path

    gray_p = 0.2
    do_gray = random.random() < gray_p
    if do_gray:
        img = TF.rgb_to_grayscale(img, num_output_channels=3)
    after_path = _save_step("random_grayscale", img)
    params.append(
        {
            "name": "random_grayscale",
            "before": str(before_path),
            "after": str(after_path),
            "params": {"p": gray_p, "applied": do_gray},
        }
    )
    before_path = after_path

    sigma = _sample_gaussian_sigma(0.1, 2.0)
    img = TF.gaussian_blur(img, kernel_size=[3, 3], sigma=sigma)
    after_path = _save_step("gaussian_blur", img)
    params.append(
        {
            "name": "gaussian_blur",
            "before": str(before_path),
            "after": str(after_path),
            "params": {"sigma": float(sigma)},
        }
    )
    before_path = after_path

    tensor = TF.to_tensor(img)
    tensor_path = output_dir / "to_tensor.png"
    TF.to_pil_image(tensor).save(tensor_path)
    params.append(
        {
            "name": "to_tensor",
            "before": str(before_path),
            "after": str(tensor_path),
            "params": {"dtype": str(tensor.dtype)},
        }
    )
    before_path = tensor_path

    erase_p = 0.3
    do_erase = random.random() < erase_p
    if do_erase:
        i, j, h, w, v = T.RandomErasing.get_params(
            tensor, scale=(0.1, 0.3), ratio=(0.8, 1.2), value="random"
        )
        tensor = TF.erase(tensor, i, j, h, w, v, inplace=False)
        erase_params = {
            "p": erase_p,
            "applied": True,
            "i": int(i),
            "j": int(j),
            "h": int(h),
            "w": int(w),
        }
    else:
        erase_params = {"p": erase_p, "applied": False}
    erase_path = output_dir / "random_erasing.png"
    TF.to_pil_image(tensor).save(erase_path)
    params.append(
        {
            "name": "random_erasing",
            "before": str(before_path),
            "after": str(erase_path),
            "params": erase_params,
        }
    )

    payload = {"seed": seed, "img_size": img_size, "steps": params}
    with open(output_dir / "params.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
