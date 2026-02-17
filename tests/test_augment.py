import pytest
from pathlib import Path
from PIL import Image

from myco.augment import (
    _to_tensor_rgb,
    apply_lemon_a1_gray_with_params,
    build_lemon_a1_gray_transform,
    save_augmentation_examples,
)


def test_apply_lemon_a1_gray_with_params_includes_valid_gaussian_sigma() -> None:
    img_size = 40
    image = Image.new("RGB", (img_size, img_size), color=(128, 64, 32))

    tensor, params = apply_lemon_a1_gray_with_params(image, img_size=img_size, seed=3)

    assert tensor.shape == (3, img_size, img_size)
    blur_step = next(step for step in params if step["name"] == "gaussian_blur")
    sigma = blur_step["params"]["sigma"]
    assert 0.1 <= sigma <= 2.0
    assert all(step["name"] != "random_erasing" for step in params)


def test_save_augmentation_examples_writes_artifacts_and_params(tmp_path: Path) -> None:
    img_size = 40
    image = Image.new("RGB", (img_size, img_size), color=(10, 20, 30))

    save_augmentation_examples(image, output_dir=tmp_path, img_size=img_size, seed=3)

    assert (tmp_path / "params.json").exists()
    assert (tmp_path / "gaussian_blur.png").exists()


def test_to_tensor_rgb_preserves_channel_order() -> None:
    image = Image.new("RGB", (1, 1), color=(10, 20, 30))

    tensor = _to_tensor_rgb(image)

    assert tensor.shape == (3, 1, 1)
    assert float(tensor[0, 0, 0]) == pytest.approx(10.0 / 255.0)
    assert float(tensor[1, 0, 0]) == pytest.approx(20.0 / 255.0)
    assert float(tensor[2, 0, 0]) == pytest.approx(30.0 / 255.0)


def test_lemon_a1_gray_transform_uses_requested_parameters() -> None:
    transform = build_lemon_a1_gray_transform(img_size=40)

    rrc = transform.transforms[0]
    random_apply = transform.transforms[2]
    gray = transform.transforms[3]
    blur = transform.transforms[4]

    assert tuple(rrc.scale) == pytest.approx((0.32, 1.0))
    jitter = random_apply.transforms[0]
    assert random_apply.p == pytest.approx(0.8)
    assert jitter.brightness == pytest.approx((0.4, 1.6))
    assert jitter.contrast == pytest.approx((0.3, 1.7))
    assert jitter.saturation == pytest.approx((0.5, 1.5))
    assert jitter.hue == pytest.approx((-0.2, 0.2))
    assert gray.p == pytest.approx(0.2)
    assert tuple(blur.sigma) == pytest.approx((0.1, 2.0))
