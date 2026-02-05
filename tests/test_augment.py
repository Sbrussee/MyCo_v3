from pathlib import Path
import random

from PIL import Image

from myco.augment import apply_lemon_a1_gray_with_params, save_augmentation_examples


def test_apply_lemon_a1_gray_with_params_includes_valid_gaussian_sigma() -> None:
    img_size = 40
    image = Image.new("RGB", (img_size, img_size), color=(128, 64, 32))

    tensor, params = apply_lemon_a1_gray_with_params(image, img_size=img_size, seed=3)

    assert tensor.shape == (3, img_size, img_size)
    blur_step = next(step for step in params if step["name"] == "gaussian_blur")
    sigma = blur_step["params"]["sigma"]
    assert 0.1 <= sigma <= 2.0


def test_save_augmentation_examples_writes_artifacts_and_params(tmp_path: Path) -> None:
    img_size = 40
    image = Image.new("RGB", (img_size, img_size), color=(10, 20, 30))

    save_augmentation_examples(image, output_dir=tmp_path, img_size=img_size, seed=3)

    assert (tmp_path / "params.json").exists()
    assert (tmp_path / "gaussian_blur.png").exists()


def test_random_erasing_uses_compatible_random_value_api(monkeypatch) -> None:
    """Ensure random erasing works when using ``value='random'`` behavior."""
    img_size = 40
    image = Image.new("RGB", (img_size, img_size), color=(128, 64, 32))

    # Force every stochastic branch to execute, including random erasing.
    monkeypatch.setattr(random, "random", lambda: 0.0)

    tensor, params = apply_lemon_a1_gray_with_params(image, img_size=img_size, seed=7)

    erase_step = next(step for step in params if step["name"] == "random_erasing")
    assert erase_step["params"]["applied"] is True
    assert tensor.shape == (3, img_size, img_size)
