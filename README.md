# MyCo v3

This repository trains a MoCo v3 vision transformer on 40x40 nuclei crops extracted from whole-slide images (WSIs). It includes:

- Self-supervised MoCo v3 training with LEMON-style augmentations.
- Linear-probe evaluation on a **balanced subset of slides (150 BID / 150 MF)** using cached embeddings.
- Automatic selection of the **best epoch by probe balanced accuracy**, with weights saved for inference.
- UMAP/TSNE patch mosaic plots of nuclei after every epoch.
- A dedicated inference script for embedding extraction.

## Installation (uv-friendly)

> **Note**: `openslide-python` requires the system OpenSlide library. On Ubuntu, install it via:
>
> ```bash
> sudo apt-get update && sudo apt-get install -y openslide-tools
> ```

Use `uv` to create an isolated environment and sync dependencies:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

For tests:

```bash
uv pip install -e ".[test]"
```

## Quick start

### Train

```bash
myco-train \
  --wsi_dir /path/to/wsis \
  --ann_dir /path/to/annotations \
  --slide_labels /path/to/slide_labels.csv \
  --outdir /path/to/output
```

### Train on Slurm (sbatch)

Use the provided sbatch script to capture **progress**, **metrics**, and **performance**
logs during SSL training. The script writes:

- `OUTDIR/logs/*/metrics.csv` via the PyTorch Lightning CSV logger.
- `OUTDIR/train_<jobid>.log` with stdout progress + evaluation metrics.
- `OUTDIR/moco-*.ckpt` and `OUTDIR/last.ckpt` checkpoints.

```bash
export WSI_DIR=/path/to/wsis
export ANN_DIR=/path/to/annotations
export SLIDE_LABELS=/path/to/slide_labels.csv
export OUTDIR=/path/to/output

sbatch scripts/train_myco.sbatch
```

### Inference

```bash
myco-infer \
  --wsi_dir /path/to/wsis \
  --ann_dir /path/to/annotations \
  --weights /path/to/output/best_probe_weights.pt \
  --out_dir /path/to/output/embeddings
```

## Data layout

Expected directory layout:

```
wsis/
  slide_001.svs
  slide_002.svs
annotations/
  slide_001.xml
  slide_002.geojson
labels.csv
```

`labels.csv` must include `slide_id` and `label` columns, where labels are `BID` or `MF` (or 0/1).

## Required inputs (format)

These inputs are required for both `myco-train` and `scripts/train_myco.sbatch`:

- **WSI directory (`--wsi_dir` / `WSI_DIR`)**: contains WSI files like `slide_001.svs`.
- **Annotation directory (`--ann_dir` / `ANN_DIR`)**: contains one `.xml` or `.geojson`
  per slide, with filenames matching the WSI slide IDs (e.g., `slide_001.xml`).
- **Slide labels CSV (`--slide_labels` / `SLIDE_LABELS`)**: CSV with columns
  `slide_id,label`, where labels are `BID`/`MF` or `0`/`1`.
- **Output directory (`--outdir` / `OUTDIR`)**: destination for checkpoints, metrics,
  mosaics, and logs.

## Training details

### Augmentations

Training uses LEMON a1+gray augmentations:

- Random resized crop
- Horizontal flip
- Color jitter
- Random grayscale
- Gaussian blur
- Random erasing
- Rotation without black corners (60x60 rotate -> center crop to 40x40)

### Key hyperparameters

| Flag | Description | Default |
| --- | --- | --- |
| `--epochs` | Number of training epochs | `150` |
| `--epoch_length` | Number of crops per epoch | `1_000_000` |
| `--batch_size` | Batch size per device | `256` |
| `--accum` | Gradient accumulation steps | `16` |
| `--lr` | AdamW learning rate | `2.5e-4` |
| `--weight_decay` | AdamW weight decay | `0.05` |
| `--temperature` | MoCo temperature | `0.2` |
| `--m` | Base momentum coefficient | `0.99` |
| `--proj_dim` | Projection head output dimension | `256` |
| `--mlp_hidden` | Projection head hidden dimension | `2048` |
| `--precision` | Lightning precision | `bf16-mixed` |

### Linear probe & best checkpointing

After each epoch, the evaluation callback:

1. Samples a **balanced slide subset** (`--probe_slides_per_class`, default 150 per class).
2. Extracts embeddings **once per epoch** for `--probe_cells_per_slide` nuclei per slide.
3. Reuses cached embeddings to train a slide-level attention-pooled linear probe.
4. Saves the **best epoch weights** (by probe balanced accuracy) to:

```
<outdir>/best_probe_weights.pt
```

This file includes encoder + projection weights and can be used directly by the inference script.

### Patch mosaic plots

For each epoch, a patch mosaic plot is saved to:

```
<outdir>/mosaic_epoch_XXX.png
```

Use `--mosaic_method umap` (default) or `--mosaic_method tsne` to control the projection method.

### Distributed training (DDP)

PyTorch Lightning handles DDP when multiple devices are specified. Example (4 GPUs):

```bash
myco-train \
  --wsi_dir /path/to/wsis \
  --ann_dir /path/to/annotations \
  --slide_labels /path/to/labels.csv \
  --outdir /path/to/output \
  --devices 4
```

The iterable dataset shards slides by rank. Make sure your `epoch_length` and `batch_size` produce a reasonable number of steps per epoch.

### Recommended workflow

1. Start with a short sanity run (`--epochs 2 --epoch_length 10000`) to confirm data paths.
2. Scale to your target epoch length and devices.
3. Monitor `probe_bal_acc` metrics to find the best epoch.
4. Use `best_probe_weights.pt` for inference embedding extraction.

## Testing

```bash
pytest
```

## Repository structure

```
src/
  myco/
    augment.py
    data.py
    eval.py
    inference.py
    model.py
    utils.py
    visualization.py
train_model.py
```

## Notes

- Ensure that OpenSlide is installed on your system before installing `openslide-python`.
- The model uses ViT-S/8 from `timm` and expects 40x40 crops.
