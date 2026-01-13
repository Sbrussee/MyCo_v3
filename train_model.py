 #!/usr/bin/env python3
"""
pl_lemon_mocov3_continue.py

PyTorch Lightning (DDP) script to continue MoCo v3 training on 40x40 nuclei crops,
using LEMON a1+gray augmentations and rotation-without-corners protocol.

Inputs:
  --wsi_dir   directory containing WSIs
  --ann_dir   directory containing XML or GeoJSON annotations (same stem as WSI)
  --slide_labels  CSV or JSON mapping slide_id -> MF/BID (for probe)

Outputs:
  checkpoints + log JSONL
"""

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

import torchvision.transforms as T
import torchvision.transforms.functional as TF

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.strategies import DDPStrategy

# Optional deps
try:
    import openslide
except Exception:
    openslide = None

try:
    from lxml import etree
except Exception:
    etree = None

try:
    from shapely.geometry import shape
except Exception:
    shape = None

try:
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
except Exception:
    roc_auc_score = None
    balanced_accuracy_score = None

try:
    from reptrix.metrics import embedding_norms, embedding_variance  # type: ignore
    REPTRIX_AVAILABLE = True
except Exception:
    REPTRIX_AVAILABLE = False

try:
    import timm
except Exception:
    timm = None


# -------------------------
# Helpers
# -------------------------

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_slide_labels(path: str) -> Dict[str, int]:
    if path.lower().endswith(".json"):
        with open(path, "r") as f:
            d = json.load(f)
        out = {}
        for k, v in d.items():
            if isinstance(v, str):
                out[k] = 1 if v.strip().upper() == "MF" else 0
            else:
                out[k] = int(v)
        return out

    out = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = row["slide_id"]
            lab = row["label"].strip().upper()
            if lab in ["MF", "1", "TRUE", "T"]:
                out[sid] = 1
            elif lab in ["BID", "0", "FALSE", "F"]:
                out[sid] = 0
            else:
                out[sid] = int(float(lab))
    return out


def center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    left = int((w - size) / 2)
    top = int((h - size) / 2)
    return img.crop((left, top, left + size, top + size))


def safe_open_slide(wsi_path: str):
    if openslide is None:
        raise RuntimeError("openslide-python not available (and system OpenSlide missing).")
    return openslide.OpenSlide(wsi_path)


# -------------------------
# LEMON RotationCrop + a1+gray augmentations
# -------------------------

class RotationCrop40:
    """
    LEMON: rotate 0-360 without black-corner artifacts by:
      take larger crop (60x60), rotate, then center-crop to 40x40
    """
    def __init__(self, big_size: int = 60, out_size: int = 40, degrees: float = 360.0):
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
    """
    LEMON augmentation a1 + gray (as listed in supplement):
      - RotationCrop degree=360 (handled separately after sampling 60x60)
      - RandomResizedCrop scale=(0.32, 1.0)
      - RandomHorizontalFlip
      - ColorJitter (0.6, 0.7, 0.5, 0.2) with p=0.8
      - RandomGrayscale p=0.2
      - RandomErasing p=0.3 scale=(0.1,0.3) ratio=(0.8,1.2)
      - GaussianBlur sigma=(0.1,2.0)
    """
    rrc = T.RandomResizedCrop(size=img_size, scale=(0.32, 1.0), ratio=(3/4, 4/3),
                              interpolation=T.InterpolationMode.BILINEAR)
    cj = T.ColorJitter(brightness=0.6, contrast=0.7, saturation=0.5, hue=0.2)
    blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    re = T.RandomErasing(p=0.3, scale=(0.1, 0.3), ratio=(0.8, 1.2), value="random")

    return T.Compose([
        rrc,
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([cj], p=0.8),
        T.RandomGrayscale(p=0.2),
        blur,
        T.ToTensor(),
        re,
    ])


# -------------------------
# Annotation parsing (XML / GeoJSON)
# -------------------------

def parse_geojson_centroids(path: str) -> List[Tuple[float, float]]:
    if shape is None:
        raise RuntimeError("shapely required for GeoJSON parsing.")
    with open(path, "r") as f:
        gj = json.load(f)
    coords = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry", None)
        if geom is None:
            continue
        g = shape(geom)
        c = g if g.geom_type == "Point" else g.centroid
        coords.append((float(c.x), float(c.y)))
    return coords


def parse_xml_centroids(path: str) -> List[Tuple[float, float]]:
    if etree is None:
        raise RuntimeError("lxml required for XML parsing.")
    # Generic ASAP/QuPath-ish: collect Coordinate nodes
    tree = etree.parse(path)
    root = tree.getroot()
    coords = []
    for c in root.findall(".//Coordinate"):
        x = c.get("X") or c.get("x")
        y = c.get("Y") or c.get("y")
        if x is None or y is None:
            continue
        coords.append((float(x), float(y)))
    return coords


def load_centroids(path: str) -> List[Tuple[float, float]]:
    p = path.lower()
    if p.endswith(".geojson") or p.endswith(".json"):
        return parse_geojson_centroids(path)
    if p.endswith(".xml"):
        return parse_xml_centroids(path)
    raise ValueError(f"Unsupported annotation format: {path}")


# -------------------------
# Directory matching
# -------------------------

@dataclass
class SlideEntry:
    slide_id: str
    wsi_path: str
    ann_path: str


def build_entries_from_dirs(wsi_dir: str, ann_dir: str,
                            wsi_exts=(".svs", ".tif", ".tiff", ".ndpi", ".mrxs")) -> List[SlideEntry]:
    wsi_dir_p = Path(wsi_dir)
    ann_dir_p = Path(ann_dir)

    wsis = []
    for ext in wsi_exts:
        wsis.extend(wsi_dir_p.glob(f"*{ext}"))
    if not wsis:
        raise RuntimeError(f"No WSIs found in {wsi_dir} with extensions {wsi_exts}")

    # Map stem -> wsi
    wsi_map = {p.stem: p for p in wsis}

    # annotations: xml or geojson
    anns = list(ann_dir_p.glob("*.xml")) + list(ann_dir_p.glob("*.geojson")) + list(ann_dir_p.glob("*.json"))
    ann_map = {p.stem: p for p in anns}

    entries = []
    for stem, wsi_path in wsi_map.items():
        if stem not in ann_map:
            continue
        entries.append(SlideEntry(slide_id=stem, wsi_path=str(wsi_path), ann_path=str(ann_map[stem])))

    if not entries:
        raise RuntimeError("No matched WSI/annotation pairs by stem name.")
    return entries


# -------------------------
# Iterable dataset for MoCo (two augmented views)
# -------------------------

class WSICellMoCoIterable(IterableDataset):
    """
    Iterable dataset: yields (view1, view2) nucleus crops.
    Implements per-rank sharding for DDP by only sampling from a subset of slides.
    """
    def __init__(
        self,
        entries: List[SlideEntry],
        epoch_length: int,
        seed: int,
        out_size: int = 40,
        big_size: int = 60,
    ):
        super().__init__()
        self.all_entries = entries
        self.epoch_length = epoch_length
        self.seed = seed
        self.out_size = out_size
        self.big_size = big_size

        self.rotcrop = RotationCrop40(big_size=big_size, out_size=out_size, degrees=360.0)
        self.aug = build_lemon_a1_gray_transform(img_size=out_size)

        # preload centroids (consider caching in practice)
        self.centroids: Dict[str, List[Tuple[float, float]]] = {}
        for e in entries:
            self.centroids[e.slide_id] = load_centroids(e.ann_path)

    def _read_patch(self, slide, cx: float, cy: float, size: int) -> Image.Image:
        half = size // 2
        x0 = int(round(cx - half))
        y0 = int(round(cy - half))
        return slide.read_region((x0, y0), 0, (size, size)).convert("RGB")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        # DDP shard
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        # shard slides by rank first
        entries = self.all_entries[rank::world_size]
        if not entries:
            entries = self.all_entries  # fallback

        rng = random.Random(self.seed + 1000 * rank + 10 * worker_id)

        # stride over epoch_length across workers
        n_yield = self.epoch_length // (world_size * num_workers)
        for _ in range(n_yield):
            e = rng.choice(entries)
            cents = self.centroids.get(e.slide_id, [])
            if not cents:
                continue
            cx, cy = rng.choice(cents)
            slide = safe_open_slide(e.wsi_path)
            try:
                big = self._read_patch(slide, cx, cy, self.big_size)  # 60x60
            finally:
                slide.close()

            img40 = self.rotcrop(big)
            v1 = self.aug(img40)
            v2 = self.aug(img40)
            yield v1, v2


class CellDataModule(pl.LightningDataModule):
    def __init__(self, entries: List[SlideEntry], epoch_length: int, batch_size: int,
                 num_workers: int, seed: int):
        super().__init__()
        self.entries = entries
        self.epoch_length = epoch_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def train_dataloader(self):
        ds = WSICellMoCoIterable(self.entries, self.epoch_length, seed=self.seed)
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, drop_last=True)


# -------------------------
# MoCo v3 LightningModule
# -------------------------

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return x


class MoCoV3Lit(pl.LightningModule):
    def __init__(self, init_ckpt: str, lr: float, weight_decay: float, temperature: float,
                 proj_dim: int, mlp_hidden: int, base_m: float, epochs: int):
        super().__init__()
        if timm is None:
            raise RuntimeError("timm required.")
        self.save_hyperparameters(ignore=["init_ckpt"])

        self.temperature = temperature
        self.base_m = base_m
        self.total_epochs = epochs

        # ViT-S/8 backbone (timm name); handles variable res via pos-embed interp
        self.q_enc = timm.create_model("vit_small_patch8_224", pretrained=False, num_classes=0, global_pool="avg")
        dim = self.q_enc.num_features
        self.q_proj = MLPHead(dim, mlp_hidden, proj_dim)

        self.k_enc = timm.create_model("vit_small_patch8_224", pretrained=False, num_classes=0, global_pool="avg")
        self.k_proj = MLPHead(dim, mlp_hidden, proj_dim)

        # init teacher = student
        self._copy_q_to_k()

        if init_ckpt and os.path.isfile(init_ckpt):
            ck = torch.load(init_ckpt, map_location="cpu")
            state = ck.get("state_dict", ck)
            missing, unexpected = self.load_state_dict(state, strict=False)
            print(f"[init] loaded {init_ckpt} missing={len(missing)} unexpected={len(unexpected)}")

    @torch.no_grad()
    def _copy_q_to_k(self):
        for p_q, p_k in zip(self.q_enc.parameters(), self.k_enc.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False
        for p_q, p_k in zip(self.q_proj.parameters(), self.k_proj.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self, m: float):
        for p_q, p_k in zip(self.q_enc.parameters(), self.k_enc.parameters()):
            p_k.data = p_k.data * m + p_q.data * (1.0 - m)
        for p_q, p_k in zip(self.q_proj.parameters(), self.k_proj.parameters()):
            p_k.data = p_k.data * m + p_q.data * (1.0 - m)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return opt

    def _momentum_schedule(self) -> float:
        # cosine schedule over training progress
        # (This is common practice; LEMON states the base m used via MoCo v3 implementation reference.)
        progress = self.current_epoch / max(1, (self.total_epochs - 1))
        return 1.0 - (1.0 - self.base_m) * (0.5 * (1.0 + math.cos(math.pi * progress)))

    def training_step(self, batch, batch_idx):
        x1, x2 = batch
        m = self._momentum_schedule()
        with torch.no_grad():
            self._momentum_update(m)

        q1 = F.normalize(self.q_proj(self.q_enc(x1)), dim=1)
        q2 = F.normalize(self.q_proj(self.q_enc(x2)), dim=1)
        with torch.no_grad():
            k1 = F.normalize(self.k_proj(self.k_enc(x1)), dim=1)
            k2 = F.normalize(self.k_proj(self.k_enc(x2)), dim=1)

        logits_12 = (q1 @ k2.t()) / self.temperature
        logits_21 = (q2 @ k1.t()) / self.temperature
        labels = torch.arange(logits_12.size(0), device=logits_12.device)

        loss = 0.5 * (F.cross_entropy(logits_12, labels) + F.cross_entropy(logits_21, labels))
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.q_proj(self.q_enc(x))
        return z


# -------------------------
# Probe + repr eval callback (epoch end)
# -------------------------

class AttentionPool(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(in_dim, attn_dim)
        self.U = nn.Linear(in_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        A = torch.tanh(self.V(H)) * torch.sigmoid(self.U(H))
        A = self.w(A).squeeze(1)
        A = torch.softmax(A, dim=0)
        Z = torch.sum(A.unsqueeze(1) * H, dim=0)
        return Z


def repr_metrics_np(emb: np.ndarray) -> Dict[str, float]:
    if REPTRIX_AVAILABLE:
        t = torch.from_numpy(emb).float()
        return {
            "embedding_variance": float(embedding_variance(t).item()),
            "mean_embedding_norm": float(embedding_norms(t).mean().item()),
        }
    return {
        "embedding_variance": float(np.mean(np.var(emb, axis=0))),
        "mean_embedding_norm": float(np.mean(np.linalg.norm(emb, axis=1))),
    }


class EvalCallback(Callback):
    def __init__(self, entries: List[SlideEntry], slide_labels: Dict[str, int],
                 cells_per_slide: int = 200, probe_epochs: int = 20, probe_lr: float = 1e-3,
                 seed: int = 0):
        super().__init__()
        self.entries = entries
        self.slide_labels = slide_labels
        self.cells_per_slide = cells_per_slide
        self.probe_epochs = probe_epochs
        self.probe_lr = probe_lr
        self.seed = seed

        self.rotcrop = RotationCrop40(big_size=60, out_size=40, degrees=360.0)
        self.totensor = T.ToTensor()

    @torch.no_grad()
    def _embed_slide(self, pl_module: MoCoV3Lit, device: torch.device, entry: SlideEntry,
                     centroids: List[Tuple[float, float]], rng: random.Random) -> Optional[np.ndarray]:
        if not centroids:
            return None
        slide = safe_open_slide(entry.wsi_path)
        try:
            embs = []
            for _ in range(self.cells_per_slide):
                cx, cy = rng.choice(centroids)
                half = 30
                patch = slide.read_region((int(cx - half), int(cy - half)), 0, (60, 60)).convert("RGB")
                patch = self.rotcrop(patch)
                x = self.totensor(patch).unsqueeze(0).to(device)
                z = pl_module.encode(x).squeeze(0).detach().cpu().numpy()
                embs.append(z)
            return np.stack(embs, axis=0)
        finally:
            slide.close()

    def _train_probe(self, embeds: Dict[str, np.ndarray], labels: Dict[str, int],
                     device: torch.device) -> Dict[str, float]:
        keys = list(embeds.keys())
        if len(keys) < 4:
            return {"probe_auc": float("nan"), "probe_bal_acc": float("nan")}

        rng = np.random.default_rng(self.seed)
        rng.shuffle(keys)
        split = int(0.8 * len(keys))
        tr, va = keys[:split], keys[split:]

        d = embeds[keys[0]].shape[1]
        attn = AttentionPool(d, 128).to(device)
        clf = nn.Linear(d, 1).to(device)
        opt = torch.optim.AdamW(list(attn.parameters()) + list(clf.parameters()), lr=self.probe_lr, weight_decay=1e-4)

        for _ in range(self.probe_epochs):
            attn.train(); clf.train()
            for sid in tr:
                H = torch.from_numpy(embeds[sid]).float().to(device)
                y = torch.tensor([labels[sid]], dtype=torch.float32, device=device)
                z = attn(H)
                logit = clf(z).squeeze(0)
                loss = F.binary_cross_entropy_with_logits(logit, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        attn.eval(); clf.eval()
        y_true, y_score = [], []
        with torch.no_grad():
            for sid in va:
                H = torch.from_numpy(embeds[sid]).float().to(device)
                y = float(labels[sid])
                z = attn(H)
                p = torch.sigmoid(clf(z)).item()
                y_true.append(y); y_score.append(p)

        out = {}
        if roc_auc_score is not None and len(set(y_true)) > 1:
            out["probe_auc"] = float(roc_auc_score(y_true, y_score))
        else:
            out["probe_auc"] = float("nan")
        if balanced_accuracy_score is not None:
            y_pred = [1 if s >= 0.5 else 0 for s in y_score]
            out["probe_bal_acc"] = float(balanced_accuracy_score(y_true, y_pred))
        else:
            out["probe_bal_acc"] = float("nan")
        return out

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: MoCoV3Lit):
        # Only run on rank 0 to avoid duplicating heavy eval
        if trainer.global_rank != 0:
            return

        device = pl_module.device
        rng = random.Random(self.seed + trainer.current_epoch)

        # sample embeddings for repr metrics
        # take a few batches from train loader (already augmented) but use encoded embeddings
        embs = []
        dl = trainer.train_dataloader
        it = iter(dl)
        for _ in range(10):
            try:
                x1, _ = next(it)
            except StopIteration:
                break
            x1 = x1.to(device)
            z = pl_module.encode(x1).detach().cpu().numpy()
            embs.append(z)
        if embs:
            emb_np = np.concatenate(embs, axis=0)
            rm = repr_metrics_np(emb_np)
        else:
            rm = {"embedding_variance": float("nan"), "mean_embedding_norm": float("nan")}

        # slide-level probe
        # load centroids per slide (consider caching in production)
        embeds_by_slide = {}
        labs = {}
        for e in self.entries:
            if e.slide_id not in self.slide_labels:
                continue
            cents = load_centroids(e.ann_path)
            H = self._embed_slide(pl_module, device, e, cents, rng)
            if H is None:
                continue
            embeds_by_slide[e.slide_id] = H
            labs[e.slide_id] = int(self.slide_labels[e.slide_id])

        pm = self._train_probe(embeds_by_slide, labs, device=device)

        trainer.logger.log_metrics({**rm, **pm}, step=trainer.global_step)
        print(f"[eval@epoch={trainer.current_epoch}] "
              f"var={rm['embedding_variance']:.6f} norm={rm['mean_embedding_norm']:.3f} "
              f"auc={pm.get('probe_auc', float('nan')):.3f} bal_acc={pm.get('probe_bal_acc', float('nan')):.3f}")


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi_dir", required=True)
    ap.add_argument("--ann_dir", required=True)
    ap.add_argument("--slide_labels", required=True)

    ap.add_argument("--outdir", required=True)
    ap.add_argument("--init_ckpt", default="")

    # LEMON reference defaults: 150 epochs, epoch length 1M images, batch 4096 :contentReference[oaicite:8]{index=8}
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--epoch_length", type=int, default=1_000_000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--accum", type=int, default=16, help="accumulate_grad_batches (for effective batch)")

    # LR adaptation: "somewhat lower LR" under compute constraints
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--m", type=float, default=0.99)

    ap.add_argument("--proj_dim", type=int, default=256)
    ap.add_argument("--mlp_hidden", type=int, default=2048)

    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--precision", default="bf16-mixed")  # bf16 recommended if available
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)

    # Probe/eval
    ap.add_argument("--probe_cells_per_slide", type=int, default=200)
    ap.add_argument("--probe_epochs", type=int, default=20)
    ap.add_argument("--probe_lr", type=float, default=1e-3)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    seed_all(args.seed)

    entries = build_entries_from_dirs(args.wsi_dir, args.ann_dir)
    slide_labels = read_slide_labels(args.slide_labels)

    dm = CellDataModule(entries, epoch_length=args.epoch_length, batch_size=args.batch_size,
                        num_workers=args.num_workers, seed=args.seed)

    model = MoCoV3Lit(
        init_ckpt=args.init_ckpt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        proj_dim=args.proj_dim,
        mlp_hidden=args.mlp_hidden,
        base_m=args.m,
        epochs=args.epochs,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=args.outdir,
        filename="moco-{epoch:03d}",
        save_top_k=-1,
        every_n_epochs=1,
        save_last=True,
    )

    eval_cb = EvalCallback(entries, slide_labels,
                           cells_per_slide=args.probe_cells_per_slide,
                           probe_epochs=args.probe_epochs,
                           probe_lr=args.probe_lr,
                           seed=args.seed)

    trainer = pl.Trainer(
        default_root_dir=args.outdir,
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=False) if args.devices > 1 else "auto",
        precision=args.precision,
        accumulate_grad_batches=args.accum,
        callbacks=[ckpt_cb, eval_cb],
        log_every_n_steps=50,
        enable_checkpointing=True,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
