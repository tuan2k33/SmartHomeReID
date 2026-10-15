"""
Pet Re-ID training — SKELETON / reference implementation of petreid.md.

Not production-ready. Assumes dataset is ALREADY cropped/preprocessed into:
    dataset/train/{identity_id}/*.jpg   (>=2 images per identity, ideally >=4-8)

The crop pipeline (detector + pad + quality filter, vpet.md section 5.2) is a
separate preprocessing step and is NOT implemented here — run it before using
this script. Eval protocol (query/gallery Rank-1/mAP, section 6) lives in
eval_utils.py / eval.py.
"""

import csv
import os
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image
from transformers import AutoModel

from torch.amp import autocast, GradScaler

from eval_utils import EvalSet, evaluate, split_val

# --------------------------------------------------------------------------
# Config (edit directly — this is a skeleton, not a CLI tool)
# Values reproduce the earlier 0.9442-mAP run; tuning history is in git.
# --------------------------------------------------------------------------
DATA_ROOT = "data/dataset/train"
DATASET_ROOT = "data/dataset"  # query/gallery test split, used only for the final eval
BACKBONE_PATH = "models/dinov3-vitb16-pretrain-lvd1689m"
IMG_SIZE = 224                       # multiple of patch size (16 here)

# Validation: this fraction of the train identities is held out (seeded). For each one,
# the first image is a query and the rest are gallery. It picks best.pt and drives early stop.
VAL_FRACTION = 0.1
VAL_SEED = 0

# --- augmentation (sup_transform below) ---
CROP_SCALE_MIN = 0.5                 # RandomResizedCrop lower bound (fraction of original area)
CROP_SCALE_MAX = 1.0
COLORJITTER_BRIGHTNESS = 0.3
COLORJITTER_CONTRAST = 0.3
COLORJITTER_SATURATION = 0.3
COLORJITTER_HUE = 0.08
GRAYSCALE_P = 0.30                   # proxy for IR-camera night captures, not for low-light colour shots
ERASING_P = 0.3
ERASING_SCALE_MIN = 0.02
ERASING_SCALE_MAX = 0.15
UNFREEZE_BLOCK_INDICES = [11]
UNFREEZE_FINAL_NORM = False
RETRIEVAL_DIM = 256

P_IDENTITIES = 64                    # PK sampling: P identities x K images each -> batch = 256
K_IMAGES = 4
USE_AMP = True                       # mixed precision -- roughly halves activation memory

ARCFACE_MARGIN = 0.25
ARCFACE_SCALE = 40.0
ARCFACE_SUBCENTERS = 3
ADAFACE_H = 0.33                     # quality-margin sensitivity

LR_HEAD = 1e-3                       # tuned for batch=256
LR_BACKBONE_MULT = 0.05              # backbone unfrozen block gets LR_HEAD * this
ADAM_EPS = 1e-10

LAMBDA_ARCFACE = 1.0                 # set one of the two to 0 for a single-loss ablation
LAMBDA1_TRIPLET = 100.0

EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- scheduler / early-stop / logging / checkpointing ---
LR_WARMUP_EPOCHS = 5
SCHEDULER_T0 = 20                    # first cosine cycle length; T0=10 forced 3 restarts before early stop
LR_MIN_FRACTION = 1e-7               # per-group LR floor as a fraction of its base LR (effectively none)

EVAL_EVERY_N_EPOCHS = 1
EARLY_STOP_PATIENCE = 25

CHECKPOINT_DIR = "checkpoints"
LOG_CSV_PATH = "train_log.csv"

# weight-only warm start (NOT a full resume: optimizer/scheduler/epoch counter
# still start fresh from 0). Set to a .pt path to init model weights from an
# earlier run's checkpoint instead of the raw pretrained backbone + random head.
# strict=False on load since older checkpoints include the now-removed ssl_head.
INIT_WEIGHTS_PATH = None


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
class IdentityFolderDataset(Dataset):
    """dataset/train/{identity_id}/*.jpg -> (path, label_idx)."""

    def __init__(self, root, identities):
        self.root = Path(root)
        self.identities = sorted(identities)
        self.label_to_idx = {name: i for i, name in enumerate(self.identities)}
        self.samples = []  # (path, label_idx)
        self.by_label = {i: [] for i in range(len(self.identities))}
        for name in self.identities:
            idx = self.label_to_idx[name]
            for img_path in (self.root / name).glob("*.jpg"):
                self.samples.append((img_path, idx))
                self.by_label[idx].append(img_path)
        # drop identities with <2 images (can't form pairs for triplet/PK)
        self.by_label = {k: v for k, v in self.by_label.items() if len(v) >= 2}
        self.valid_labels = list(self.by_label.keys())

    def __len__(self):
        return len(self.samples)


class PKBatchSampler(Sampler):
    """Yields batches of P identities x K images, for the supervised branch."""

    def __init__(self, dataset: IdentityFolderDataset, p: int, k: int, iters_per_epoch: int):
        self.dataset = dataset
        self.p = p
        self.k = k
        self.iters_per_epoch = iters_per_epoch

    def __iter__(self):
        for _ in range(self.iters_per_epoch):
            chosen_ids = random.sample(self.dataset.valid_labels, self.p)
            batch = []
            for label in chosen_ids:
                paths = self.dataset.by_label[label]
                k = min(self.k, len(paths))
                chosen_paths = random.sample(paths, k)
                # pad by resampling with replacement if identity has < K images
                while len(chosen_paths) < self.k:
                    chosen_paths.append(random.choice(paths))
                batch.extend((p, label) for p in chosen_paths)
            yield batch

    def __len__(self):
        return self.iters_per_epoch


sup_transform = transforms.Compose([
    # Deliberately NOT segmenting/swapping backgrounds (real backgrounds are kept
    # as-is) -- instead leaning on stronger appearance aug so the model can't
    # shortcut on background color/texture: wider crop range, grayscale, heavier
    # erasing. See vpet.md discussion on background invariance.
    transforms.RandomResizedCrop(IMG_SIZE, scale=(CROP_SCALE_MIN, CROP_SCALE_MAX)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(COLORJITTER_BRIGHTNESS, COLORJITTER_CONTRAST, COLORJITTER_SATURATION, COLORJITTER_HUE),
    transforms.RandomGrayscale(p=GRAYSCALE_P),
    transforms.ToTensor(),
    transforms.RandomErasing(p=ERASING_P, scale=(ERASING_SCALE_MIN, ERASING_SCALE_MAX)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SupDataset(Dataset):
    """Thin wrapper so DataLoader(batch_sampler=PKBatchSampler(...)) works."""

    def __init__(self, base: IdentityFolderDataset):
        self.base = base

    def __getitem__(self, item):
        path, label = item
        img = Image.open(path).convert("RGB")
        return sup_transform(img), label

    def __len__(self):
        return len(self.base)


def sup_batch_collate(items):
    imgs, labels = zip(*items)
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class DinoReIDModel(nn.Module):
    def __init__(self, backbone_path, retrieval_dim, unfreeze_block_indices, unfreeze_final_norm=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_path)
        hidden = self.backbone.config.hidden_size

        # freeze everything, then unfreeze the specified transformer block(s) by index
        for p in self.backbone.parameters():
            p.requires_grad = False
        blocks = self.backbone.model.layer  # DINOv3ViTModel: encoder submodule is named "model", not "encoder"
        for idx in unfreeze_block_indices:
            for p in blocks[idx].parameters():
                p.requires_grad = True
        # self.backbone.norm is a SEPARATE final LayerNorm applied after all 12 blocks
        # (not one of the 12 -- confirmed via model introspection), only 768*2=1536
        # params (scale+bias). Unfreezing it alone tests how much of a block's benefit
        # is just "some adaptation allowed" vs actually retuning attention/MLP.
        if unfreeze_final_norm:
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

        self.retrieval_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, retrieval_dim),
            nn.BatchNorm1d(retrieval_dim),
        )

    def backbone_features(self, x):
        out = self.backbone(pixel_values=x)
        return out.last_hidden_state[:, 0]  # CLS token

    def forward_retrieval(self, x):
        feat = self.backbone_features(x)
        emb = self.retrieval_head(feat)
        return F.normalize(emb, dim=-1)

    def trainable_param_groups(self, lr_head, backbone_lr_mult):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.retrieval_head.parameters())
        return [
            {"params": backbone_params, "lr": lr_head * backbone_lr_mult},
            {"params": head_params, "lr": lr_head},
        ]


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------
class SubCenterAdaFaceArcFace(nn.Module):
    """Sub-center ArcFace with AdaFace-style quality margin applied post-selection."""

    def __init__(self, embed_dim, num_classes, subcenters, margin, scale, h=0.33, eps=1e-3):
        super().__init__()
        self.subcenters = subcenters
        self.margin = margin
        self.scale = scale
        self.h = h
        self.eps = eps
        self.weight = nn.Parameter(torch.randn(num_classes * subcenters, embed_dim) * 0.01)
        self.register_buffer("norm_mean", torch.tensor(20.0))
        self.register_buffer("norm_std", torch.tensor(5.0))

    def forward(self, embeddings, raw_feat_norm, labels):
        # embeddings: already L2-normalized (B, D); raw_feat_norm: pre-norm feature norm (B,) for quality
        w = F.normalize(self.weight, dim=-1)
        cos_all = embeddings @ w.t()  # (B, C*K)
        cos_all = cos_all.view(-1, cos_all.size(1) // self.subcenters, self.subcenters)  # (B, C, K)

        # sub-center selection: nearest (easiest-matching) sub-center per class, for
        # BOTH negative classes and the sample's own (target) class -- original
        # sub-center ArcFace design (tolerates multiple "modes" per identity).
        # (reverted from farthest/hardest-positive selection -- didn't resolve the
        # mAP decline and added another source of overfit pressure on top of it)
        cos_best, _ = cos_all.max(dim=-1)  # (B, C)

        # running batch-norm-style update of feature-norm stats for AdaFace quality term
        with torch.no_grad():
            self.norm_mean.mul_(0.99).add_(raw_feat_norm.mean() * 0.01)
            self.norm_std.mul_(0.99).add_(raw_feat_norm.std() * 0.01)
        quality = (raw_feat_norm - self.norm_mean) / (self.norm_std + self.eps)
        quality = torch.clamp(quality, -1, 1).detach()  # as in AdaFace: the margin must not give the model a gradient on its own feature norm
        adaptive_margin = self.margin * (1 + self.h * quality)  # per-sample margin

        target_cos = cos_best.gather(1, labels.view(-1, 1)).squeeze(1)
        theta = torch.acos(torch.clamp(target_cos, -1 + self.eps, 1 - self.eps))
        target_cos_margin = torch.cos(theta + adaptive_margin)

        logits = cos_best.clone()
        # torch.acos/cos are auto-promoted to fp32 by autocast (numerically sensitive
        # ops), so target_cos_margin may not match logits' dtype under AMP -- cast back.
        logits.scatter_(1, labels.view(-1, 1), target_cos_margin.to(logits.dtype).view(-1, 1))
        logits = logits * self.scale
        return F.cross_entropy(logits, labels)


class TripletBatchHard(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        dist = torch.cdist(embeddings, embeddings, p=2)  # (B, B)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        diff = ~same
        eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        pos_mask = same & ~eye

        hardest_pos = (dist * pos_mask).max(dim=1).values
        dist_for_neg = dist.clone()
        dist_for_neg[~diff] = float("inf")
        hardest_neg = dist_for_neg.min(dim=1).values

        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------
class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row):
        # fixed 8-decimal-place formatting for floats (no scientific notation,
        # no runaway float repr precision) -- ints (epoch) and "" (unevaluated
        # rank1/mAP) pass through unchanged
        formatted = {
            k: f"{v:.8f}" if isinstance(v, float) else v
            for k, v in row.items()
        }
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(formatted)


def print_config():
    # dumped at the top of every run's log so each train_run.log is self-contained --
    # no more reconstructing hyperparameters from code-comment history after the fact.
    keys = [
        "DATA_ROOT", "DATASET_ROOT", "CHECKPOINT_DIR", "LOG_CSV_PATH", "VAL_FRACTION", "VAL_SEED",
        "UNFREEZE_BLOCK_INDICES", "UNFREEZE_FINAL_NORM", "RETRIEVAL_DIM",
        "P_IDENTITIES", "K_IMAGES", "USE_AMP",
        "CROP_SCALE_MIN", "CROP_SCALE_MAX", "COLORJITTER_BRIGHTNESS", "COLORJITTER_CONTRAST",
        "COLORJITTER_SATURATION", "COLORJITTER_HUE", "GRAYSCALE_P",
        "ERASING_P", "ERASING_SCALE_MIN", "ERASING_SCALE_MAX",
        "ARCFACE_MARGIN", "ARCFACE_SCALE", "ARCFACE_SUBCENTERS", "ADAFACE_H",
        "LR_HEAD", "LR_BACKBONE_MULT", "ADAM_EPS", "LAMBDA_ARCFACE", "LAMBDA1_TRIPLET",
        "EPOCHS", "LR_WARMUP_EPOCHS", "SCHEDULER_T0", "LR_MIN_FRACTION",
        "EVAL_EVERY_N_EPOCHS", "EARLY_STOP_PATIENCE", "INIT_WEIGHTS_PATH",
    ]
    print("[config] " + " ".join(f"{k}={globals()[k]}" for k in keys))


def train():
    print_config()
    train_names, val_split = split_val(DATA_ROOT, VAL_FRACTION, VAL_SEED)
    base_ds = IdentityFolderDataset(DATA_ROOT, train_names)
    num_classes = len(base_ds.identities)  # labels are indices into identities, so size by all of them
    val_set = EvalSet(*val_split, IMG_SIZE)

    sup_ds = SupDataset(base_ds)
    iters_per_epoch = max(1, len(base_ds.samples) // (P_IDENTITIES * K_IMAGES))
    sup_sampler = PKBatchSampler(base_ds, P_IDENTITIES, K_IMAGES, iters_per_epoch)
    sup_loader = DataLoader(sup_ds, batch_sampler=sup_sampler, collate_fn=sup_batch_collate, num_workers=4)

    model = DinoReIDModel(BACKBONE_PATH, RETRIEVAL_DIM, UNFREEZE_BLOCK_INDICES, UNFREEZE_FINAL_NORM).to(DEVICE)
    if INIT_WEIGHTS_PATH:
        missing, unexpected = model.load_state_dict(torch.load(INIT_WEIGHTS_PATH, map_location=DEVICE), strict=False)
        print(f"[warm start] loaded weights from {INIT_WEIGHTS_PATH} "
              f"(missing={len(missing)}, unexpected={len(unexpected)} -- e.g. old ssl_head keys, expected)")
    arcface = SubCenterAdaFaceArcFace(
        RETRIEVAL_DIM, num_classes, ARCFACE_SUBCENTERS, ARCFACE_MARGIN, ARCFACE_SCALE, ADAFACE_H
    ).to(DEVICE)
    triplet = TripletBatchHard().to(DEVICE)

    param_groups = model.trainable_param_groups(LR_HEAD, LR_BACKBONE_MULT)
    param_groups.append({"params": arcface.parameters(), "lr": LR_HEAD})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05, eps=ADAM_EPS)
    base_lrs = [g["lr"] for g in optimizer.param_groups]  # captured pre-scheduler, for the floor clamp below

    # linear warmup, then cosine decay with warm restarts (SGDR). eta_min stays 0 because it is one
    # scalar for all groups; the per-group floor clamp after scheduler.step() handles the floor instead.
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1 / LR_WARMUP_EPOCHS, end_factor=1.0, total_iters=LR_WARMUP_EPOCHS
    )
    restart_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=SCHEDULER_T0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, restart_scheduler], milestones=[LR_WARMUP_EPOCHS]
    )
    amp_enabled = USE_AMP and DEVICE == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    logger = CSVLogger(LOG_CSV_PATH, [
        "epoch", "lr_backbone", "lr_head", "loss_arc", "loss_trip", "rank1", "mAP",
    ])

    # untrained baseline, for reference only: pretrained DINOv3 features can already score high on
    # this kind of data, so compare later epochs against this number.
    rank1_0, mAP_0 = evaluate(model, val_set, DEVICE)
    print(f"[baseline] val Rank-1={rank1_0:.4f} mAP={mAP_0:.4f}")

    best_mAP = -1.0  # not the baseline, so best.pt is always written by the first eval of this run
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        # read LR before scheduler.step() so logs reflect the LR actually
        # used THIS epoch, not the one that will apply next epoch
        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]
        model.train()

        sum_arc = sum_trip = 0.0
        n_steps = 0

        for imgs, labels in sup_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with autocast(device_type=DEVICE, enabled=amp_enabled):
                feat = model.backbone_features(imgs)
                raw_norm = feat.norm(dim=-1)
                emb = F.normalize(model.retrieval_head(feat), dim=-1)
                loss_arc = arcface(emb, raw_norm, labels)
                loss_trip = triplet(emb, labels)
                loss = LAMBDA_ARCFACE * loss_arc + LAMBDA1_TRIPLET * loss_trip
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            sum_arc += loss_arc.item()
            sum_trip += loss_trip.item()
            n_steps += 1

        scheduler.step()
        for g, base_lr in zip(optimizer.param_groups, base_lrs):
            g["lr"] = max(g["lr"], base_lr * LR_MIN_FRACTION)

        rank1 = mAP = ""
        if (epoch + 1) % EVAL_EVERY_N_EPOCHS == 0 or epoch == EPOCHS - 1:
            rank1, mAP = evaluate(model, val_set, DEVICE)
            print(f"epoch {epoch}: loss_arc={sum_arc/n_steps:.4f} loss_trip={sum_trip/n_steps:.4f} "
                  f"Rank-1={rank1:.4f} mAP={mAP:.4f}")

            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "last.pt"))
            if mAP > best_mAP:
                best_mAP = mAP
                epochs_without_improvement = 0
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best.pt"))
                print(f"  -> new best mAP={mAP:.4f}, saved checkpoints/best.pt")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping: no mAP improvement for {EARLY_STOP_PATIENCE} eval rounds "
                          f"(best={best_mAP:.4f}, current={mAP:.4f})")
                    logger.log({
                        "epoch": epoch, "lr_backbone": lr_backbone, "lr_head": lr_head,
                        "loss_arc": sum_arc / n_steps, "loss_trip": sum_trip / n_steps,
                        "rank1": rank1, "mAP": mAP,
                    })
                    break
        else:
            print(f"epoch {epoch}: loss_arc={sum_arc/n_steps:.4f} loss_trip={sum_trip/n_steps:.4f}")

        logger.log({
            "epoch": epoch, "lr_backbone": lr_backbone, "lr_head": lr_head,
            "loss_arc": sum_arc / n_steps, "loss_trip": sum_trip / n_steps,
            "rank1": rank1, "mAP": mAP,
        })

    # archive this run's best checkpoint under its val-mAP-named file, so the next run
    # does not overwrite it, then report best.pt once on the held-out query/gallery test split.
    best_path = os.path.join(CHECKPOINT_DIR, "best.pt")
    if os.path.exists(best_path):
        archive_path = os.path.join(CHECKPOINT_DIR, f"{best_mAP:.4f}.pt")
        shutil.copy2(best_path, archive_path)
        print(f"[archive] copied best.pt -> {archive_path}")
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        rank1, mAP = evaluate(model, EvalSet.from_dataset_root(DATASET_ROOT, IMG_SIZE), DEVICE)
        print(f"[test] best.pt on query/gallery: Rank-1={rank1:.4f} mAP={mAP:.4f}")


if __name__ == "__main__":
    train()
