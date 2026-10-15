"""Shared Rank-1/mAP eval helpers, split out so both train_skeleton.py
(per-epoch validation) and eval.py (standalone test eval) can import them
without a circular import (eval.py already imports the model/config from
train_skeleton.py)."""

import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.amp import autocast
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_split(dataset_root, name):
    labels_csv = os.path.join(dataset_root, f"{name}_labels.csv")
    img_dir = os.path.join(dataset_root, name)
    with open(labels_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    paths = [os.path.join(img_dir, r["filename"]) for r in rows]
    identities = [r["identity"] for r in rows]
    return paths, identities


def split_val(train_root, fraction, seed):
    """Hold out a seeded fraction of the train identities as validation.
    Returns (train identity names, (query_paths, query_ids, gallery_paths, gallery_ids)):
    per held-out identity the first image is the query and the rest are gallery."""
    root = Path(train_root)
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    random.Random(seed).shuffle(names)
    n_val = max(1, round(len(names) * fraction))
    val_names, train_names = sorted(names[:n_val]), sorted(names[n_val:])
    qp, qi, gp, gi = [], [], [], []
    for name in val_names:
        imgs = sorted((root / name).glob("*.jpg"))
        if len(imgs) < 2:
            continue
        qp.append(imgs[0])
        qi.append(name)
        gp += imgs[1:]
        gi += [name] * (len(imgs) - 1)
    return train_names, (qp, qi, gp, gi)


def _load_uint8(paths, img_size):
    resize = transforms.Resize((img_size, img_size))
    out = []
    for p in paths:
        with Image.open(p) as im:
            out.append(pil_to_tensor(resize(im.convert("RGB"))))
    return torch.stack(out)


class EvalSet:
    """Query/gallery images decoded and resized once (uint8), so repeated evals only re-encode."""

    def __init__(self, query_paths, query_ids, gallery_paths, gallery_ids, img_size):
        self.query_ids = query_ids
        self.gallery_ids = gallery_ids
        self.query = _load_uint8(query_paths, img_size)
        self.gallery = _load_uint8(gallery_paths, img_size)

    @classmethod
    def from_dataset_root(cls, dataset_root, img_size):
        query_paths, query_ids = load_split(dataset_root, "query")
        gallery_paths, gallery_ids = load_split(dataset_root, "gallery")
        return cls(query_paths, query_ids, gallery_paths, gallery_ids, img_size)


@torch.no_grad()
def encode(model, images, device, batch_size=64):
    was_training = model.training
    model.eval()
    embeddings = []
    for i in range(0, len(images), batch_size):
        x = images[i:i + batch_size].to(device).float().div_(255)
        x = (x - MEAN.to(device)) / STD.to(device)
        with autocast(device_type=device, enabled=(device == "cuda")):
            emb = model.forward_retrieval(x)
        embeddings.append(emb.float().cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(embeddings, axis=0)


def compute_rank1_map(query_emb, query_ids, gallery_emb, gallery_ids):
    query_ids = np.array(query_ids)
    gallery_ids = np.array(gallery_ids)
    sim = query_emb @ gallery_emb.T  # (Nq, Ng), cosine since embeddings are L2-normalized

    rank1_hits = 0
    aps = []
    for i in range(len(query_ids)):
        order = np.argsort(-sim[i])
        matches = (gallery_ids[order] == query_ids[i]).astype(np.int32)
        n_relevant = matches.sum()
        if n_relevant == 0:
            continue  # no gallery image for this identity: not scored, in neither metric

        rank1_hits += int(matches[0] == 1)

        cum_hits = np.cumsum(matches)
        precision_at_k = cum_hits / (np.arange(len(matches)) + 1)
        ap = (precision_at_k * matches).sum() / n_relevant
        aps.append(ap)

    if not aps:
        raise ValueError("no query identity has a gallery image")
    return rank1_hits / len(aps), float(np.mean(aps))


def evaluate(model, eval_set, device):
    query_emb = encode(model, eval_set.query, device)
    gallery_emb = encode(model, eval_set.gallery, device)
    return compute_rank1_map(query_emb, eval_set.query_ids, gallery_emb, eval_set.gallery_ids)
