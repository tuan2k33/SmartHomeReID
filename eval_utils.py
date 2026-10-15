"""Shared Rank-1/mAP eval helpers, split out so both train_skeleton.py
(periodic in-training eval) and eval.py (standalone eval) can import them
without a circular import (eval.py already imports the model/config from
train_skeleton.py)."""

import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.amp import autocast
from torchvision import transforms


def make_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_split(dataset_root, name):
    labels_csv = os.path.join(dataset_root, f"{name}_labels.csv")
    img_dir = os.path.join(dataset_root, name)
    with open(labels_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    paths = [os.path.join(img_dir, r["filename"]) for r in rows]
    identities = [r["identity"] for r in rows]
    return paths, identities


@torch.no_grad()
def encode(model, paths, eval_transform, device, batch_size=64):
    was_training = model.training
    model.eval()
    embeddings = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        imgs = torch.stack([eval_transform(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        with autocast(device_type=device, enabled=(device == "cuda")):
            emb = model.forward_retrieval(imgs)
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
            continue  # identity has no gallery images left (shouldn't happen with our split rule)

        rank1_hits += int(matches[0] == 1)

        cum_hits = np.cumsum(matches)
        precision_at_k = cum_hits / (np.arange(len(matches)) + 1)
        ap = (precision_at_k * matches).sum() / n_relevant
        aps.append(ap)

    rank1 = rank1_hits / len(query_ids)
    mAP = float(np.mean(aps))
    return rank1, mAP


def evaluate_model(model, dataset_root, img_size, device):
    """Convenience wrapper: full query/gallery encode + Rank-1/mAP in one call."""
    eval_transform = make_eval_transform(img_size)
    query_paths, query_ids = load_split(dataset_root, "query")
    gallery_paths, gallery_ids = load_split(dataset_root, "gallery")
    query_emb = encode(model, query_paths, eval_transform, device)
    gallery_emb = encode(model, gallery_paths, eval_transform, device)
    return compute_rank1_map(query_emb, query_ids, gallery_emb, gallery_ids)
