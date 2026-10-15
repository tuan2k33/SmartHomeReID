"""
Standalone Rank-1/mAP eval on the query/gallery split produced by
split_dataset.py (vpet.md section 6). See eval_utils.py for the metric
implementation (shared with train_skeleton.py's periodic in-training eval).

Usage: edit CHECKPOINT_PATH below (or leave None to eval an untrained
model, e.g. to smoke-test the pipeline before training finishes) and run.
"""

import torch

from train_skeleton import DinoReIDModel, BACKBONE_PATH, IMG_SIZE, RETRIEVAL_DIM, UNFREEZE_BLOCK_INDICES, UNFREEZE_FINAL_NORM, DEVICE
from eval_utils import make_eval_transform, load_split, encode, compute_rank1_map

DATASET_ROOT = "/home/user/Documents/dataset/pet-data/DogFaceNet/dataset"
CHECKPOINT_PATH = None  # e.g. "checkpoints/best.pt" once training produces one


def main():
    model = DinoReIDModel(BACKBONE_PATH, RETRIEVAL_DIM, UNFREEZE_BLOCK_INDICES, UNFREEZE_FINAL_NORM).to(DEVICE)
    if CHECKPOINT_PATH:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE), strict=False)
    else:
        print("[warn] CHECKPOINT_PATH is None -- evaluating an UNTRAINED model (smoke test only).")
    model.eval()

    eval_transform = make_eval_transform(IMG_SIZE)
    query_paths, query_ids = load_split(DATASET_ROOT, "query")
    gallery_paths, gallery_ids = load_split(DATASET_ROOT, "gallery")
    print(f"query: {len(query_paths)} images, gallery: {len(gallery_paths)} images")

    query_emb = encode(model, query_paths, eval_transform, DEVICE)
    gallery_emb = encode(model, gallery_paths, eval_transform, DEVICE)

    rank1, mAP = compute_rank1_map(query_emb, query_ids, gallery_emb, gallery_ids)
    print(f"Rank-1: {rank1:.4f}")
    print(f"mAP:    {mAP:.4f}")


if __name__ == "__main__":
    main()
