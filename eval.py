"""
Standalone Rank-1/mAP eval on the query/gallery split produced by
split_dataset.py (vpet.md section 6). See eval_utils.py for the metric
implementation (shared with train_skeleton.py's periodic in-training eval).

Usage: edit CHECKPOINT_PATH below (or leave None to eval an untrained
model, e.g. to smoke-test the pipeline before training finishes) and run.
"""

import torch

from train_skeleton import DinoReIDModel, BACKBONE_PATH, IMG_SIZE, RETRIEVAL_DIM, UNFREEZE_BLOCK_INDICES, UNFREEZE_FINAL_NORM, DEVICE, DATASET_ROOT
from eval_utils import EvalSet, evaluate

CHECKPOINT_PATH = None  # e.g. "checkpoints/best.pt" once training produces one


def main():
    model = DinoReIDModel(BACKBONE_PATH, RETRIEVAL_DIM, UNFREEZE_BLOCK_INDICES, UNFREEZE_FINAL_NORM).to(DEVICE)
    if CHECKPOINT_PATH:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE), strict=False)
    else:
        print("[warn] CHECKPOINT_PATH is None -- evaluating an UNTRAINED model (smoke test only).")
    model.eval()

    eval_set = EvalSet.from_dataset_root(DATASET_ROOT, IMG_SIZE)
    print(f"query: {len(eval_set.query)} images, gallery: {len(eval_set.gallery)} images")

    rank1, mAP = evaluate(model, eval_set, DEVICE)
    print(f"Rank-1: {rank1:.4f}")
    print(f"mAP:    {mAP:.4f}")


if __name__ == "__main__":
    main()
