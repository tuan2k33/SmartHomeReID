# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## State of the repo

- The folder is named `truck-tarp-detection` and the README says "detect whether a truck is covered by a tarp", but the code in it is a **pet Re-ID** pipeline (DINOv3 ViT-B/16 backbone, DogFaceNet dataset). No truck or tarp code exists yet. Ask which one the user wants before extending it.
- The three `.py` files were pasted into `README.md` and split out; they have never been run here. There are no tests, no linter config and no build step.
- The code refers to `vpet.md`, `petreid.md` and `split_dataset.py`; none of these exist in this repo.
- Python deps: `torch`, `torchvision`, `transformers`, `pillow`, `numpy`. The shared base env (`~/.dev/venv`) has only numpy and pillow among these, so install the rest into this project's venv: `uv pip install --python .venv/bin/python torch torchvision transformers` (see `/mnt/c/Dev/.claude/CLAUDE.md` for venv rules).

## Commands

Config is edited directly in the module-level constants (no CLI, no argparse). All paths default to `/home/user/Documents/...` and must be changed first (`DATA_ROOT`, `DATASET_ROOT`, `BACKBONE_PATH` in `train_skeleton.py`, `DATASET_ROOT` and `CHECKPOINT_PATH` in `eval.py`).

- Train: `python train_skeleton.py` (writes `checkpoints/{last,best,<mAP>}.pt` and `train_log.csv` to the cwd)
- Eval a checkpoint: set `CHECKPOINT_PATH` in `eval.py`, then `python eval.py`; with `None` it evaluates an untrained model (smoke test)

## Architecture

- `train_skeleton.py` holds config, data, model, losses and the training loop. `DinoReIDModel` = DINOv3 backbone (frozen except the blocks in `UNFREEZE_BLOCK_INDICES`) + MLP retrieval head, L2-normalised output. Loss = `LAMBDA_ARCFACE * SubCenterAdaFaceArcFace + LAMBDA1_TRIPLET * TripletBatchHard`.
- Batches come from `PKBatchSampler` (P identities x K images) passed as `batch_sampler`; it yields `(path, label)` tuples, and `SupDataset.__getitem__` receives that tuple instead of an index. Keep that contract when changing either class.
- `eval_utils.py` is shared by training (per-epoch eval via `evaluate_model`) and `eval.py`. `eval.py` imports the model and config from `train_skeleton.py`, which is why the metric code lives in a third file (avoids a circular import).
- Expected data layout: `<root>/train/<identity>/*.jpg` for training; `<root>/{query,gallery}/` plus `query_labels.csv` and `gallery_labels.csv` (columns `filename`, `identity`) for eval.
- The training loop picks `best.pt` by mAP on the same query/gallery split used for reporting (no separate validation split).
- Checkpoints hold only `model.state_dict()` (not the ArcFace weights, optimizer or scheduler), so `INIT_WEIGHTS_PATH` is a weight-only warm start, not a resume.
