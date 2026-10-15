# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## State of the repo

- The three `.py` files were pasted into `README.md` and split out; the training code has never been run here (no GPU data or model on this machine). Only the metric in `eval_utils.py` has a test; there is no linter config and no build step.
- The code refers to `vpet.md`, `petreid.md` and `split_dataset.py`; none of these exist in this repo.
- Python deps: `torch`, `torchvision`, `transformers`, `pillow`, `numpy`. The shared base env (`~/.dev/venv`) has only numpy and pillow among these, so install the rest into this project's venv: `uv pip install --python .venv/bin/python torch torchvision transformers` (see `/mnt/c/Dev/.claude/CLAUDE.md` for venv rules).

## Commands

Config is edited directly in the module-level constants (no CLI, no argparse). The default paths are relative placeholders (`data/dataset/...`, `models/dinov3-vitb16-pretrain-lvd1689m`) and must be pointed at the real dataset and backbone first (`DATA_ROOT`, `DATASET_ROOT`, `BACKBONE_PATH` in `train_skeleton.py`, `DATASET_ROOT` and `CHECKPOINT_PATH` in `eval.py`).

- Train: `python train_skeleton.py` (writes `checkpoints/{last,best,<mAP>}.pt` and `train_log.csv` to the cwd)
- Test: `pytest tests/` (single test: `pytest tests/test_eval_utils.py::test_no_scorable_query_raises`); needs `torch` because `eval_utils` imports it
- Eval a checkpoint: set `CHECKPOINT_PATH` in `eval.py`, then `python eval.py`; with `None` it evaluates an untrained model (smoke test)

## Architecture

- `train_skeleton.py` holds config, data, model, losses and the training loop. `DinoReIDModel` = DINOv3 backbone (frozen except the blocks in `UNFREEZE_BLOCK_INDICES`) + MLP retrieval head, L2-normalised output. Loss = `LAMBDA_ARCFACE * SubCenterAdaFaceArcFace + LAMBDA1_TRIPLET * TripletBatchHard`.
- Batches come from `PKBatchSampler` (P identities x K images) passed as `batch_sampler`; it yields `(path, label)` tuples, and `SupDataset.__getitem__` receives that tuple instead of an index. Keep that contract when changing either class.
- `eval_utils.py` is shared by training (per-epoch validation) and `eval.py`. `EvalSet` decodes images once as uint8 and re-encodes them on every `evaluate` call. `eval.py` imports the model and config from `train_skeleton.py`, which is why the metric code lives in a third file (avoids a circular import).
- Expected data layout: `<root>/train/<identity>/*.jpg` for training; `<root>/{query,gallery}/` plus `query_labels.csv` and `gallery_labels.csv` (columns `filename`, `identity`) for eval.
- `split_val` holds out `VAL_FRACTION` of the train identities (seeded); the first image of each is a query, the rest gallery. That validation set picks `best.pt` and drives early stopping. The `DATASET_ROOT` query/gallery split is used once, at the end of `train()`, on `best.pt`, and by `eval.py`.
- `SubCenterAdaFaceArcFace` sizes its weights by `len(identities)`, because labels are indices into all train identities (including ones the sampler skips for having <2 images).
- Checkpoints hold only `model.state_dict()` (not the ArcFace weights, optimizer or scheduler), so `INIT_WEIGHTS_PATH` is a weight-only warm start, not a resume.
