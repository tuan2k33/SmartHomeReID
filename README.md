# truck-tarp-detection

Work in progress. The code here is currently a pet Re-ID training and evaluation pipeline
(DINOv3 backbone, ArcFace + triplet loss), not yet a truck-tarp detector.

- `train_skeleton.py`: config, data, model, losses, training loop
- `eval_utils.py`: Rank-1 / mAP metric and the validation split
- `eval.py`: standalone evaluation of a checkpoint
- `tests/`: `pytest tests/` (needs `torch` installed)
