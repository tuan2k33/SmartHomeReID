import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_utils import compute_rank1_map


def test_rank1_and_map_on_hand_computed_ranking():
    query = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    query_ids = ["a", "b", "z"]  # "z" has no gallery image: must not be scored
    gallery = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0], [0.8, 0.6]])
    gallery_ids = ["a", "a", "b", "b"]
    # q0: ranking a b a b -> hits [1,0,1,0], AP = (1 + 2/3) / 2 = 5/6
    # q1: ranking b a b a -> hits [1,0,1,0], AP = 5/6
    rank1, mAP = compute_rank1_map(query, query_ids, gallery, gallery_ids)
    assert rank1 == 1.0
    assert mAP == pytest.approx(5 / 6)


def test_no_scorable_query_raises():
    with pytest.raises(ValueError):
        compute_rank1_map(np.eye(2), ["x", "y"], np.eye(2), ["a", "b"])
