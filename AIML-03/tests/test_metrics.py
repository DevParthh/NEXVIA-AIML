import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import recall_at_k, ndcg_at_k


def test_perfect_ranking():
    assert ndcg_at_k([1, 2, 3], {1, 2}, 3) == 1.0
    assert recall_at_k([1, 2, 3], {1, 2}, 2) == 1.0


def test_miss():
    assert ndcg_at_k([9, 8], {1}, 2) == 0.0
    assert recall_at_k([9, 8], {1}, 2) == 0.0


def test_position_matters():
    assert ndcg_at_k([9, 1], {1}, 2) < ndcg_at_k([1, 9], {1}, 2)
