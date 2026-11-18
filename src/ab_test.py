"""OFFLINE-SIMULATED A/B test: users are randomly split; A sees retrieval order, B sees the DCN ranker.
A 'click' = a shown item the user really liked in the test period (replay). This is a simulation, not live traffic —
say so in your report. Reports CTR@10 and a two-proportion z-test."""
import json
import math
import numpy as np
from config import ART
from core import Recommender
from evaluate import test_rankings

K = 10


def z_test(c1, n1, c2, n2):
    p = (c1 + c2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (c2 / n2 - c1 / n1) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


if __name__ == "__main__":
    rec = Recommender(seen_splits=("train", "val"))
    users, rel, systems, _ = test_rankings(rec)
    rng = np.random.default_rng(42)
    arm = rng.random(len(users)) < 0.5          # False -> A, True -> B
    imp = {"A": 0, "B": 0}
    clk = {"A": 0, "B": 0}
    for u, a, b, is_b in zip(users, systems["retrieval"], systems["retrieval+ranker"], arm):
        shown = list(b if is_b else a)[:K]
        key = "B" if is_b else "A"
        imp[key] += len(shown)
        clk[key] += sum(i in rel[u] for i in shown)
    z, p = z_test(clk["A"], imp["A"], clk["B"], imp["B"])
    res = {"A(retrieval)_ctr@10": clk["A"] / imp["A"], "B(ranker)_ctr@10": clk["B"] / imp["B"],
           "impressions": imp, "clicks": clk, "z": z, "p_value": p, "note": "offline replay simulation"}
    json.dump(res, open(ART / "ab_results.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
