"""Offline evaluation on the held-out TEST period: popularity vs retrieval vs retrieval+ranker."""
import json
import numpy as np
import pandas as pd
from config import PROC, ART, POS_THRESHOLD, N_CANDIDATES
from core import Recommender
from metrics import recall_at_k, ndcg_at_k

KS = (10, 20)


def test_rankings(rec):
    """Returns {user: relevant_set}, and ranked item-idx lists per system."""
    te = pd.read_parquet(PROC / "test.parquet")
    te = te[(te.rating >= POS_THRESHOLD) & te.item_id.isin(rec.store.item_index)]
    rel = te.groupby("user_id").item_id.apply(lambda s: {rec.store.item_index[i] for i in s}).to_dict()
    users = [u for u in rel if rec.knows(u)]
    cands = [rec.candidates(u, N_CANDIDATES) for u in users]
    systems = {"retrieval": [c[0] for c in cands]}
    if rec.ranker is not None:
        systems["retrieval+ranker"] = [r[0] for r in rec.rank_many(users, cands)]
    pop = rec.store.popularity_order
    systems["popularity"] = [[i for i in pop if i not in rec.seen.get(u, ())][:N_CANDIDATES] for u in users]
    return users, rel, systems, cands


if __name__ == "__main__":
    rec = Recommender(seen_splits=("train", "val"))
    users, rel, systems, cands = test_rankings(rec)
    out = {"n_test_users": len(users)}
    out["candidate_recall@%d" % N_CANDIDATES] = float(np.mean([recall_at_k(list(c[0]), rel[u], N_CANDIDATES) for u, c in zip(users, cands)]))
    for name, lists in systems.items():
        for k in KS:
            out[f"{name}/recall@{k}"] = float(np.mean([recall_at_k(list(l), rel[u], k) for u, l in zip(users, lists)]))
            out[f"{name}/ndcg@{k}"] = float(np.mean([ndcg_at_k(list(l), rel[u], k) for u, l in zip(users, lists)]))
    json.dump(out, open(ART / "eval_metrics.json", "w"), indent=2)
    print(f"{len(users)} test users | candidate recall@{N_CANDIDATES} = {out['candidate_recall@%d' % N_CANDIDATES]:.3f}")
    print(f"{'system':<18}" + "".join(f"{m+'@'+str(k):>12}" for k in KS for m in ("recall", "ndcg")))
    for name in systems:
        print(f"{name:<18}" + "".join(f"{out[f'{name}/{m}@{k}']:>12.4f}" for k in KS for m in ("recall", "ndcg")))
