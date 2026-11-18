"""Feature store used for BOTH ranker training and serving (same code = no train/serve skew)."""
import numpy as np
import pandas as pd
from config import POS_THRESHOLD


class FeatureStore:
    def __init__(self, train: pd.DataFrame, items: pd.DataFrame):
        self.item_ids = items.item_id.tolist()
        self.item_index = {i: k for k, i in enumerate(self.item_ids)}
        self.genres = sorted({g for s in items.genres for g in s.split("|") if g})
        gi = {g: k for k, g in enumerate(self.genres)}
        G = np.zeros((len(items), len(self.genres)), dtype=np.float32)
        for k, s in enumerate(items.genres):
            for g in s.split("|"):
                if g:
                    G[k, gi[g]] = 1
        self.G = G
        st = train.groupby("item_id").rating.agg(["count", "mean"]).reindex(self.item_ids)
        self.pop = st["count"].fillna(0).to_numpy()
        self.logpop = np.log1p(self.pop).astype(np.float32)
        self.item_avg = st["mean"].fillna(train.rating.mean()).to_numpy().astype(np.float32)
        self.popularity_order = np.argsort(-self.pop)

        us = train.groupby("user_id").rating.agg(["count", "mean"])
        self.user_pos = {u: k for k, u in enumerate(us.index)}
        n = len(us)
        self.user_logn = np.append(np.log1p(us["count"].to_numpy()), 0).astype(np.float32)
        self.user_avg = np.append(us["mean"].to_numpy(), train.rating.mean()).astype(np.float32)
        P = np.zeros((n + 1, len(self.genres)), dtype=np.float32)   # last row = unknown user
        pos = train[train.rating >= POS_THRESHOLD]
        rows = pos.user_id.map(self.user_pos).to_numpy()
        cols = pos.item_id.map(self.item_index)
        ok = cols.notna().to_numpy()
        np.add.at(P, rows[ok], G[cols[ok].astype(int).to_numpy()])
        s = P.sum(1, keepdims=True)
        self.P = np.divide(P, s, out=np.zeros_like(P), where=s > 0)   # user genre profile

    def featurize(self, users, item_idx, retrieval_score):
        u = np.array([self.user_pos.get(x, len(self.user_pos)) for x in users])
        item_idx = np.asarray(item_idx)
        aff = (self.P[u] * self.G[item_idx]).sum(1)
        base = np.stack([np.asarray(retrieval_score, dtype=np.float32), self.logpop[item_idx], self.item_avg[item_idx],
                         self.user_logn[u], self.user_avg[u], aff], axis=1)
        return np.concatenate([base, self.G[item_idx]], axis=1).astype(np.float32)

    def cold_start(self, k, genres=None, liked_items=None):
        """Popularity fallback, optionally steered by chosen genres / liked items (content-based)."""
        prof = np.zeros(len(self.genres), dtype=np.float32)
        for g in genres or []:
            if g in self.genres:
                prof[self.genres.index(g)] += 1
        for it in liked_items or []:
            if it in self.item_index:
                prof += self.G[self.item_index[it]]
        if prof.sum() == 0:
            return self.popularity_order[:k]
        prof /= prof.sum()
        score = (self.G @ prof) * 10 + self.logpop / max(self.logpop.max(), 1)
        score[(self.G @ prof) == 0] -= 100
        return np.argsort(-score)[:k]
