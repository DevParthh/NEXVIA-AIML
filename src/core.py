"""Two-stage recommender: FAISS candidate retrieval -> DCN ranker, with cold-start fallback."""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import json
import numpy as np
import pandas as pd
import faiss
from config import PROC, ART, N_CANDIDATES
from features import FeatureStore


class Recommender:
    def __init__(self, seen_splits=("train", "val", "test"), load_ranker=True):
        self.item_ids = json.load(open(ART / "item_ids.json"))
        self.user_ids = json.load(open(ART / "user_ids.json"))
        self.user_row = {u: i for i, u in enumerate(self.user_ids)}
        self.user_emb = np.load(ART / "user_emb.npy").astype("float32")
        self.index = faiss.read_index(str(ART / "items.faiss"))
        items = pd.read_parquet(PROC / "items.parquet").set_index("item_id").loc[self.item_ids].reset_index()
        self.items = items
        train = pd.read_parquet(PROC / "train.parquet")
        self.store = FeatureStore(train, items)
        self.seen = {}
        for s in seen_splits:
            df = pd.read_parquet(PROC / f"{s}.parquet")
            idx = df.item_id.map(self.store.item_index)
            for u, i in zip(df.user_id[idx.notna()], idx.dropna().astype(int)):
                self.seen.setdefault(u, set()).add(i)
        self.ranker = None
        if load_ranker and (ART / "ranker").exists():
            import tensorflow as tf
            import tensorflow_recommenders  # noqa: F401  (registers the Cross layer)
            self.ranker = tf.keras.models.load_model(ART / "ranker")

    def knows(self, user_id):
        return user_id in self.user_row

    def candidates(self, user_id, n=N_CANDIDATES):
        """Stage 1: top-n unseen items by embedding similarity."""
        seen = self.seen.get(user_id, set())
        width = min(n + len(seen), self.index.ntotal)
        d, i = self.index.search(self.user_emb[self.user_row[user_id]][None], width)
        keep = [(a, b) for a, b in zip(i[0], d[0]) if a >= 0 and a not in seen][:n]
        return np.array([a for a, _ in keep]), np.array([b for _, b in keep], dtype=np.float32)

    def rank_many(self, users, cands):
        """Stage 2: re-score every user's candidates in one batched model call."""
        X = np.concatenate([self.store.featurize([u] * len(ci), ci, cs) for u, (ci, cs) in zip(users, cands)])
        p = self.ranker.predict(X, batch_size=8192, verbose=0)[:, 0]
        out, pos = [], 0
        for ci, _ in cands:
            s = p[pos:pos + len(ci)]
            pos += len(ci)
            o = np.argsort(-s)
            out.append((ci[o], s[o]))
        return out

    def recommend(self, user_id, k=10, mode="ranker", genres=None, liked_items=None):
        if not self.knows(user_id):                       # cold start
            idx, sc, src = self.store.cold_start(k, genres, liked_items), None, "cold_start"
        else:
            ci, cs = self.candidates(user_id)
            if mode == "ranker" and self.ranker is not None:
                (idx, sc), = self.rank_many([user_id], [(ci, cs)])
                src = "retrieval+ranker"
            else:
                idx, sc, src = ci, cs, "retrieval"
            idx = idx[:k]
            sc = sc[:k]
        rows = self.items.iloc[idx]
        return [{"item_id": r.item_id, "title": r.title, "genres": r.genres, "score": None if sc is None else float(sc[j]), "source": src}
                for j, r in enumerate(rows.itertuples())][:k]
