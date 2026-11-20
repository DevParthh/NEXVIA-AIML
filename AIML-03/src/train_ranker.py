"""Stage 2: deep ranking model (Deep & Cross Network, TFRS) that re-scores FAISS candidates.
Trained on retrieved candidates, labelled with the VALIDATION period -> no leakage from the test period."""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from sklearn.metrics import roc_auc_score
from config import PROC, ART, POS_THRESHOLD
from core import Recommender


def build_ranker(dim):
    x0 = tf.keras.Input(shape=(dim,))
    x0n = tf.keras.layers.BatchNormalization()(x0)
    cross = x0n
    for _ in range(2):
        cross = tfrs.layers.dcn.Cross()(x0n, cross)
    deep = tf.keras.layers.Dense(64, activation="relu")(x0n)
    deep = tf.keras.layers.Dense(32, activation="relu")(deep)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(tf.keras.layers.Concatenate()([cross, deep]))
    m = tf.keras.Model(x0, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--n-cand", type=int, default=100)
    args = ap.parse_args()

    rec = Recommender(seen_splits=("train",), load_ranker=False)
    val = pd.read_parquet(PROC / "val.parquet")
    val = val[(val.rating >= POS_THRESHOLD) & val.item_id.isin(rec.store.item_index)]
    val_pos = val.groupby("user_id").item_id.apply(lambda s: {rec.store.item_index[i] for i in s}).to_dict()
    users = [u for u in val_pos if rec.knows(u)]

    X, y, grp = [], [], []
    for n, u in enumerate(users):
        ci, cs = rec.candidates(u, args.n_cand)
        X.append(rec.store.featurize([u] * len(ci), ci, cs))
        y.append(np.array([c in val_pos[u] for c in ci], dtype=np.float32))
        grp.append(np.full(len(ci), n))
    X, y, grp = np.concatenate(X), np.concatenate(y), np.concatenate(grp)
    print(f"ranker data: {len(y):,} rows, positive rate {y.mean():.4f}, {len(users)} users")

    rng = np.random.default_rng(0)
    holdout = rng.random(len(users)) < 0.2            # split by USER so candidates of one user never straddle
    te = holdout[grp]
    model = build_ranker(X.shape[1])
    model.fit(X[~te], y[~te], validation_data=(X[te], y[te]), epochs=args.epochs, batch_size=512, verbose=2)
    p = model.predict(X[te], batch_size=8192, verbose=0)[:, 0]
    auc_rank, auc_retr = roc_auc_score(y[te], p), roc_auc_score(y[te], X[te][:, 0])
    print(f"holdout AUC  ranker={auc_rank:.4f}  retrieval-score={auc_retr:.4f}")
    json.dump({"auc_ranker": auc_rank, "auc_retrieval": auc_retr, "n_rows": int(len(y)), "pos_rate": float(y.mean())},
              open(ART / "ranker_metrics.json", "w"), indent=2)
    model.save(ART / "ranker", save_format="tf")
    print("saved ranker")
