"""Stage 1: two-tower retrieval model (TensorFlow Recommenders). Exports user/item embeddings."""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")   # TFRS needs Keras 2 (pip install tf-keras)
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from config import PROC, ART, EMB_DIM, POS_THRESHOLD


class TwoTower(tfrs.Model):
    def __init__(self, user_ids, item_ids, dim):
        super().__init__()
        def tower(vocab):
            return tf.keras.Sequential([
                tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None),
                tf.keras.layers.Embedding(len(vocab) + 1, dim)])
        self.user_model, self.item_model = tower(user_ids), tower(item_ids)
        self.task = tfrs.tasks.Retrieval()   # in-batch sampled softmax

    def compute_loss(self, features, training=False):
        u = self.user_model(features["user_id"])
        i = self.item_model(features["item_id"])
        return self.task(u, i, candidate_ids=features["item_id"])   # candidate_ids masks accidental hits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args()

    train = pd.read_parquet(PROC / "train.parquet")
    pos = train[train.rating >= POS_THRESHOLD]
    user_ids, item_ids = sorted(pos.user_id.unique()), sorted(pos.item_id.unique())
    ds = (tf.data.Dataset.from_tensor_slices({"user_id": pos.user_id.values, "item_id": pos.item_id.values})
          .shuffle(100_000, seed=0).batch(args.batch).cache())

    model = TwoTower(user_ids, item_ids, EMB_DIM)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
    hist = model.fit(ds, epochs=args.epochs, verbose=2)

    np.save(ART / "user_emb.npy", model.user_model(tf.constant(user_ids)).numpy())
    np.save(ART / "item_emb.npy", model.item_model(tf.constant(item_ids)).numpy())
    json.dump(user_ids, open(ART / "user_ids.json", "w"))
    json.dump(item_ids, open(ART / "item_ids.json", "w"))
    json.dump({"loss": [float(x) for x in hist.history["loss"]]}, open(ART / "retrieval_history.json", "w"))
    print("saved embeddings:", len(user_ids), "users,", len(item_ids), "items")
