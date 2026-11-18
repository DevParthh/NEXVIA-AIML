
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROC


def synthetic(n_users=2000, n_items=600, n_ratings=150000, seed=0):
    rng = np.random.default_rng(seed)

    genres_list = [
        "Action", "Comedy", "Drama", "Horror",
        "Romance", "SciFi", "Thriller", "Animation"
    ]

    items = pd.DataFrame({
        "item_id": [str(i) for i in range(1, n_items + 1)],
        "title": [f"Movie {i}" for i in range(1, n_items + 1)],
    })

    items["genres"] = [
        "|".join(rng.choice(
            genres_list,
            size=rng.integers(1, 3),
            replace=False
        ))
        for _ in range(n_items)
    ]

    genre_matrix = np.zeros((n_items, len(genres_list)))

    for i, value in enumerate(items["genres"]):
        for genre in value.split("|"):
            genre_matrix[i, genres_list.index(genre)] = 1

    taste = rng.dirichlet(
        np.ones(len(genres_list)) * 0.3,
        size=n_users
    )

    popularity = rng.lognormal(0, 1.2, size=n_items)
    popularity /= popularity.sum()

    users = rng.integers(0, n_users, size=n_ratings)

    candidates = rng.choice(
        n_items, size=(n_ratings, 5), p=popularity
    )

    affinity = np.einsum(
        "ng,ncg->nc",
        taste[users],
        genre_matrix[candidates]
    ) / np.maximum(genre_matrix[candidates].sum(2), 1)

    selected = (
        affinity + rng.gumbel(size=affinity.shape) * 0.05
    ).argmax(1)

    item_indices = candidates[np.arange(n_ratings), selected]

    ratings = np.clip(
        np.round(
            2 + 8 * affinity[np.arange(n_ratings), selected]
            + rng.normal(0, 0.6, n_ratings)
        ),
        1, 5
    )

    timestamps = rng.integers(
        1500000000, 1600000000, size=n_ratings
    )

    df = pd.DataFrame({
        "user_id": ["u" + str(u) for u in users],
        "item_id": [str(i + 1) for i in item_indices],
        "rating": ratings,
        "ts": timestamps
    })

    return df.drop_duplicates(["user_id", "item_id"]), items


def load_movielens(ml_dir, sample_users=None, seed=0):
    ml_dir = Path(ml_dir)

    ratings_path = ml_dir / "ratings.csv"
    movies_path = ml_dir / "movies.csv"

    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing file: {ratings_path}")

    if not movies_path.exists():
        raise FileNotFoundError(f"Missing file: {movies_path}")

    print("PASS 1: Reading user IDs...", flush=True)

    user_ids = set()

    for chunk in pd.read_csv(
        ratings_path,
        usecols=["userId"],
        chunksize=500000
    ):
        user_ids.update(chunk["userId"].unique())

    user_ids = np.array(sorted(user_ids))

    print(f"Total users: {len(user_ids):,}", flush=True)

    if sample_users:
        rng = np.random.default_rng(seed)
        selected = set(
            rng.choice(
                user_ids,
                size=min(sample_users, len(user_ids)),
                replace=False
            )
        )
    else:
        selected = set(user_ids)

    print(f"Selected users: {len(selected):,}", flush=True)
    print("PASS 2: Reading selected ratings...", flush=True)

    parts = []

    for number, chunk in enumerate(pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "rating", "timestamp"],
        chunksize=500000
    ), start=1):

        filtered = chunk[chunk["userId"].isin(selected)]

        if not filtered.empty:
            parts.append(filtered)

        print(f"Processed chunk {number}", flush=True)

    if not parts:
        raise ValueError("No ratings found for selected users.")

    r = pd.concat(parts, ignore_index=True)
    del parts

    df = pd.DataFrame({
        "user_id": "u" + r["userId"].astype(str),
        "item_id": r["movieId"].astype(str),
        "rating": r["rating"],
        "ts": r["timestamp"]
    })

    del r

    df = df.drop_duplicates(["user_id", "item_id"])

    print("Reading movie metadata...", flush=True)

    m = pd.read_csv(
        movies_path,
        usecols=["movieId", "title", "genres"]
    )

    items = pd.DataFrame({
        "item_id": m["movieId"].astype(str),
        "title": m["title"],
        "genres": m["genres"].replace("(no genres listed)", "")
    })

    items = items[items["item_id"].isin(df["item_id"].unique())].copy()

    print(f"Ratings loaded: {len(df):,}", flush=True)
    print(f"Movies loaded: {len(items):,}", flush=True)

    return df, items


def time_split(df):
    df = df.sort_values("ts").reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, val, test


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--ml-dir", type=str)
    parser.add_argument("--sample-users", type=int)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.synthetic:
        print("Generating synthetic data...", flush=True)
        df, items = synthetic(seed=args.seed)

    elif args.ml_dir:
        print("Loading MovieLens dataset...", flush=True)
        df, items = load_movielens(
            args.ml_dir,
            sample_users=args.sample_users,
            seed=args.seed
        )

    else:
        parser.error("Pass --synthetic or --ml-dir")

    print("Filtering interactions...", flush=True)

    for _ in range(2):
        df = df[
            df.groupby("user_id")["item_id"].transform("size") >= 5
        ]
        df = df[
            df.groupby("item_id")["user_id"].transform("size") >= 5
        ]

    if df.empty:
        raise ValueError("No interactions remain after filtering.")

    items = items[items["item_id"].isin(df["item_id"].unique())].copy()

    print("Splitting data by timestamp...", flush=True)

    train, val, test = time_split(df)

    PROC.mkdir(parents=True, exist_ok=True)

    for name, part in [
        ("train", train),
        ("val", val),
        ("test", test)
    ]:
        part.to_parquet(PROC / f"{name}.parquet", index=False)

        print(
            f"{name}: {len(part):,} rows, "
            f"{part['user_id'].nunique():,} users",
            flush=True
        )

    items.to_parquet(PROC / "items.parquet", index=False)

    print(f"items: {len(items):,}", flush=True)
    print(f"Saved to: {PROC}", flush=True)
    print("DATA PREPARATION COMPLETED!", flush=True)


if __name__ == "__main__":
    main()
