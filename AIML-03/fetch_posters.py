import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Project paths
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
LINKS = Path(r"C:\Users\biswa\Downloads\ml-25m\ml-25m\links.csv")
OUTPUT = DATA / "poster_cache.json"

# Load API key
load_dotenv(ROOT / ".env")
API_KEY = os.getenv("TMDB_API_KEY")

if not API_KEY:
    raise ValueError("TMDB_API_KEY not found in .env file")

# Load movies and TMDB mapping
items = pd.read_parquet(DATA / "items.parquet")
links = pd.read_csv(LINKS)

# Convert both movie IDs to integers
items["item_id"] = pd.to_numeric(
    items["item_id"], errors="coerce"
).astype("Int64")

links["movieId"] = pd.to_numeric(
    links["movieId"], errors="coerce"
).astype("Int64")

# Merge MovieLens movies with TMDB IDs
movies = items.merge(
    links[["movieId", "tmdbId"]],
    left_on="item_id",
    right_on="movieId",
    how="left"
)

session = requests.Session()
poster_cache = {}

print(f"Fetching posters for {len(movies)} movies...")

for index, movie in movies.iterrows():
    movie_id = str(int(movie["item_id"]))
    tmdb_id = movie["tmdbId"]

    if pd.isna(tmdb_id):
        poster_cache[movie_id] = None
        continue

    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}"

        response = session.get(
            url,
            params={"api_key": API_KEY},
            timeout=15
        )

        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")

            poster_cache[movie_id] = (
                f"https://image.tmdb.org/t/p/w500{poster_path}"
                if poster_path else None
            )

        elif response.status_code == 404:
            poster_cache[movie_id] = None

        else:
            print(f"TMDB error for {movie_id}: {response.status_code}")
            poster_cache[movie_id] = None

        if index % 100 == 0:
            print(f"Processed {index}/{len(movies)} movies")

        time.sleep(0.05)

    except requests.RequestException as error:
        print(f"Request failed for {movie_id}: {error}")
        poster_cache[movie_id] = None

# Save poster URLs
with open(OUTPUT, "w", encoding="utf-8") as file:
    json.dump(poster_cache, file, indent=2)

available = sum(1 for value in poster_cache.values() if value)

print("\nPOSTER FETCH COMPLETED")
print(f"Total movies: {len(poster_cache)}")
print(f"Posters found: {available}")
print(f"Posters missing: {len(poster_cache) - available}")
print(f"Saved to: {OUTPUT}")