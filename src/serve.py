"""FastAPI service. GET /recommend/{user_id}?k=10  -> cached in Redis (falls back to in-memory if Redis is down).
Run:  uvicorn serve:app --port 8000   (from the src/ folder)"""
import json
import os
import time
from fastapi import FastAPI, Query
from core import Recommender

app = FastAPI(title="Two-stage recommender")
rec = Recommender()
TTL = 300
_mem = {}
try:
    import redis
    r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), socket_connect_timeout=0.3)
    r.ping()
except Exception:
    r = None


def cache_get(key):
    if r is not None:
        v = r.get(key)
        return json.loads(v) if v else None
    hit = _mem.get(key)
    return hit[1] if hit and hit[0] > time.time() else None


def cache_set(key, val):
    if r is not None:
        r.setex(key, TTL, json.dumps(val))
    else:
        _mem[key] = (time.time() + TTL, val)


@app.get("/health")
def health():
    return {"ok": True, "redis": r is not None, "ranker": rec.ranker is not None}


@app.get("/recommend/{user_id}")
def recommend(user_id: str, k: int = 10, mode: str = "ranker", genres: list[str] = Query(default=[])):
    key = f"rec:{user_id}:{k}:{mode}:{','.join(sorted(genres))}"
    t0 = time.perf_counter()
    cached = cache_get(key)
    if cached is not None:
        return {"user_id": user_id, "cached": True, "latency_ms": (time.perf_counter() - t0) * 1000, "items": cached}
    items = rec.recommend(user_id, k, mode, genres=genres)
    cache_set(key, items)
    return {"user_id": user_id, "cached": False, "latency_ms": (time.perf_counter() - t0) * 1000, "items": items}
