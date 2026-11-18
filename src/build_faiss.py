"""Index item embeddings in FAISS for fast nearest-neighbour search."""
import numpy as np
import faiss
from config import ART

emb = np.load(ART / "item_emb.npy").astype("float32")
# Inner product matches the two-tower training objective (dot product).
# Exact search is fine up to ~100k items. For millions use: faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, str(ART / "items.faiss"))
print("FAISS index:", index.ntotal, "items, dim", emb.shape[1])
