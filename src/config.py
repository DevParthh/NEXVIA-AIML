from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
ART = ROOT / "artifacts"
PROC.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)

EMB_DIM = 32
POS_THRESHOLD = 4.0      # rating >= 4 counts as a positive interaction
N_CANDIDATES = 200       # how many items stage 1 (FAISS) hands to stage 2 (ranker)
