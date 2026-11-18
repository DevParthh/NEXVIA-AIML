#!/usr/bin/env bash
# End-to-end check on SYNTHETIC data (~2 min on a laptop CPU). Numbers it prints are NOT results to report.
set -e
cd "$(dirname "$0")/src"
export TF_USE_LEGACY_KERAS=1 TF_CPP_MIN_LOG_LEVEL=3
python data_prep.py --synthetic
python train_retrieval.py --epochs 10
python build_faiss.py
python train_ranker.py --epochs 5
python evaluate.py
python ab_test.py
python export_triton.py
cd .. && python -m pytest -q tests
