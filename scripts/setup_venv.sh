#!/usr/bin/env bash
# ============================================================
# Setup a virtual environment for the training script.
# Usage:  bash setup_venv.sh
# Then:   source venv_train/bin/activate
#         python train_benchmark.py electronics.parquet
# ============================================================
set -e
echo "[INFO] creating venv 'venv_train'..."
python3 -m venv venv_train
source venv_train/bin/activate
echo "[INFO] upgrading pip..."
pip install --upgrade pip

echo "[INFO] installing packages..."
pip install numpy pandas scikit-learn pyarrow

# PyTorch — CPU build by default (smaller, always works).
# For GPU (CUDA 12.1), comment the line below and use the cu121 line instead.
pip install torch --index-url https://download.pytorch.org/whl/cpu
# pip install torch --index-url https://download.pytorch.org/whl/cu121   # <-- GPU version

# DGL — CPU build by default.
pip install dgl -f https://data.dgl.ai/wheels/repo.html
# For GPU DGL, see https://www.dgl.ai/pages/start.html for the matching CUDA wheel.

echo "[INFO] done. Activate with:  source venv_train/bin/activate"
echo "[INFO] then run:            python train_benchmark.py electronics.parquet"
