#!/bin/bash
# requirements.sh
# Sets up the GestureBall CV environment
# Tested on Ubuntu 22.04 / macOS 14
# Run: bash requirements.sh

set -e  # exit on error

echo "=== GestureBall CV Setup ==="

# ── Python version check ──────────────────────────────────────────────────────
python3 --version | grep -E "3\.(9|10|11|12)" > /dev/null 2>&1 || {
    echo "ERROR: Python 3.9–3.12 required. Found: $(python3 --version)"
    exit 1
}
echo "✓ Python $(python3 --version)"

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
echo "✓ Virtual environment active"

# ── Pip upgrade ───────────────────────────────────────────────────────────────
pip install --upgrade pip --quiet

# ── Core dependencies ─────────────────────────────────────────────────────────
echo "Installing dependencies..."

pip install --quiet \
    mediapipe>=0.10.0 \
    opencv-python>=4.8.0 \
    numpy>=1.24.0 \
    scikit-learn>=1.3.0 \
    matplotlib>=3.7.0 \
    joblib>=1.3.0

# ── PyTorch (CPU version — sufficient for inference and light training) ────────
echo "Installing PyTorch (CPU)..."
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ── Verify installs ───────────────────────────────────────────────────────────
echo ""
echo "=== Verifying installs ==="
python3 -c "import cv2;        print(f'✓ OpenCV     {cv2.__version__}')"
python3 -c "import mediapipe;  print(f'✓ MediaPipe  {mediapipe.__version__}')"
python3 -c "import torch;      print(f'✓ PyTorch    {torch.__version__}')"
python3 -c "import numpy;      print(f'✓ NumPy      {numpy.__version__}')"
python3 -c "import sklearn;    print(f'✓ sklearn    {sklearn.__version__}')"
python3 -c "import matplotlib; print(f'✓ matplotlib {matplotlib.__version__}')"

echo ""
echo "=== Setup complete ==="
echo "To activate the environment in future sessions:"
echo "  source venv/bin/activate"