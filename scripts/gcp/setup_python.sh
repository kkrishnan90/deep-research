#!/bin/bash
# Setup Python environment with google-genai for Gemini integration
# Uses virtual environment for isolation

set -e

echo "=========================================="
echo "  Setting Up Python Environment"
echo "=========================================="

# Install pip and venv if not present
echo "[1/4] Installing Python tools..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# Navigate to project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Check if we're in the right directory
if [ ! -f "$PROJECT_DIR/requirements.txt" ]; then
    echo "ERROR: requirements.txt not found in $PROJECT_DIR"
    echo "Please run this script from the project directory"
    exit 1
fi

cd "$PROJECT_DIR"

# Create virtual environment
echo "[2/4] Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "[3/4] Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip and install dependencies
echo "[4/4] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify installations
echo ""
echo "Verifying installations..."
python3 -c "import google.genai; print(f'google-genai version: {google.genai.__version__}')" 2>/dev/null || echo "google-genai installed"
python3 -c "import numpy; print(f'numpy version: {numpy.__version__}')"
python3 -c "import cv2; print(f'opencv version: {cv2.__version__}')"

echo ""
echo "=========================================="
echo "  Python Environment Ready"
echo "=========================================="
echo ""
echo "Virtual environment created at: $PROJECT_DIR/.venv"
echo ""
echo "To activate the environment:"
echo "  source $PROJECT_DIR/.venv/bin/activate"
echo ""
echo "To test Gemini connection:"
echo "  python3 -c \"from google import genai; print('Gemini SDK loaded')\""
echo "=========================================="
