#!/bin/bash
# Install NVIDIA drivers for L4 GPU on Ubuntu 20.04
# This script will reboot the system after installation

set -e

echo "=========================================="
echo "  Installing NVIDIA Drivers for L4 GPU"
echo "=========================================="

# Check if drivers are already installed
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers already installed:"
    nvidia-smi
    echo ""
    echo "If you want to reinstall, first run:"
    echo "  sudo apt-get remove --purge '^nvidia-.*'"
    exit 0
fi

# Update package list
echo "[1/4] Updating package list..."
sudo apt-get update

# Remove any existing NVIDIA installations
echo "[2/4] Removing any existing NVIDIA installations..."
sudo apt-get remove --purge -y '^nvidia-.*' 2>/dev/null || true
sudo apt-get remove --purge -y '^libnvidia-.*' 2>/dev/null || true
sudo apt-get autoremove -y

# Install ubuntu-drivers-common
echo "[3/4] Installing ubuntu-drivers-common..."
sudo apt-get install -y ubuntu-drivers-common

# Install recommended NVIDIA driver
echo "[4/4] Installing NVIDIA driver..."
# For L4 GPU, driver 535 or newer is recommended
sudo ubuntu-drivers autoinstall

echo ""
echo "=========================================="
echo "  NVIDIA Driver Installation Complete"
echo "=========================================="
echo ""
echo "The system needs to reboot to load the drivers."
echo "After reboot, run 'nvidia-smi' to verify installation."
echo ""
read -p "Reboot now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo reboot
else
    echo "Please reboot manually with: sudo reboot"
fi
