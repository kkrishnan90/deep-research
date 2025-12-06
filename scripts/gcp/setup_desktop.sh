#!/bin/bash
# Setup Chrome Remote Desktop for GUI access to Gazebo
# Reference: https://cloud.google.com/architecture/chrome-desktop-remote-on-compute-engine

set -e

echo "=========================================="
echo "  Installing Chrome Remote Desktop"
echo "=========================================="

# Update packages
echo "[1/4] Updating package list..."
sudo apt-get update

# Install Ubuntu Desktop (GNOME)
echo "[2/4] Installing Ubuntu Desktop..."
echo "This may take several minutes..."
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ubuntu-desktop

# Download and install Chrome Remote Desktop
echo "[3/4] Installing Chrome Remote Desktop..."
wget -q https://dl.google.com/linux/direct/chrome-remote-desktop_current_amd64.deb
sudo apt-get install -y ./chrome-remote-desktop_current_amd64.deb
rm chrome-remote-desktop_current_amd64.deb

# Configure Chrome Remote Desktop to use GNOME
echo "[4/4] Configuring Chrome Remote Desktop..."
sudo bash -c 'echo "exec /etc/X11/Xsession /usr/bin/gnome-session" > /etc/chrome-remote-desktop-session'

# Add user to chrome-remote-desktop group
sudo usermod -a -G chrome-remote-desktop $USER

echo ""
echo "=========================================="
echo "  Chrome Remote Desktop Installed"
echo "=========================================="
echo ""
echo "To complete setup, follow these steps:"
echo ""
echo "1. On your local machine, visit:"
echo "   https://remotedesktop.google.com/headless"
echo ""
echo "2. Click 'Begin' -> 'Next' -> 'Authorize'"
echo ""
echo "3. Select 'Debian Linux' and copy the command"
echo ""
echo "4. Paste and run the command on this VM"
echo ""
echo "5. Set a 6-digit PIN when prompted"
echo ""
echo "6. Access your desktop at:"
echo "   https://remotedesktop.google.com/access"
echo ""
echo "=========================================="
