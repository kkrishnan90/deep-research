#!/bin/bash
# Sync local changes to GCP via Git
# Usage: ./sync_to_gcp.sh [commit message]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_DIR"

# Get commit message from argument or use default
COMMIT_MSG="${1:-Update from local development}"

echo "=========================================="
echo "  Syncing to GCP"
echo "=========================================="

# Check for changes
if [ -z "$(git status --porcelain)" ]; then
    echo "No changes to commit."
    echo "Pushing any unpushed commits..."
    git push origin main
    exit 0
fi

# Show changes
echo "Changes to be committed:"
git status --short

# Stage all changes
echo ""
echo "Staging changes..."
git add .

# Commit
echo "Committing..."
git commit -m "$COMMIT_MSG"

# Push to origin
echo "Pushing to GitHub..."
git push origin main

echo ""
echo "=========================================="
echo "  Sync Complete"
echo "=========================================="
echo ""
echo "To pull changes on GCP:"
echo "  cd ~/catkin_ws/src/deep-research && git pull"
echo ""
echo "To rebuild the ROS workspace:"
echo "  cd ~/catkin_ws && catkin build"
echo "=========================================="
