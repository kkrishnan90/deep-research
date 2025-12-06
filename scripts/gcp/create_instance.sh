#!/bin/bash
# Create GCP VM with NVIDIA L4 GPU for ROS Gazebo simulation
# Usage: ./create_instance.sh

set -e

# Configuration
PROJECT_ID="account-pocs"
INSTANCE_NAME="ros-kinova-sim"
ZONE="us-central1-a"
REGION="us-central1"
MACHINE_TYPE="g2-standard-8"  # 8 vCPU, 32GB RAM, 1x L4 GPU (24GB VRAM)

echo "=========================================="
echo "  Creating GCP Instance with NVIDIA L4"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Instance: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo "Machine Type: $MACHINE_TYPE"
echo "=========================================="

# Check if instance already exists
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID &>/dev/null; then
    echo "Instance $INSTANCE_NAME already exists. Delete it first or choose a different name."
    exit 1
fi

# Create the instance
echo "Creating instance..."
gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --image-family=ubuntu-pro-2004-lts \
    --image-project=ubuntu-os-pro-cloud \
    --maintenance-policy=TERMINATE \
    --restart-on-failure \
    --tags=http-server,https-server \
    --metadata=enable-oslogin=TRUE

echo "Instance created successfully!"

# Get external IP
EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo ""
echo "=========================================="
echo "  Instance Details"
echo "=========================================="
echo "Name: $INSTANCE_NAME"
echo "External IP: $EXTERNAL_IP"
echo "Zone: $ZONE"
echo ""
echo "To SSH into the instance:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
echo ""
echo "Next steps:"
echo "1. SSH into the instance"
echo "2. Clone this repository"
echo "3. Run: bash scripts/gcp/master_setup.sh"
echo "=========================================="
