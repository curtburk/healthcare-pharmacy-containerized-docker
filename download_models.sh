#!/bin/bash

# Download GGUF models to the local models/ directory
# These get mounted into the container at runtime

echo "================================================"
echo "Medical AI Model Download Script"
echo "================================================"
echo ""
echo "Medical Finetuned Mixtral Q4 (~28GB)..."

cd /healthcare-pharmacy-containerized-docker/

echo "creating local models folder"
mkdir models

pip install huggingface_hub
from huggingface_hub import snapshot_download

snapshot_download(repo_id="curtburk/healthcare-polypharmacy-finetune", local_dir="./models")


echo ""
echo "✅ Models downloaded to ./models/"
echo "   You can now run: ./start.sh"
