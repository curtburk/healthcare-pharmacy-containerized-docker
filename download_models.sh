#!/bin/bash

# Download GGUF models to the local models/ directory
# These get mounted into the container at runtime

echo "================================================"
echo "Medical AI Model Download Script"
echo "================================================"
echo ""
echo "Medical Finetuned Mixtral Q4 (~28GB)..."


pip install huggingface_hub
huggingface-cli download curtburk/healthcare-polypharmacy-finetune \
    --local-dir ./models \
    --local-dir-use-symlinks False


echo ""
echo "✅ Models downloaded to ./models/"
echo "   You can now run: ./start.sh"
