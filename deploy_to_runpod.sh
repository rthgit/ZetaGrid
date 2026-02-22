#!/bin/bash
# ZETAGRID 50B - RUNPOD DEPLOYMENT HELPER
# Use this to prepare your A40/H100 Pod for Sigma SFT.

echo "🌌 Initializing ZetaGrid RunPod Environment..."

# 1. Create Workspace
mkdir -p /workspace/zetagrid_50b/phase4_sft_checkpoints
mkdir -p /workspace/zetagrid_50b/data/sft

# 2. Check Dependencies
echo "📦 Checking Python Dependencies..."
pip install torch numpy gradio huggingface_hub

# 3. Verify and Download Data
if [ ! -f "/workspace/zetagrid_50b/data/sft/merged_finetune_data.jsonl" ]; then
    echo "📥 Downloading SFT Dataset from Hugging Face..."
    huggingface-cli download RthItalia/Rth-lm-25b data/sft/merged_finetune_data.jsonl --local-dir /workspace/zetagrid_50b/
fi

if [ ! -f "/workspace/zetagrid_50b/zetagrid_25b_production.npy" ]; then
    echo "⚠️ Warning: Genome Bank (npy) not found at /workspace/zetagrid_50b/"
fi

# 4. Instructions
echo ""
echo "🚀 Ready to Launch!"
echo "To start the Sigma SFT, run:"
echo "python A40_TRAIN_50B_SIGMA_SFT.py"
echo ""
echo "Monitor loss in the terminal. Checkpoints will be saved to:"
echo "/workspace/zetagrid_50b/phase4_sft_checkpoints/"
