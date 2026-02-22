import torch
import sys
import os

# Add the directory to path so we can import architecture if needed
sys.path.append(os.getcwd())

from A40_TRAIN_50B_SIGMA_SFT import SFTDataset, SEQ_LEN

def test_dataset_streaming():
    print("🔍 Testing SFT Dataset Streaming...")
    # Path to the real dataset but we only need a few lines
    data_path = "c:/Users/PC/Desktop/cpu-da/cpu_da_v2/merged_finetune_data.jsonl"
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at {data_path}")
        return

    dataset = SFTDataset(data_path, SEQ_LEN)
    count = 0
    for x, y, m in dataset:
        print(f"✅ Batch {count+1} loaded.")
        print(f"   Shape: {x.shape}")
        print(f"   Mask Sum (Assistant tokens): {m.sum().item()}")
        count += 1
        if count >= 3: break
    
    print("🏆 Dataset Streaming Test Passed.")

if __name__ == "__main__":
    test_dataset_streaming()
