import os
from huggingface_hub import HfApi

# --- CONFIGURATION ---
REPO_ID = "RthItalia/Rth-lm-25b" # Or a dedicated dataset repo if you prefer
LOCAL_DATA_PATH = r"c:\Users\PC\Desktop\cpu-da\cpu_da_v2\merged_finetune_data.jsonl"
TARGET_NAME = "data/sft/merged_finetune_data.jsonl"

def upload_dataset():
    api = HfApi()
    
    if not os.path.exists(LOCAL_DATA_PATH):
        print(f"❌ Error: Dataset not found at {LOCAL_DATA_PATH}")
        return

    print(f"🚀 Starting Upload of SFT Dataset (1.5GB) to {REPO_ID}...")
    print(f"📦 This might take a few minutes depending on your upload speed...")
    
    try:
        api.upload_file(
            path_or_fileobj=LOCAL_DATA_PATH,
            path_in_repo=TARGET_NAME,
            repo_id=REPO_ID,
            repo_type="model" # Keeping it in the same model repo for convenience
        )
        print(f"✅ Success! Dataset uploaded as: {TARGET_NAME}")
        print(f"\n💡 On your RunPod instance, you can now download it using:")
        print(f"   huggingface-cli download {REPO_ID} {TARGET_NAME} --local-dir /workspace/zetagrid_50b/")
    except Exception as e:
        print(f"❌ Error during upload: {e}")

if __name__ == "__main__":
    upload_dataset()
