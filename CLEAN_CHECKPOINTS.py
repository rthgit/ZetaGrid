import os
import time
import glob

# Config
BASE_DIR = "/workspace/zetagrid_50b"
CKPT_DIR = f"{BASE_DIR}/phase4_sft_checkpoints"
KEEP_LAST_N = 3  # Keep only the last 3 checkpoints for safety
CHECK_INTERVAL = 60 # Check every 60 seconds

def clean_checkpoints():
    print(f"🧹 MONITORING CHECKPOTINS: {CKPT_DIR}")
    print(f"   (Keeping last {KEEP_LAST_N} files)")
    
    while True:
        try:
            files = glob.glob(f"{CKPT_DIR}/*.pt")
            
            # Sort by modification time (latest last)
            files = sorted(files, key=os.path.getmtime)
            
            if len(files) > KEEP_LAST_N:
                # Identify files to delete (all except last N)
                to_delete = files[:-KEEP_LAST_N]
                
                for f in to_delete:
                    # Double check it's not the FINAL one if we name it specially later
                    if "final" in f: continue
                        
                    print(f"🗑️ Deleting old checkpoint: {os.path.basename(f)}")
                    os.remove(f)
            
            else:
                pass # Not enough files to delete yet
                
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")
            
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    if not os.path.exists(CKPT_DIR):
        print(f"⚠️ Folder not found: {CKPT_DIR} - Waiting for creation...")
        while not os.path.exists(CKPT_DIR):
            time.sleep(5)
            
    clean_checkpoints()
