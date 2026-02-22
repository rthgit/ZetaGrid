#!/usr/bin/env python3
"""
ZETAGRID REPAIR DATASET PREPARATOR (A40)
========================================
Goal: Download standard knowledge datasets (WikiText, C4 subset)
and combine with 'golden_mix_220b.jsonl' to create a repair dataset.
"""

import json
import os
import random
from datasets import load_dataset # Requires `pip install datasets`

BASE_DIR = "/workspace/zetagrid_50b"
GOLDEN_MIX = f"{BASE_DIR}/golden_mix_220b.jsonl" # Uploaded by user
OUTPUT_FILE = f"{BASE_DIR}/repair_mix.jsonl"

def main():
    print("🚀 PREPARING DATASET FOR REPAIR...")
    
    # 1. Load Golden Mix (Uploaded from PC)
    golden_data = []
    if os.path.exists(GOLDEN_MIX):
        print(f"📖 Loading Golden Mix: {GOLDEN_MIX}")
        with open(GOLDEN_MIX, 'r') as f:
            golden_data = [json.loads(line) for line in f]
        print(f"✅ Loaded {len(golden_data)} Golden examples.")
    else:
        print(f"⚠️  Golden Mix not found at {GOLDEN_MIX}. Assuming you will upload it later.")
        
    # 2. Download World Knowledge (WikiText-103)
    # This is small enough (~180MB) to be fast but has good density.
    print("🌍 Downloading WikiText-103 (Knowledge injection)...")
    try:
        wiki = load_dataset("wikitext", "wikitext-103-v1", split="train")
        # WikiText uses 'text' field.
        # We need to format it as ChatML for compatibility?
        # Or just raw pre-training?
        # Our model is SFT. If we feed raw text, it might forget Chat format.
        # STRATEGY: Wrap Wiki text in "User: Read this.\nAssistant: {text}"?
        # OR: Just "Assistant: {text}" (Partial injection).
        # BETTER: "User: Tell me about X.\nAssistant: {text}" (Synthetic).
        # SIMPLEST: Just use raw text with `User: Continue...\nAssistant: {text}`
        
        wiki_data = []
        for item in wiki:
            text = item['text']
            if len(text) < 100: continue # Skip short headers
            
            # Synthetic Prompt: "Explain this topic." or "Continue."
            entry = {
                "messages": [
                    {"role": "user", "content": "Elaborate on the following topic."},
                    {"role": "assistant", "content": text}
                ]
            }
            wiki_data.append(entry)
            
        # Limit to 50k examples to keep it "Fast Repair"
        random.shuffle(wiki_data)
        wiki_data = wiki_data[:50000]
        print(f"✅ Extracted {len(wiki_data)} WikiText paragraphs.")
        
    except Exception as e:
        print(f"❌ Failed to download WikiText: {e}")
        wiki_data = []
        
    # 3. Download C4 (Common Crawl Clean) - 1GB Subset
    print("🕸️  Downloading C4 (Real-world Knowledge) - Streaming 1GB...")
    try:
        # Stream the dataset to avoid downloading petabytes
        c4 = load_dataset("allenai/c4", "en", split="train", streaming=True)
        
        c4_data = []
        c4_size_bytes = 0
        limit_bytes = 1024 * 1024 * 1024 # 1 GB
        
        for item in c4:
            text = item['text']
            if len(text) < 200: continue # Skip tiny fragments
            
            # Simple wrapper to maintain chat consistency in the file
            # "User: ... \nAssistant: {text}"
            entry = {
                "messages": [
                    {"role": "user", "content": "Article:"}, # Minimal prompt
                    {"role": "assistant", "content": text}
                ]
            }
            
            c4_data.append(entry)
            c4_size_bytes += len(text)
            
            if c4_size_bytes >= limit_bytes:
                print(f"   Reached 1GB limit ({len(c4_data)} docs). Stopping stream.")
                break
                
        print(f"✅ Extracted {len(c4_data)} C4 documents.")
    except Exception as e:
        print(f"❌ Failed to stream C4: {e}")
        c4_data = []
    
    # Combine
    # Golden Mix (Higher Weight x10) + WikiText (x1) + C4 (x1)
    # 25B needs repetition of instructions to not forget format.
    repair_data = (golden_data * 10) + wiki_data + c4_data
    random.shuffle(repair_data)
    
    print(f"💾 Saving {len(repair_data)} total examples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for entry in repair_data:
            f.write(json.dumps(entry) + "\n")
            
    print("✨ REPAIR DATASET READY.")

if __name__ == "__main__":
    main()
