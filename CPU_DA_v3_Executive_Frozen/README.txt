# Executive Product: CPU-DA v3.0 (Frozen Release)

This is the stable, frozen release of the CPU-DA v3.0 Training Engine.
Engine optimized for AVX2 with "Beta=0" Fix for zero-alloc safety.

## Performance
- Speed: ~1.98 seconds/step (on 8 threads)
- Stability: No NaNs (Verified)

## How to Run
1. Go to `cpu_da_v2` folder.
2. Run `./run_executive.sh` (or `bash run_executive.sh`).

## Requirements
- This folder must remain next to `cpu_da_framework` (structure is preserved).
- You must copy `morph_specialized_text_data.bin` and `morph_checkpoint_latest.bin` into the `cpu_da_v2` folder to resume training.
