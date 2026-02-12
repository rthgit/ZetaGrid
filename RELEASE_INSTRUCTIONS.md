# ZETAGRID 25B - OFFICIAL RELEASE INSTRUCTIONS 📦

Follow these steps to package the first **Fractal TCN Model** for the public.

## 1. Create Release Folder
Create a focused folder, e.g., `ZETAGRID_25B_RELEASE_v1`.

## 2. Gather Files
Copy the following files into the folder:

| File | Source | Description |
| :--- | :--- | :--- |
| `zeta25b_step15000.pt` | (Downloaded) | **Main Model Weights** (trainable params only, ~500MB) |
| `zetagrid_25b_production.npy` | `E:\ZETAGRID\` | **Genome Backbone** (Frozen DNA, ~7GB) |
| `zeta25b_2bit.qulp` | `E:\ZETAGRID\` | **Quantized Model** (Ultra-Light Inference) |
| `ZETAGRID_INFERENCE.py` | `Desktop\cpu-da\` | **Python Inference Script** |
| `ZETAGRID_25B_CARD.md` | `Desktop\cpu-da\` | **Model Card / Documentation** |

## 3. Verify Integrity
Ensure `zeta25b_step15000.pt` and `zetagrid_25b_production.npy` are both present. The model CANNOT run without the Genome.

## 4. Zip & Publish
1.  Zip the folder (exclude the `.npy` if hosting separately due to size, but include instructions to download it).
2.  Upload to:
    *   **HuggingFace:** Upload `step15000.pt` and `ZETAGRID_INFERENCE.py`. Add the Model Card content to the repo `README.md`.
    *   **GitHub:** Push the code and documentation. Use `git lfs` for the weights if needed.

## 5. Announcement
Use `ZETAGRID_25B_CARD.md` as the base for your announcement post (social media/blog).

---
*RTH Italia - Advanced AI Systems*
