# Uploading ZetaGrid 25B to HuggingFace 🤗

GitHub rejected the large files. HuggingFace is the correct home for them.

## 1. Create a Model Repository
Go to [huggingface.co/new](https://huggingface.co/new) and create a new model repo, e.g., `rth-italia/zetagrid-25b`.

## 2. Upload Files (Web Interface - Easiest)
Since you are on Windows and the files are in `E:\ZETAGRID`, the Web UI is the fastest method.

1.  Go to your new Repo page > **Files and versions**.
2.  Click **Add file** > **Upload files**.
3.  Drag and drop these files from `E:\ZETAGRID`:
    *   `zetagrid_25b_production.npy` (The Genome - 7GB)
    *   `zeta25b_step15000.pt` (The Weights - 500MB)
    *   `zeta25b_2bit.qulp` (The Quantized Model - 240MB)
    *   `README.md` (The beautiful one we made)
    *   `ZETAGRID_INFERENCE.py`
4.  Click **Commit changes**.

## 3. Alternative: Using CLI (PowerShell)
If you prefer the command line:

```powershell
pip install huggingface_hub

# Login
huggingface-cli login

# Upload Genome
huggingface-cli upload rth-italia/zetagrid-25b E:\ZETAGRID\zetagrid_25b_production.npy --repo-type model

# Upload Checkpoint
huggingface-cli upload rth-italia/zetagrid-25b E:\ZETAGRID\zeta25b_step15000.pt --repo-type model

# Upload Quantized
huggingface-cli upload rth-italia/zetagrid-25b E:\ZETAGRID\zeta25b_2bit.qulp --repo-type model
```

## 4. Final Polish
Once uploaded, your README.md will automatically render the badges and instructions.
The Release is then **100% COMPLETE**.
