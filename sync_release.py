import shutil
import os

files = [
    "README.md",
    "ZENODO_PAPER.md",
    "SCALING_REPORT_120B.md",
    "RTH-LM_TECH_PAPER.html",
    "RTH-LM_TECH_PAPER_STANDALONE.html",
    "LICENSE.md",
    "rth_logo.png"
]

src_dir = r"C:/Users/PC/Desktop/cpu-da"
dst_dir = r"E:/ZETAGRID"

for f in files:
    src = os.path.join(src_dir, f)
    dst = os.path.join(dst_dir, f)
    try:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"✅ Copied: {f}")
        else:
            print(f"⚠️ Missing: {src}")
    except Exception as e:
        print(f"❌ Error copying {f}: {e}")
