import torch
import os
import glob

ckpt_dir = r"D:\MP2\results\checkpoints"
files = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))

print(f"{'File':<40} {'Policy':<6} {'TPC':<5} {'ISR':<8} {'Ep':<6} {'Keys'}")
print("-" * 100)

for f in files:
    name = os.path.basename(f)
    ckpt = torch.load(f, map_location="cpu", weights_only=False)
    keys = sorted(ckpt.keys())
    isr = ckpt.get("isr", "N/A")
    ep = ckpt.get("episode", "N/A")
    
    # Parse policy and tpc from filename
    parts = name.replace("_best.pt", "").split("_")
    policy = parts[1] if len(parts) > 1 else "?"
    tpc = parts[2].replace("tpc", "") if len(parts) > 2 else "?"
    
    print(f"{name:<40} {policy:<6} {tpc:<5} {isr:<8.4f} {ep:<6} {keys}")
