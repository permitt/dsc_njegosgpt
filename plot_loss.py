"""Plots the loss curve from out/log.csv to out/loss.png."""

import csv
import os
import shutil
import sys

import matplotlib.pyplot as plt

out_dir = sys.argv[1] if len(sys.argv) > 1 else "out"
with open(f"{out_dir}/log.csv") as f:
    rows = list(csv.DictReader(f))
steps = [int(r["step"]) for r in rows]

plt.figure(figsize=(8, 4.5))
plt.plot(steps, [float(r["train_loss"]) for r in rows], label="training")
plt.plot(steps, [float(r["val_loss"]) for r in rows], label="validation")
plt.xlabel("step")
plt.ylabel("loss (cross-entropy)")
plt.title("NjegošGPT: learning curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out_dir}/loss.png", dpi=150)
print(f"saved: {out_dir}/loss.png")
if os.path.isdir("presentation"):  # the slides load their own copy
    shutil.copyfile(f"{out_dir}/loss.png", "presentation/loss.png")
    print("copied to presentation/loss.png")
