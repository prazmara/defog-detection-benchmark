import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import linregress

# Data
methods = [
    "Ground Truth (GT)",
    "DehazeFormer-trained",
    "DehazeFormer",
    "Flux COT",
    "Flux (Not COT)",
    "FocalNet",
    "Fog density β=0.01"
]
# [GT, DehazeFormer-trained, DehazeFormer, Flux (Improved), FLUX cot, Flux (Not COT), FocalNet, Fog density β=0.01]
mAP = [ 0.256,
0.2557,
0.2379,
0.2376,
0.2306,
0.2353,
0.2295]
scores = [29.85,
        26.36,
        20.40,
        23.70,
        16.98,
        19.26,
        15.09]
df = pd.DataFrame({"Method": methods, "PQ": mAP, "Score": scores})

# Regression line
slope, intercept, r_value, _, _ = linregress(df["mAP"], df["Score"])
line_x = np.linspace(min(mAP), max(mAP), 200)
line_y = slope * line_x + intercept

# Figure
plt.figure(figsize=(7.2, 5.5), dpi=300)

# Plot regression line
plt.plot(line_x, line_y, "--", color="firebrick", linewidth=2,
         label=f"Linear Fit (r = {r_value:.2f})")

# Plot points
for method, x, y in zip(methods, mAP, scores):
    if method == "Ground Truth (GT)":
        plt.scatter(x, y, color="gold", edgecolor="black", s=150, marker="*", zorder=3, label="Ground Truth")
    elif method == "FocalNet":
        plt.scatter(x, y, color="royalblue", edgecolor="black", s=70, marker="o", zorder=3, label="Methods")
    else:
        plt.scatter(x, y, color="royalblue", edgecolor="black", s=70, marker="o", zorder=3)

# Specify offsets for each label (dx, dy)
label_offsets = {
    "Ground Truth (GT)": (-0.003, 1),
    "DehazeFormer-trained": (-0.005, -1.1),
    "DehazeFormer": (+0.001, -0.2),
    "Flux COT": (-0.003, 1),
    "Flux (Not COT)": (-0.003, 1),
    "FocalNet": (-0.001, -1.),
    "Fog density β=0.01": (-0.003, -1.1)
}

# Add labels with custom offsets
for method, x, y in zip(methods, mAP, scores):
    dx, dy = label_offsets.get(method, (0.000, 1.0))  # default offset if not specified
    plt.text(x + dx, y + dy, method, fontsize=9, family="Times New Roman")

# Axis labels & title
plt.xlabel("mAP", fontsize=13, family="Times New Roman")
plt.ylabel("Total Qualitative Score", fontsize=13, family="Times New Roman")
plt.title("Correlation between mAP and Qualitative Score",
          fontsize=14, family="Times New Roman", pad=15)

# Axis limits
plt.xlim(min(mAP) - 0.006, max(mAP) + 0.006)
plt.ylim(min(scores) - 3, max(scores) + 3)

# Grid & style
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend(frameon=True, fontsize=10, loc="upper left", edgecolor="lightgray")
plt.xticks(fontsize=11, family="Times New Roman")
plt.yticks(fontsize=11, family="Times New Roman")
plt.savefig("correlation.png", bbox_inches="tight", dpi=300)



