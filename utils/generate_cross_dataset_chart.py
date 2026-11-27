import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ────────────────────────────────────────────────────────────────────
models = ["VideoMAE", "3D CNN", "Ensemble"]
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

data = {
    "VideoMAE":  [63.0, 67.44, 73.11, 70.16],
    "3D CNN":    [61.5, 66.41, 71.43, 68.83],
    "Ensemble":  [63.5, 67.42, 74.79, 70.92],
}

colors = {
    "VideoMAE": "#4C72B0",
    "3D CNN":   "#DD8452",
    "Ensemble": "#55A868",
}

x = np.arange(len(metrics))
width = 0.25
offsets = [-width, 0, width]

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for i, model in enumerate(models):
    bars = ax.bar(x + offsets[i], data[model], width,
                  label=model, color=colors[model], edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, data[model]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

# ── Styling ──────────────────────────────────────────────────────────────────
ax.set_xlabel("Metric", fontsize=12)
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Cross-Dataset Validation (Set B — 912 Unseen Videos)\nModel Comparison Across All Metrics", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 90)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=10)

# Annotate generalisation gap
ax.axhline(y=63.5, color="#55A868", linestyle=":", linewidth=1.2, alpha=0.5)
ax.text(3.38, 64.2, "Ensemble Acc: 63.5%", color="#55A868", fontsize=8)

plt.tight_layout()
out_path = "c:/Users/XinYi/Fyp/result/cross_dataset/cross_dataset_overall.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
