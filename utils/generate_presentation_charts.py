import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PURPLE_DARK = '#4B0082'
PURPLE_MID = '#7B2FBE'
PURPLE_LIGHT = '#E6E6FA'
PURPLE_ACCENT = '#9B59B6'
GREEN = '#2ECC71'
RED = '#E74C3C'
GREY = '#7F8C8D'

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# ─────────────────────────────────────────────
# Chart 1: F1-Score comparison (Slide 5)
# ─────────────────────────────────────────────
models = ['VideoMAE', 'R(2+1)D CNN', 'Video Swin', 'GenConViT VAE', 'Ensemble\n(MAE+CNN)']
f1_scores = [85.96, 83.74, 85.52, 94.53, 88.11]
colors = [PURPLE_MID, PURPLE_MID, PURPLE_MID, GREY, PURPLE_DARK]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(models, f1_scores, color=colors, edgecolor='white', height=0.55)

for bar, val in zip(bars, f1_scores):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}%', va='center', fontsize=11, fontweight='bold', color='#2c2c2c')

ax.set_xlabel('F1-Score (%)', fontsize=12, labelpad=8)
ax.set_title('Model F1-Score Comparison — Set A (Known Domain)', fontsize=13, fontweight='bold', pad=14, color=PURPLE_DARK)
ax.set_xlim(70, 100)
ax.axvline(x=88.11, color=PURPLE_DARK, linestyle='--', linewidth=1.2, alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=11)

legend_patches = [
    mpatches.Patch(color=PURPLE_DARK, label='Ensemble (deployed)'),
    mpatches.Patch(color=PURPLE_MID, label='Individual model'),
    mpatches.Patch(color=GREY, label='Excluded from ensemble'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=10)

note = '* GenConViT VAE excluded: outputs reconstruction error,\n  not a class probability — incompatible with fusion formula'
ax.text(70.3, -0.7, note, fontsize=9, color=GREY, style='italic')

plt.tight_layout()
plt.savefig('chart1_f1_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 1 saved: chart1_f1_comparison.png")


# ─────────────────────────────────────────────
# Chart 2: Ablation study line graph (Slide 6)
# ─────────────────────────────────────────────
mae_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
f1_values =   [86.25, 86.65, 86.70, 86.75, 87.00, 86.97, 86.64, 86.38, 86.03, 85.68, 85.29]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(mae_weights, f1_values, color=PURPLE_MID, linewidth=2.5, marker='o',
        markersize=7, markerfacecolor=PURPLE_DARK, markeredgecolor='white', markeredgewidth=1.5)

# highlight peak
peak_x, peak_y = 0.4, 87.00
ax.axvline(x=peak_x, color=PURPLE_DARK, linestyle='--', linewidth=1.5, alpha=0.7)
ax.scatter([peak_x], [peak_y], color=PURPLE_DARK, s=120, zorder=5)
ax.annotate(f'Best: MAE(0.4)+CNN(0.6)\nF1 = {peak_y}%',
            xy=(peak_x, peak_y), xytext=(0.5, 86.5),
            fontsize=10, color=PURPLE_DARK, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=PURPLE_DARK, lw=1.5))

ax.set_xlabel('VideoMAE Weight  (CNN Weight = 1 − MAE Weight)', fontsize=12, labelpad=8)
ax.set_ylabel('F1-Score (%)', fontsize=12, labelpad=8)
ax.set_title('Ablation Study — F1-Score Across 11 Weight Combinations', fontsize=13, fontweight='bold', pad=14, color=PURPLE_DARK)
ax.set_xticks(mae_weights)
ax.set_xticklabels([f'{w:.1f}' for w in mae_weights])
ax.set_ylim(84.5, 87.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.fill_between(mae_weights, f1_values, 84.5, alpha=0.08, color=PURPLE_MID)

plt.tight_layout()
plt.savefig('chart2_ablation_study.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 2 saved: chart2_ablation_study.png")


# ─────────────────────────────────────────────
# Chart 3: Generalisation gap bar chart (Slide 7)
# ─────────────────────────────────────────────
datasets = ['FaceForensics++\n(Seen)', 'Celeb-DF\n(Seen)', 'DF40\n(Unseen)', 'DeepFake-Eval\n2024 (Unseen)']
accuracies = [80.67, 86.73, 42.86, 49.51]
bar_colors = [GREEN, GREEN, RED, RED]

fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.bar(datasets, accuracies, color=bar_colors, edgecolor='white', width=0.5)

for bar, val in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
            f'{val}%', ha='center', fontsize=12, fontweight='bold', color='#2c2c2c')

# avg lines
ax.axhline(y=83.70, color=GREEN, linestyle='--', linewidth=1.8, alpha=0.8)
ax.axhline(y=46.19, color=RED, linestyle='--', linewidth=1.8, alpha=0.8)
ax.text(3.55, 84.3, 'Seen avg: 83.70%', color=GREEN, fontsize=10, fontweight='bold')
ax.text(3.55, 46.9, 'Unseen avg: 46.19%', color=RED, fontsize=10, fontweight='bold')

# gap annotation
ax.annotate('', xy=(3.7, 46.19), xytext=(3.7, 83.70),
            arrowprops=dict(arrowstyle='<->', color='#2c2c2c', lw=2))
ax.text(3.75, 65, 'Gap\n37.51 pp', fontsize=11, color='#2c2c2c', fontweight='bold', va='center')

ax.set_ylabel('Ensemble Accuracy (%)', fontsize=12, labelpad=8)
ax.set_title('Cross-Dataset Validation — Generalisation Gap', fontsize=13, fontweight='bold', pad=14, color=PURPLE_DARK)
ax.set_ylim(0, 100)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

seen_patch = mpatches.Patch(color=GREEN, label='Seen during training (Set A)')
unseen_patch = mpatches.Patch(color=RED, label='Never seen — tested without retraining (Set B)')
ax.legend(handles=[seen_patch, unseen_patch], fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig('chart3_generalisation_gap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 3 saved: chart3_generalisation_gap.png")

print("\nAll 3 charts generated successfully.")
