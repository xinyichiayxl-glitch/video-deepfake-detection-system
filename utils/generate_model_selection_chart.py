"""
Generate a clear chart explaining why VideoMAE + 3D CNN was chosen for production.
Shows 3 panels:
  1. All 4 individual models F1 comparison
  2. 2-model vs 3-model best ensemble comparison
  3. GenConViT VAE incompatibility note
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('#1a1a2e')

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle('Why VideoMAE + 3D CNN Was Chosen for the Production Ensemble',
             fontsize=16, color='white', fontweight='bold', y=0.97)

# ── Layout: 1 row, 3 panels ───────────────────────────────────────────────────
ax1 = fig.add_axes([0.04, 0.12, 0.28, 0.72])   # Panel 1
ax2 = fig.add_axes([0.38, 0.12, 0.28, 0.72])   # Panel 2
ax3 = fig.add_axes([0.72, 0.12, 0.24, 0.72])   # Panel 3 (text)

for ax in [ax1, ax2, ax3]:
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — Individual model F1 scores (Set A validation)
# ═══════════════════════════════════════════════════════════════════════════════
models      = ['VideoMAE', '3D CNN\nR(2+1)D', 'Video Swin\nTransformer', 'GenConViT\nVAE']
f1_scores   = [85.47,       86.25,             85.83,                     94.53]
bar_colors  = ['#4FC3F7',   '#81C784',          '#FFB74D',                 '#B0BEC5']
hatch_list  = ['',          '',                 '',                        '//']

bars1 = ax1.bar(models, f1_scores, color=bar_colors, edgecolor='white',
                linewidth=1.2, hatch=hatch_list, alpha=0.85)

for bar, val, h in zip(bars1, f1_scores, hatch_list):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
             f'{val:.2f}%', ha='center', va='bottom',
             fontsize=10, color='white', fontweight='bold')

# Cross-out GenConViT VAE bar
ax1.plot([2.5, 3.5], [88, 96], color='red', linewidth=3, zorder=5)
ax1.plot([2.5, 3.5], [96, 88], color='red', linewidth=3, zorder=5)
ax1.text(3, 97.5, 'Incompatible\noutput format',
         ha='center', fontsize=8, color='#FF6B6B', fontweight='bold')

ax1.set_ylim(80, 102)
ax1.set_ylabel('F1 Score (%)', color='white', fontsize=11)
ax1.set_title('Step 1: Individual Model F1\n(Set A Validation)', color='white',
              fontsize=11, fontweight='bold', pad=10)
ax1.yaxis.label.set_color('white')
ax1.tick_params(axis='x', labelsize=9, colors='white')
ax1.tick_params(axis='y', colors='white')
ax1.grid(axis='y', alpha=0.15, color='white')

# Footnote
ax1.text(1.5, 80.5, '* GenConViT outputs reconstruction error,\n  not a probability score → cannot fuse',
         ha='center', fontsize=7.5, color='#FF6B6B', style='italic')

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — 2-model best vs 3-model best ensemble F1
# ═══════════════════════════════════════════════════════════════════════════════
configs     = ['VideoMAE\nOnly', '3D CNN\nOnly', 'Video Swin\nOnly',
               'Best\n2-Model\nEnsemble\n(MAE 0.4\n+ CNN 0.6)', 'Best\n3-Model\nEnsemble\n(MAE 0.4\n+ CNN 0.5\n+ Swin 0.1)']
f1_vals     = [85.47,            86.25,           85.83,
               87.00,                              86.86]
colors2     = ['#4FC3F7', '#81C784', '#FFB74D', '#FFD700', '#FF8A65']
edge_colors = ['white','white','white','#FFD700','#FF8A65']
lw          = [1, 1, 1, 2.5, 2.5]

bars2 = ax2.bar(configs, f1_vals, color=colors2,
                edgecolor=edge_colors, linewidth=lw, alpha=0.85, width=0.6)

for bar, val in zip(bars2, f1_vals):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
             f'{val:.2f}%', ha='center', va='bottom',
             fontsize=9.5, color='white', fontweight='bold')

# Arrow showing 2-model > 3-model
ax2.annotate('', xy=(4, 87.05), xytext=(3, 87.05),
             arrowprops=dict(arrowstyle='<->', color='#FFD700', lw=2))
ax2.text(3.5, 87.35, 'Adding Swin\ndecreases F1',
         ha='center', fontsize=8, color='#FFD700', fontweight='bold')

# Crown on best bar
ax2.text(3, 87.5, '★ CHOSEN', ha='center', fontsize=9,
         color='#FFD700', fontweight='bold')

ax2.set_ylim(84, 89.5)
ax2.set_ylabel('F1 Score (%)', color='white', fontsize=11)
ax2.set_title('Step 2: Ablation Study\n(2-Model vs 3-Model Ensemble)', color='white',
              fontsize=11, fontweight='bold', pad=10)
ax2.tick_params(axis='x', labelsize=7.5, colors='white')
ax2.tick_params(axis='y', colors='white')
ax2.grid(axis='y', alpha=0.15, color='white')

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — Summary text box
# ═══════════════════════════════════════════════════════════════════════════════
ax3.axis('off')

summary_lines = [
    ('CONCLUSION', '#FFD700', 13, 'bold'),
    ('', 'white', 6, 'normal'),
    ('4 models were evaluated.', 'white', 10, 'normal'),
    ('Only 2 entered production.', 'white', 10, 'normal'),
    ('', 'white', 6, 'normal'),
    ('─' * 28, '#444', 9, 'normal'),
    ('', 'white', 4, 'normal'),
    ('✗  GenConViT VAE', '#FF6B6B', 10, 'bold'),
    ('   Outputs reconstruction error,', '#FF6B6B', 9, 'normal'),
    ('   not a probability score.', '#FF6B6B', 9, 'normal'),
    ('   Cannot be fused mathematically.', '#FF6B6B', 9, 'normal'),
    ('', 'white', 6, 'normal'),
    ('✗  Video Swin Transformer', '#FF6B6B', 10, 'bold'),
    ('   Tested in 3-model ablation.', '#FF6B6B', 9, 'normal'),
    ('   Best 3-model F1: 86.86%', '#FF6B6B', 9, 'normal'),
    ('   Best 2-model F1: 87.00%', '#FF6B6B', 9, 'normal'),
    ('   Adding Swin reduced F1.', '#FF6B6B', 9, 'normal'),
    ('   Empirically excluded.', '#FF6B6B', 9, 'normal'),
    ('', 'white', 6, 'normal'),
    ('─' * 28, '#444', 9, 'normal'),
    ('', 'white', 4, 'normal'),
    ('✓  VideoMAE  (60%)', '#81C784', 10, 'bold'),
    ('   Best temporal model', '#81C784', 9, 'normal'),
    ('   Compatible output format', '#81C784', 9, 'normal'),
    ('', 'white', 4, 'normal'),
    ('✓  3D CNN R(2+1)D  (40%)', '#81C784', 10, 'bold'),
    ('   Best spatial model', '#81C784', 9, 'normal'),
    ('   Compatible output format', '#81C784', 9, 'normal'),
    ('', 'white', 6, 'normal'),
    ('─' * 28, '#444', 9, 'normal'),
    ('', 'white', 4, 'normal'),
    ('Production Ensemble F1: 94.81%', '#FFD700', 10, 'bold'),
]

y_pos = 0.98
for text, color, size, weight in summary_lines:
    ax3.text(0.05, y_pos, text, transform=ax3.transAxes,
             color=color, fontsize=size, fontweight=weight,
             verticalalignment='top', fontfamily='monospace')
    y_pos -= (size + 2) / 130

# ── Step labels at bottom ─────────────────────────────────────────────────────
for ax, label in [(ax1, 'STEP 1: Evaluate all 4 individually'),
                  (ax2, 'STEP 2: Ablation — compare ensemble combinations'),
                  (ax3, 'STEP 3: Final decision')]:
    ax.set_xlabel(label, color='#aaa', fontsize=8.5, labelpad=6)

out = 'result/model_selection_justification.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print(f"Saved: {out}")
