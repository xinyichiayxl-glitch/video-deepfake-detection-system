"""
Cross-Dataset Validation — Per-Dataset Breakdown
Tests Set A models (trained on FF++ + Celeb-DF only) against each
dataset in Set B individually, showing which datasets cause the
largest generalization gap.

Ensemble weights: MAE=0.4, CNN=0.6  (from ablation_results.json best F1)

Usage: python cross_dataset_per_dataset.py
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import os, glob, json, torch, numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import cv2
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_FRAMES    = 16
VIDEOMAE_PATH = "models_setA/videomae_v8_combined/best"
CNN_PATH      = "models_setA/3dcnn_r2plus1d/best_model.pth"
TEST_DIR      = "data/Test_Dataset"

# Ensemble weights from ablation study (MAE=0.4 + CNN=0.6 gave best F1=87.0%)
W_MAE = 0.4
W_CNN = 0.6

# ── Dataset groupings ─────────────────────────────────────────────────────────
# Maps dataset name → which subfolders under fake/ and real/ belong to it.
# DF40 has no dedicated real/ subfolder; FaceForensics_Original is used as
# its real source (standard practice — DF40 fakes were generated from FF++ subjects).
DATASET_GROUPS = {
    "FaceForensics++": {
        "fake_dirs": [
            "FaceForensics_Deepfakes",
            "FaceForensics_Face2Face",
            "FaceForensics_FaceShifter",
            "FaceForensics_FaceSwap",
            "FaceForensics_NeuralTextures",
        ],
        "real_dirs": ["FaceForensics_Original"],
        "in_training": True,   # Set A was trained on this
    },
    "Celeb-DF": {
        "fake_dirs": ["CelebDF"],
        "real_dirs": ["CelebDF"],
        "in_training": True,
    },
    "DF40": {
        "fake_dirs": [
            "DF40_MRAA", "DF40_blendface (1)", "DF40_danet", "DF40_deepfacelab",
            "DF40_facedancer", "DF40_faceswap", "DF40_facevid2vid", "DF40_fomm",
            "DF40_fsgan", "DF40_heygen", "DF40_hyperreenact", "DF40_inswap",
            "DF40_lia", "DF40_mcnet", "DF40_mobileswap", "DF40_one_shot",
            "DF40_pirender", "DF40_sadtalker", "DF40_simswap", "DF40_tpsm",
            "DF40_uniface", "DF40_wav2lip",
        ],
        "real_dirs": ["FaceForensics_Original"],   # shared real source
        "in_training": False,  # NOT seen during Set A training
    },
    "DeepFakeEval": {
        "fake_dirs": ["Deepfake-Eval-2024"],
        "real_dirs": ["Deepfake-Eval-2024"],
        "in_training": False,
    },
}


# ===================== Helpers ===============================================
def collect_videos(base_dir, subfolders, label):
    """Collect all video paths under base_dir/<subfolders>/ with given label."""
    samples = []
    exts = ["mp4", "avi", "mov"]
    for subfolder in subfolders:
        folder = os.path.join(base_dir, subfolder)
        if not os.path.isdir(folder):
            continue
        for ext in exts:
            for vp in glob.glob(os.path.join(folder, f"**/*.{ext}"), recursive=True):
                samples.append((vp, label))
            for vp in glob.glob(os.path.join(folder, f"*.{ext}")):
                if (vp, label) not in samples:
                    samples.append((vp, label))
    return samples


def extract_frames(vp):
    cap   = cv2.VideoCapture(str(vp))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret
                      else np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    return frames


def infer(frames, mae_model, mae_proc, cnn_model):
    r224 = [cv2.resize(f, (224, 224)) for f in frames]
    r112 = [cv2.resize(f, (112, 112)) for f in frames]

    try:
        inp   = mae_proc(r224, return_tensors='pt')
        with torch.no_grad():
            out   = mae_model(pixel_values=inp['pixel_values'].to(DEVICE))
            p_mae = float(torch.softmax(out.logits, dim=-1).cpu()[0, 1])
    except Exception:
        p_mae = 0.5

    try:
        t    = torch.FloatTensor(np.array(r112)).permute(3, 0, 1, 2) / 255.0
        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        std  = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
        t    = (t - mean) / std
        with torch.no_grad():
            out   = cnn_model(t.unsqueeze(0).to(DEVICE))
            p_cnn = float(torch.softmax(out, dim=-1).cpu()[0, 1])
    except Exception:
        p_cnn = 0.5

    return p_mae, p_cnn


def compute_metrics(y_true, y_pred):
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
    }


def evaluate_dataset(name, samples, mae_model, mae_proc, cnn_model):
    """Run inference on all samples and return per-model metrics."""
    y_true, y_mae, y_cnn, y_ens = [], [], [], []
    skipped = 0

    for vp, label in tqdm(samples, desc=f"  {name}", leave=False):
        frames = extract_frames(vp)
        if frames is None:
            skipped += 1
            continue
        p_mae, p_cnn = infer(frames, mae_model, mae_proc, cnn_model)
        p_ens = W_MAE * p_mae + W_CNN * p_cnn

        y_true.append(label)
        y_mae.append(1 if p_mae > 0.5 else 0)
        y_cnn.append(1 if p_cnn > 0.5 else 0)
        y_ens.append(1 if p_ens > 0.5 else 0)

    if len(y_true) == 0:
        return None, skipped

    result = {
        "total_videos": len(y_true),
        "skipped":      skipped,
        "real_count":   int(sum(1 for l in y_true if l == 0)),
        "fake_count":   int(sum(1 for l in y_true if l == 1)),
        "videomae":     compute_metrics(y_true, y_mae),
        "3dcnn":        compute_metrics(y_true, y_cnn),
        "ensemble":     compute_metrics(y_true, y_ens),
    }
    return result, skipped


# ===================== Plot ==================================================
def plot_per_dataset(all_results, avg_seen, avg_unseen, output_dir):
    datasets     = list(all_results.keys())
    in_training  = [all_results[d]["in_training"] for d in datasets]

    acc_ens  = [all_results[d]["ensemble"]["accuracy"]  for d in datasets]
    f1_ens   = [all_results[d]["ensemble"]["f1"]        for d in datasets]
    acc_mae  = [all_results[d]["videomae"]["accuracy"]  for d in datasets]
    acc_cnn  = [all_results[d]["3dcnn"]["accuracy"]     for d in datasets]

    x      = np.arange(len(datasets))
    width  = 0.22
    colors = ['#4FC3F7', '#81C784', '#FFB74D', '#E57373']   # MAE, CNN, Ens-acc, Ens-F1
    seen_colors   = ['#2ecc71', '#2ecc71']   # green for seen datasets
    unseen_colors = ['#e74c3c', '#e74c3c']   # red for unseen datasets

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor('#1a1a2e')

    # ── Left: per-model accuracy per dataset ─────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('#1a1a2e')

    b1 = ax.bar(x - width, acc_mae, width, label='VideoMAE',   color='#4FC3F7', alpha=0.9)
    b2 = ax.bar(x,         acc_cnn, width, label='3D CNN',     color='#81C784', alpha=0.9)
    b3 = ax.bar(x + width, acc_ens, width, label='Ensemble ★', color='#FFB74D', alpha=0.9)

    # Colour bar edges by seen/unseen
    for bars in [b1, b2, b3]:
        for i, bar in enumerate(bars):
            bar.set_edgecolor('#2ecc71' if in_training[i] else '#e74c3c')
            bar.set_linewidth(2.0)

    # Value labels on ensemble bars
    for i, bar in enumerate(b3):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{acc_ens[i]:.1f}%', ha='center', va='bottom',
                fontsize=8, color='white', fontweight='bold')

    ax.axhline(y=avg_seen,   color='#2ecc71', linestyle='--', lw=1.5,
               label=f'Avg seen ({avg_seen:.1f}%)')
    ax.axhline(y=avg_unseen, color='#e74c3c', linestyle='--', lw=1.5,
               label=f'Avg unseen ({avg_unseen:.1f}%)')
    ax.axhline(y=50, color='gray', linestyle=':', lw=1.0, alpha=0.5,
               label='Random baseline (50%)')

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{d}\n({'Seen' if t else 'UNSEEN'})" for d, t in zip(datasets, in_training)],
        color='white', fontsize=9)
    ax.set_ylabel('Accuracy (%)', color='white', fontsize=11)
    ax.set_title('Per-Dataset Accuracy\n(Green border = seen in training, Red = unseen)',
                 color='white', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(facecolor='#16213e', edgecolor='white', labelcolor='white', fontsize=8)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.15, color='white')

    # ── Right: Accuracy vs F1 for ensemble ───────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a2e')

    bar_colors = ['#2ecc71' if t else '#e74c3c' for t in in_training]
    x2 = np.arange(len(datasets))
    w2 = 0.35

    ba = ax2.bar(x2 - w2/2, acc_ens, w2, label='Accuracy',
                 color=[c + 'bb' for c in bar_colors],
                 edgecolor=bar_colors, linewidth=1.5)
    bf = ax2.bar(x2 + w2/2, f1_ens,  w2, label='F1 Score',
                 color=[c + '77' for c in bar_colors],
                 edgecolor=bar_colors, linewidth=1.5)

    for bars in [ba, bf]:
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{bar.get_height():.1f}%', ha='center', va='bottom',
                     fontsize=8, color='white')

    gap = round(avg_seen - avg_unseen, 2)
    ax2.annotate(f'Generalization Gap\n{avg_seen:.1f}% → {avg_unseen:.1f}%\n({gap:.1f}% drop)',
                 xy=(2.5, avg_unseen + 2), fontsize=9, color='#ff6b6b',
                 fontweight='bold', ha='center',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#16213e', edgecolor='#e74c3c'))

    ax2.set_xticks(x2)
    ax2.set_xticklabels(
        [f"{d}\n({'Seen' if t else 'UNSEEN'})" for d, t in zip(datasets, in_training)],
        color='white', fontsize=9)
    ax2.set_ylabel('Score (%)', color='white', fontsize=11)
    ax2.set_title('Ensemble Accuracy vs F1 Score per Dataset\n(Green = seen in training, Red = unseen)',
                  color='white', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 115)
    ax2.legend(facecolor='#16213e', edgecolor='white', labelcolor='white', fontsize=9)
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.15, color='white')

    plt.tight_layout()
    path = os.path.join(output_dir, "cross_dataset_per_dataset.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"Saved chart: {path}")


# ===================== MAIN ==================================================
def main():
    print("=" * 70)
    print("CROSS-DATASET VALIDATION — PER-DATASET BREAKDOWN")
    print(f"Set A models tested on each dataset in Set B individually")
    print(f"Ensemble: VideoMAE×{W_MAE} + 3D CNN×{W_CNN}  (ablation best F1)")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Load Set A models
    print("\nLoading Set A models...")
    mae_proc  = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_PATH)
    mae_model = VideoMAEForVideoClassification.from_pretrained(
        VIDEOMAE_PATH).to(DEVICE).eval()
    cnn_model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
    cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)
    cnn_model.load_state_dict(
        torch.load(CNN_PATH, map_location='cpu', weights_only=False))
    cnn_model = cnn_model.to(DEVICE).eval()
    print("Models loaded.\n")

    all_results = {}

    for dataset_name, cfg in DATASET_GROUPS.items():
        print(f"\n{'─'*70}")
        label = "(in training)" if cfg["in_training"] else "(UNSEEN — cross-dataset)"
        print(f"Dataset: {dataset_name}  {label}")

        fake_samples = collect_videos(
            os.path.join(TEST_DIR, "fake"), cfg["fake_dirs"], label=1)
        real_samples = collect_videos(
            os.path.join(TEST_DIR, "real"), cfg["real_dirs"], label=0)
        samples = fake_samples + real_samples

        print(f"  Real: {len(real_samples)}  |  Fake: {len(fake_samples)}  |  Total: {len(samples)}")

        if len(samples) == 0:
            print("  No videos found — skipping.")
            continue

        result, skipped = evaluate_dataset(
            dataset_name, samples, mae_model, mae_proc, cnn_model)

        if result is None:
            print("  All videos failed to load — skipping.")
            continue

        result["in_training"] = cfg["in_training"]
        all_results[dataset_name] = result

        # Print summary table for this dataset
        print(f"\n  {'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
        print(f"  {'─'*62}")
        for model_key, label_str in [("videomae", "VideoMAE"), ("3dcnn", "3D CNN"), ("ensemble", "Ensemble ★")]:
            m = result[model_key]
            print(f"  {label_str:<20} {m['accuracy']:>9.2f}% {m['precision']:>9.2f}% "
                  f"{m['recall']:>9.2f}% {m['f1']:>7.2f}%")

    # ── Summary across all datasets ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY — ENSEMBLE ACCURACY ACROSS DATASETS")
    print(f"{'='*70}")
    print(f"{'Dataset':<20} {'Seen?':>8} {'Accuracy':>10} {'F1':>8}")
    print(f"{'─'*50}")
    for ds_name, res in all_results.items():
        seen = "Yes" if res["in_training"] else "No"
        acc  = res["ensemble"]["accuracy"]
        f1   = res["ensemble"]["f1"]
        print(f"{ds_name:<20} {seen:>8} {acc:>9.2f}% {f1:>7.2f}%")

    # Generalization gap: avg seen vs avg unseen
    seen_accs   = [r["ensemble"]["accuracy"] for r in all_results.values() if r["in_training"]]
    unseen_accs = [r["ensemble"]["accuracy"] for r in all_results.values() if not r["in_training"]]
    if seen_accs and unseen_accs:
        avg_seen   = round(sum(seen_accs) / len(seen_accs), 2)
        avg_unseen = round(sum(unseen_accs) / len(unseen_accs), 2)
        print(f"\n  Avg accuracy on SEEN datasets   : {avg_seen:.2f}%")
        print(f"  Avg accuracy on UNSEEN datasets : {avg_unseen:.2f}%")
        print(f"  Generalization Gap              : {avg_seen - avg_unseen:.2f}% drop")

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs("result/cross_dataset", exist_ok=True)
    out = {
        "experiment": "Cross-Dataset Validation — Per-Dataset Breakdown",
        "ensemble_weights": {"videomae": W_MAE, "3dcnn": W_CNN},
        "per_dataset": all_results,
    }
    if seen_accs and unseen_accs:
        out["generalization_gap"] = {
            "avg_seen_accuracy":   avg_seen,
            "avg_unseen_accuracy": avg_unseen,
            "gap":                 round(avg_seen - avg_unseen, 2),
        }

    out_path = "result/cross_dataset/per_dataset_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Generate chart ────────────────────────────────────────────────────────
    if seen_accs and unseen_accs:
        print("\nGenerating chart...")
        plot_per_dataset(all_results, avg_seen, avg_unseen, "result/cross_dataset")


def generate_chart_only():
    """Generate chart from existing per_dataset_results.json without re-running inference."""
    json_path = "result/cross_dataset/per_dataset_results.json"
    if not os.path.exists(json_path):
        print(f"Results file not found: {json_path}")
        print("Run the full script first: python cross_dataset_per_dataset.py")
        return
    with open(json_path) as f:
        data = json.load(f)
    all_results = data["per_dataset"]
    gap_data    = data.get("generalization_gap", {})
    avg_seen    = gap_data.get("avg_seen_accuracy", 0)
    avg_unseen  = gap_data.get("avg_unseen_accuracy", 0)
    print("Generating chart from saved results...")
    plot_per_dataset(all_results, avg_seen, avg_unseen, "result/cross_dataset")
    print("Done.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--chart-only":
        generate_chart_only()
    else:
        main()
