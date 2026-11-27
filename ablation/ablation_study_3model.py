"""
Ablation Study — 3-Model Ensemble Weight Grid Search
Tests all weight combinations across VideoMAE, R(2+1)D 3D CNN, and
Video Swin Transformer on the Set A Validation set to find the
empirically optimal ensemble configuration.

Strategy:
  Step 1 — Run each model individually to get baseline accuracy.
  Step 2 — Derive performance-proportional weights from val accuracies.
  Step 3 — Grid search all 3-model weight combos (0.1 step, sum = 1.0)
            and compare against single-model and 2-model baselines.

Usage: python ablation_study_3model.py
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import os, json, torch, torch.nn as nn, numpy as np, cv2
from pathlib import Path
from tqdm import tqdm
from transformers import (
    VideoMAEForVideoClassification, VideoMAEImageProcessor,
    AutoImageProcessor, SwinForImageClassification
)
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
VIDEOMAE_PATH = "models_setA/videomae_v8_combined/best"
CNN_PATH      = "models_setA/3dcnn_r2plus1d/best_model.pth"
SWIN_PATH     = "models_setA/video_swin_deepfake/best"
SPLIT_FILE    = "dataset_splits_setA.json"
OUTPUT_DIR    = "result/ablation_study"
NUM_FRAMES    = 16
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


# ===================== Video Swin Wrapper =====================================
class VideoSwinWrapper(nn.Module):
    """Applies a Swin image classifier frame-by-frame, then averages logits."""
    def __init__(self, base_model, num_frames=16):
        super().__init__()
        self.base_model = base_model
        self.num_frames = num_frames

    def forward(self, pixel_values):
        batch_size  = pixel_values.size(0)
        num_frames  = pixel_values.size(1)
        h, w        = pixel_values.size(3), pixel_values.size(4)
        frames      = pixel_values.view(batch_size * num_frames,
                                        pixel_values.size(2), h, w)
        outputs     = self.base_model.swin(frames)
        pooled      = outputs.pooler_output
        hidden_dim  = pooled.size(-1)
        pooled      = pooled.view(batch_size, num_frames, hidden_dim)
        temporal    = pooled.mean(dim=1)
        return self.base_model.classifier(temporal)


# ===================== Load Models ============================================
def load_models():
    print(f"Device: {DEVICE}")
    print("Loading Set A models...")

    print("  [1/3] VideoMAE...")
    mae_proc  = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_PATH)
    mae_model = VideoMAEForVideoClassification.from_pretrained(
        VIDEOMAE_PATH).to(DEVICE).eval()

    print("  [2/3] R(2+1)D 3D CNN...")
    cnn_model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
    cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)
    cnn_model.load_state_dict(torch.load(CNN_PATH, map_location='cpu', weights_only=False))
    cnn_model = cnn_model.to(DEVICE).eval()

    print("  [3/3] Video Swin Transformer...")
    base_swin   = SwinForImageClassification.from_pretrained(
        "microsoft/swin-base-simmim-window6-192",
        num_labels=2, ignore_mismatched_sizes=True)
    swin_proc   = AutoImageProcessor.from_pretrained(SWIN_PATH)
    ckpt        = torch.load(Path(SWIN_PATH) / "model.pth",
                             map_location=DEVICE, weights_only=False)
    swin_model  = VideoSwinWrapper(base_swin, num_frames=NUM_FRAMES)
    swin_model.load_state_dict(ckpt['model_state_dict'])
    swin_model  = swin_model.to(DEVICE).eval()
    print(f"  Video Swin loaded (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.2f}%)")

    return mae_model, mae_proc, cnn_model, swin_model, swin_proc


# ===================== Frame Extraction =======================================
def extract_frames(video_path):
    cap   = cv2.VideoCapture(str(video_path))
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


# ===================== Get Raw Probabilities ==================================
def get_raw_probs(val_samples, mae_model, mae_proc,
                  cnn_model, swin_model, swin_proc):
    y_true, p_mae, p_cnn, p_swin = [], [], [], []

    print(f"\nRunning inference on {len(val_samples)} validation videos...")
    for video_path, label in tqdm(val_samples, desc="Inferencing"):
        frames = extract_frames(video_path)
        if frames is None:
            continue

        # VideoMAE
        try:
            r224    = [cv2.resize(f, (224, 224)) for f in frames]
            inp     = mae_proc(r224, return_tensors="pt")
            with torch.no_grad():
                out = mae_model(pixel_values=inp["pixel_values"].to(DEVICE))
                p_mae.append(float(torch.softmax(out.logits, dim=-1).cpu()[0, 1]))
        except Exception:
            p_mae.append(0.5)

        # 3D CNN
        try:
            r112   = [cv2.resize(f, (112, 112)) for f in frames]
            t      = torch.FloatTensor(np.array(r112)).permute(3, 0, 1, 2) / 255.0
            mean   = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
            std    = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
            t      = (t - mean) / std
            with torch.no_grad():
                out = cnn_model(t.unsqueeze(0).to(DEVICE))
                p_cnn.append(float(torch.softmax(out, dim=-1).cpu()[0, 1]))
        except Exception:
            p_cnn.append(0.5)

        # Video Swin (frame-by-frame)
        try:
            swin_frames = [swin_proc(images=f, return_tensors="pt")["pixel_values"].squeeze(0)
                           for f in frames]
            swin_tensor = torch.stack(swin_frames, dim=0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = swin_model(swin_tensor)
                p_swin.append(float(torch.softmax(out, dim=-1).cpu()[0, 1]))
        except Exception:
            p_swin.append(0.5)

        y_true.append(label)

    return (np.array(y_true), np.array(p_mae),
            np.array(p_cnn), np.array(p_swin))


# ===================== Derive Proportional Weights ============================
def derive_proportional_weights(y_true, p_mae, p_cnn, p_swin):
    acc_mae  = accuracy_score(y_true, (p_mae  > 0.5).astype(int))
    acc_cnn  = accuracy_score(y_true, (p_cnn  > 0.5).astype(int))
    acc_swin = accuracy_score(y_true, (p_swin > 0.5).astype(int))
    total    = acc_mae + acc_cnn + acc_swin

    w_mae  = round(acc_mae  / total, 4)
    w_cnn  = round(acc_cnn  / total, 4)
    w_swin = round(acc_swin / total, 4)

    print("\n" + "=" * 55)
    print("PERFORMANCE-PROPORTIONAL WEIGHT DERIVATION")
    print("=" * 55)
    print(f"  VideoMAE      val accuracy : {acc_mae*100:.2f}%  → weight {w_mae:.4f}")
    print(f"  3D CNN        val accuracy : {acc_cnn*100:.2f}%  → weight {w_cnn:.4f}")
    print(f"  Video Swin    val accuracy : {acc_swin*100:.2f}%  → weight {w_swin:.4f}")
    print("=" * 55)

    return w_mae, w_cnn, w_swin, acc_mae, acc_cnn, acc_swin


# ===================== Build Config List ======================================
def build_configs(w_mae_auto, w_cnn_auto, w_swin_auto):
    configs = []

    # ── Single-model baselines ────────────────────────────────────────────────
    configs += [
        ("VideoMAE Only",       1.0, 0.0, 0.0),
        ("3D CNN Only",         0.0, 1.0, 0.0),
        ("Video Swin Only",     0.0, 0.0, 1.0),
    ]

    # ── 2-model equal-weight baselines ───────────────────────────────────────
    configs += [
        ("MAE(0.5) + CNN(0.5)",        0.5, 0.5, 0.0),
        ("MAE(0.5) + Swin(0.5)",       0.5, 0.0, 0.5),
        ("CNN(0.5) + Swin(0.5)",       0.0, 0.5, 0.5),
    ]

    # ── 3-model grid search (0.1 step, w1+w2+w3 = 1.0) ───────────────────────
    steps = [round(i * 0.1, 1) for i in range(11)]
    for w1 in steps:
        for w2 in steps:
            w3 = round(1.0 - w1 - w2, 1)
            if w3 < 0 or w3 > 1.0:
                continue
            name = f"MAE({w1}) + CNN({w2}) + Swin({w3})"
            configs.append((name, w1, w2, w3))

    # ── Auto-derived proportional weights ─────────────────────────────────────
    auto_name = (f"MAE({w_mae_auto:.2f}) + CNN({w_cnn_auto:.2f}) "
                 f"+ Swin({w_swin_auto:.2f}) ★ [Auto]")
    configs.append((auto_name, w_mae_auto, w_cnn_auto, w_swin_auto))

    return configs


# ===================== Evaluate One Config ====================================
def evaluate_config(name, wm, wc, ws, y_true, p_mae, p_cnn, p_swin):
    prob  = wm * p_mae + wc * p_cnn + ws * p_swin
    y_pred = (prob > 0.5).astype(int)
    return {
        "name":      name,
        "weights":   {"mae": wm, "cnn": wc, "swin": ws},
        "accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "f1_score":  round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
    }


# ===================== Plot ===================================================
def plot_ablation(highlight_results, output_dir):
    """Plot only the key configurations for readability."""
    names = [r["name"] for r in highlight_results]
    x     = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(20, 7))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    b1 = ax.bar(x - 1.5*width, [r["accuracy"]  for r in highlight_results],
                width, label='Accuracy',  color='#4FC3F7', alpha=0.9)
    b2 = ax.bar(x - 0.5*width, [r["precision"] for r in highlight_results],
                width, label='Precision', color='#81C784', alpha=0.9)
    b3 = ax.bar(x + 0.5*width, [r["recall"]    for r in highlight_results],
                width, label='Recall',    color='#FFB74D', alpha=0.9)
    b4 = ax.bar(x + 1.5*width, [r["f1_score"]  for r in highlight_results],
                width, label='F1 Score',  color='#E57373', alpha=0.9)

    for bars in [b1, b2, b3, b4]:
        for i, bar in enumerate(bars):
            if "★" in names[i] or "BEST" in names[i]:
                bar.set_edgecolor('#FFD700')
                bar.set_linewidth(2.5)
            else:
                bar.set_edgecolor('white')
                bar.set_linewidth(0.4)

    ax.set_xlabel('Ensemble Configuration', fontsize=11, color='white', fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=11, color='white', fontweight='bold')
    ax.set_title('Ablation Study: 3-Model Ensemble Weight Comparison\n'
                 '(Set A Validation Set — Single, 2-Model, and 3-Model configurations)',
                 fontsize=12, color='white', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=7, color='white')
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10, facecolor='#16213e', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.15, color='white')

    plt.tight_layout()
    path = os.path.join(output_dir, "ablation_3model.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"Saved chart: {path}")


# ===================== MAIN ===================================================
def main():
    print("=" * 70)
    print("ABLATION STUDY — 3-MODEL ENSEMBLE WEIGHT GRID SEARCH")
    print("Models : VideoMAE | R(2+1)D 3D CNN | Video Swin Transformer")
    print("Data   : Set A Validation Set (15% split)")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(SPLIT_FILE) as f:
        val_samples = json.load(f)["val"]
    print(f"Validation set: {len(val_samples)} videos")

    mae_model, mae_proc, cnn_model, swin_model, swin_proc = load_models()

    y_true, p_mae, p_cnn, p_swin = get_raw_probs(
        val_samples, mae_model, mae_proc, cnn_model, swin_model, swin_proc)
    print(f"Processed {len(y_true)} videos successfully")

    # Derive proportional weights
    w_mae_a, w_cnn_a, w_swin_a, acc_mae, acc_cnn, acc_swin = \
        derive_proportional_weights(y_true, p_mae, p_cnn, p_swin)

    # Build and evaluate all configurations
    configs = build_configs(w_mae_a, w_cnn_a, w_swin_a)

    print(f"\nEvaluating {len(configs)} configurations...")
    results = []
    for name, wm, wc, ws in configs:
        r = evaluate_config(name, wm, wc, ws, y_true, p_mae, p_cnn, p_swin)
        results.append(r)

    # Sort 3-model grid results by F1
    baselines   = [r for r in results if r["weights"]["mae"] in [1.0, 0.0]
                   and r["weights"]["cnn"] in [1.0, 0.0]
                   and r["weights"]["swin"] in [1.0, 0.0]]
    two_model   = [r for r in results if sum(1 for v in r["weights"].values() if v == 0.0) == 1
                   and r not in baselines]
    three_model = [r for r in results if all(v > 0 for v in r["weights"].values())
                   and "★" not in r["name"]]
    auto_result = next(r for r in results if "★" in r["name"])

    best_overall = max(results, key=lambda x: x["f1_score"])
    top5_3model  = sorted(three_model, key=lambda x: x["f1_score"], reverse=True)[:5]

    # ── Print Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SINGLE-MODEL BASELINES")
    print(f"{'Model':<35} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    print("-" * 80)
    for r in baselines:
        print(f"{r['name']:<35} {r['accuracy']:>9.2f}% {r['precision']:>9.2f}% "
              f"{r['recall']:>9.2f}% {r['f1_score']:>7.2f}%")

    print("\n2-MODEL BASELINES (equal weights)")
    print(f"{'Config':<35} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    print("-" * 80)
    for r in two_model:
        print(f"{r['name']:<35} {r['accuracy']:>9.2f}% {r['precision']:>9.2f}% "
              f"{r['recall']:>9.2f}% {r['f1_score']:>7.2f}%")

    print("\nTOP 5 — 3-MODEL CONFIGURATIONS (by F1)")
    print(f"{'Config':<45} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    print("-" * 80)
    for r in top5_3model:
        marker = " ← BEST" if r == best_overall else ""
        print(f"{r['name']:<45} {r['accuracy']:>9.2f}% {r['precision']:>9.2f}% "
              f"{r['recall']:>9.2f}% {r['f1_score']:>7.2f}%{marker}")

    print(f"\nAUTO-DERIVED (proportional weights)")
    print(f"{auto_result['name']:<45} {auto_result['accuracy']:>9.2f}% "
          f"{auto_result['precision']:>9.2f}% {auto_result['recall']:>9.2f}% "
          f"{auto_result['f1_score']:>7.2f}%")

    print("\n" + "=" * 80)
    print(f"Best overall : {best_overall['name']}")
    print(f"             : F1={best_overall['f1_score']}%  "
          f"Acc={best_overall['accuracy']}%  Recall={best_overall['recall']}%")

    # ── Chart — highlight key configs only ───────────────────────────────────
    highlight = baselines + two_model + top5_3model + [auto_result]
    if best_overall not in highlight:
        highlight.append(best_overall)
    plot_ablation(highlight, OUTPUT_DIR)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = {
        "experiment": "Ablation Study — 3-Model Ensemble Weight Grid Search",
        "evaluation_set": "Set A Validation (15%)",
        "models": ["VideoMAE", "R(2+1)D 3D CNN", "Video Swin Transformer"],
        "individual_val_accuracy": {
            "videomae":   round(acc_mae  * 100, 2),
            "3dcnn":      round(acc_cnn  * 100, 2),
            "video_swin": round(acc_swin * 100, 2),
        },
        "auto_derived_weights": {
            "videomae":   w_mae_a,
            "3dcnn":      w_cnn_a,
            "video_swin": w_swin_a,
        },
        "best_config": best_overall,
        "top5_3model": top5_3model,
        "all_results": results,
    }
    path = os.path.join(OUTPUT_DIR, "ablation_3model_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved JSON  : {path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
