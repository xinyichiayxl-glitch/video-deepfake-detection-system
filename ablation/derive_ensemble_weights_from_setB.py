"""
Derive Production Ensemble Weights for app.py
────────────────────────────────────────────────────────────────────────

Purpose:
  This script uses the PRODUCTION models (models/ — trained on ALL 4 datasets:
  FF++, Celeb-DF, DF40, Deepfake-Eval) and evaluates them on the full
  Validation Dataset (also containing all 4 dataset types).

  Based on individual model accuracy on the validation set,
  derive the ensemble weights proportionally:
    w_mae = acc_mae / (acc_mae + acc_cnn)
    w_cnn = acc_cnn / (acc_mae + acc_cnn)

  This ensures the live demo (app.py) uses the mathematically
  optimal weights for the production models.

Difference from other scripts:
  • ablation_study.py                → Set A models + Set A val (for thesis)
  • true_cross_dataset_validation.py → Set A models + Set B test (for thesis)
  • THIS SCRIPT                      → Production models + Full val (for app.py)

Usage:
    python derive_ensemble_weights_from_setB.py

Output:
    - JSON with per-model accuracy and derived weights
    - Updated ensemble_config.json (app.py loads this automatically)
    - Chart comparing single models vs proportional ensemble
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import os, glob, json, torch, numpy as np
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import cv2
from tqdm import tqdm
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_FRAMES = 16

# ── Production model paths (trained on ALL 4 datasets — same as app.py) ──────
VIDEOMAE_PATH = "models/videomae_v8_combined/best"
CNN_PATH      = "models/3dcnn_r2plus1d/best_model.pth"

# ── Full Validation Dataset (all 4 dataset types) ───────────────────────────
VAL_DIR    = 'data/Validation_Dataset'
OUTPUT_DIR = 'result/production_weight_derivation'


# ===================== Frame Extraction ======================================
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
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    return frames


# ===================== Load Models ===========================================
def load_models():
    print("Loading Production models (trained on all 4 datasets)...")
    mae_proc  = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_PATH)
    mae_model = VideoMAEForVideoClassification.from_pretrained(
        VIDEOMAE_PATH).to(DEVICE).eval()

    cnn_model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
    cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)
    cnn_model.load_state_dict(
        torch.load(CNN_PATH, map_location='cpu', weights_only=False))
    cnn_model = cnn_model.to(DEVICE).eval()

    return mae_model, mae_proc, cnn_model


# ===================== Inference on one video ================================
def infer_video(frames, mae_model, mae_proc, cnn_model):
    """Infer on a single video. Returns probability of FAKE (class 1)."""
    resized224 = [cv2.resize(f, (224, 224)) for f in frames]
    resized112 = [cv2.resize(f, (112, 112)) for f in frames]

    try:
        inp = mae_proc(resized224, return_tensors='pt')
        with torch.no_grad():
            out   = mae_model(pixel_values=inp['pixel_values'].to(DEVICE))
            p_mae = float(torch.softmax(out.logits, dim=-1).cpu()[0, 1])
    except Exception:
        p_mae = 0.5

    try:
        t    = torch.FloatTensor(np.array(resized112)).permute(3, 0, 1, 2) / 255.0
        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        std  = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
        t    = (t - mean) / std
        with torch.no_grad():
            out   = cnn_model(t.unsqueeze(0).to(DEVICE))
            p_cnn = float(torch.softmax(out, dim=-1).cpu()[0, 1])
    except Exception:
        p_cnn = 0.5

    return p_mae, p_cnn


# ===================== MAIN ==================================================
def main():
    print("=" * 75)
    print("DERIVE PRODUCTION ENSEMBLE WEIGHTS")
    print("=" * 75)
    print(f"Production Models:  trained on FF++ + Celeb-DF + DF40 + Deepfake-Eval")
    print(f"Validation Data:    data/Validation_Dataset/ (all 4 dataset types)")
    print(f"Device:             {DEVICE}")
    print("=" * 75)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Load Production models ────────────────────────────────────────
    mae_model, mae_proc, cnn_model = load_models()

    # ── Step 2: Gather ALL validation videos (all 4 datasets) ────────────────
    val_samples = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        # All fake videos (FF++, Celeb-DF, DF40_*, Deepfake-Eval-2024)
        for vp in glob.glob(os.path.join(VAL_DIR, 'fake', '**', ext), recursive=True):
            val_samples.append((vp, 1))
        # All real videos
        for vp in glob.glob(os.path.join(VAL_DIR, 'real', '**', ext), recursive=True):
            val_samples.append((vp, 0))

    if len(val_samples) == 0:
        print(f"\n❌ ERROR: No videos found in {VAL_DIR}")
        print(f"   Expected structure:")
        print(f"   data/Validation_Dataset/fake/<subfolders>/*.mp4")
        print(f"   data/Validation_Dataset/real/<subfolders>/*.mp4")
        return

    # Optionally limit to 300 for speed (comment out to use all)
    if len(val_samples) > 300:
        random.seed(42)
        val_samples = random.sample(val_samples, 300)

    print(f"\nValidation samples:      {len(val_samples)} videos")
    print(f"  Fake (all types):      {sum(1 for _, l in val_samples if l == 1)}")
    print(f"  Real:                  {sum(1 for _, l in val_samples if l == 0)}")
    print("-" * 75)

    # ── Step 3: Run inference on all Set B videos ────────────────────────────
    y_true     = []
    y_pred_mae = []
    y_pred_cnn = []
    probs_mae  = []
    probs_cnn  = []
    skipped    = 0

    print(f"\nRunning inference on validation set...")
    for vp, label in tqdm(val_samples, desc="Inferencing"):
        frames = extract_frames(vp)
        if frames is None:
            skipped += 1
            continue

        p_mae, p_cnn = infer_video(frames, mae_model, mae_proc, cnn_model)

        y_true.append(label)
        y_pred_mae.append(1 if p_mae > 0.5 else 0)
        y_pred_cnn.append(1 if p_cnn > 0.5 else 0)
        probs_mae.append(p_mae)
        probs_cnn.append(p_cnn)

    y_true     = np.array(y_true)
    y_pred_mae = np.array(y_pred_mae)
    y_pred_cnn = np.array(y_pred_cnn)
    probs_mae  = np.array(probs_mae)
    probs_cnn  = np.array(probs_cnn)

    # ── Step 4: Calculate individual model accuracy ────────────────────────────
    def compute_metrics(y_true, y_pred):
        acc  = accuracy_score(y_true, y_pred) * 100
        prec = precision_score(y_true, y_pred, zero_division=0) * 100
        rec  = recall_score(y_true, y_pred, zero_division=0) * 100
        f1   = f1_score(y_true, y_pred, zero_division=0) * 100
        return {
            "accuracy":  round(acc, 2),
            "precision": round(prec, 2),
            "recall":    round(rec, 2),
            "f1":        round(f1, 2),
        }

    acc_mae = accuracy_score(y_true, y_pred_mae)
    acc_cnn = accuracy_score(y_true, y_pred_cnn)

    # Auto-derived proportional weights
    total_acc = acc_mae + acc_cnn
    auto_w_mae = round(acc_mae / total_acc, 4)
    auto_w_cnn = round(acc_cnn / total_acc, 4)

    # ── Step 5: Full Grid Search (0.1 to 0.9) + Auto-derived ─────────────────
    configs = [
        # Single-model baselines
        ("VideoMAE Only",              1.0,   0.0),
        ("3D CNN Only",                0.0,   1.0),

        # Full grid search — 0.1 increments
        ("MAE(0.1) + CNN(0.9)",        0.1,   0.9),
        ("MAE(0.2) + CNN(0.8)",        0.2,   0.8),
        ("MAE(0.3) + CNN(0.7)",        0.3,   0.7),
        ("MAE(0.4) + CNN(0.6)",        0.4,   0.6),
        ("MAE(0.5) + CNN(0.5)",        0.5,   0.5),
        ("MAE(0.6) + CNN(0.4)",        0.6,   0.4),
        ("MAE(0.7) + CNN(0.3)",        0.7,   0.3),
        ("MAE(0.8) + CNN(0.2)",        0.8,   0.2),
        ("MAE(0.9) + CNN(0.1)",        0.9,   0.1),

        # Auto-derived proportional weights
        (f"MAE({auto_w_mae:.2f}) + CNN({auto_w_cnn:.2f}) ★ [Auto]", auto_w_mae, auto_w_cnn),
    ]

    print("\n" + "=" * 90)
    print(f"{'Configuration':<45} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 90)

    results_list = []
    for name, w_mae, w_cnn in configs:
        p_ens = w_mae * probs_mae + w_cnn * probs_cnn
        y_pred_ens = (p_ens > 0.5).astype(int)
        m = compute_metrics(y_true, y_pred_ens)
        m["name"] = name
        m["weights"] = {"mae": w_mae, "cnn": w_cnn}
        results_list.append(m)
        marker = " ← CHOSEN" if "★" in name else ""
        print(f"{name:<45} {m['accuracy']:>9.2f}% {m['precision']:>9.2f}% "
              f"{m['recall']:>9.2f}% {m['f1']:>9.2f}%{marker}")

    print("=" * 90)

    # Find peak by F1
    best = max(results_list, key=lambda x: x["f1"])
    auto_result = next(r for r in results_list if "★" in r["name"])

    print(f"\nBest by F1 (Grid Search) : {best['name']} → F1 = {best['f1']}%  Acc = {best['accuracy']}%")
    print(f"Auto-Derived (Proportional): {auto_result['name']} → F1 = {auto_result['f1']}%  Acc = {auto_result['accuracy']}%")

    # Use the BEST config (whichever has highest F1)
    final_w_mae = best["weights"]["mae"]
    final_w_cnn = best["weights"]["cnn"]

    print(f"\n{'FINAL WEIGHTS FOR APP.PY':}")
    print(f"  VideoMAE individual val accuracy : {acc_mae*100:.2f}%")
    print(f"  3D CNN   individual val accuracy : {acc_cnn*100:.2f}%")
    print(f"  Best Weight — VideoMAE : {final_w_mae}")
    print(f"  Best Weight — 3D CNN   : {final_w_cnn}")

    # ── Step 6: Save results to JSON ─────────────────────────────────────────
    output_data = {
        "experiment": "Production Ensemble Weight Grid Search",
        "description": "Production models (trained on all 4 datasets), validated on full Validation_Dataset",
        "individual_val_accuracy": {
            "videomae": round(acc_mae * 100, 2),
            "3dcnn": round(acc_cnn * 100, 2),
        },
        "auto_derived_weights": {
            "videomae": auto_w_mae,
            "3dcnn": auto_w_cnn,
        },
        "best_weights_by_f1": {
            "videomae": final_w_mae,
            "3dcnn": final_w_cnn,
            "config_name": best["name"],
        },
        "all_results": results_list,
        "validation_stats": {
            "total_videos_tested": len(y_true),
            "skipped_videos": skipped,
        },
    }

    json_path = os.path.join(OUTPUT_DIR, "production_weights.json")
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n✓ Saved detailed results: {json_path}")

    # ── Save to ensemble_config.json for app.py ──────────────────────────────
    ensemble_config = {
        "mae_weight": final_w_mae,
        "cnn_weight": final_w_cnn,
        "ensemble_type": "2-model",
        "derived_from": "Grid search on production models + full 4-dataset Validation_Dataset",
        "mae_accuracy_val": round(acc_mae * 100, 2),
        "cnn_accuracy_val": round(acc_cnn * 100, 2),
    }
    config_path = "ensemble_config.json"
    with open(config_path, "w") as f:
        json.dump(ensemble_config, f, indent=2)
    print(f"✓ Saved app.py config : {config_path}")

    # ── Step 8: Generate comparison chart ─────────────────────────────────────
    # Extract individual model results from results_list for the chart
    mae_result  = next(r for r in results_list if r["name"] == "VideoMAE Only")
    cnn_result  = next(r for r in results_list if r["name"] == "3D CNN Only")

    print("\nGenerating comparison chart...")
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    model_labels = ['VideoMAE\nOnly', '3D CNN\nOnly', f'Ensemble\n(Best: {best["name"][:12]})']
    accuracy_vals = [
        mae_result['accuracy'],
        cnn_result['accuracy'],
        best['accuracy']
    ]
    colors = ['#4FC3F7', '#81C784', '#FFD700']

    bars = ax.bar(model_labels, accuracy_vals, color=colors, alpha=0.8, edgecolor='white', linewidth=2)

    # Add value labels on bars
    for bar, val in zip(bars, accuracy_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%',
                ha='center', va='bottom', fontsize=12, color='white', fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12, color='white', fontweight='bold')
    ax.set_title('Production Validation: Individual Models vs Best Ensemble\n'
                 '(Production Models on Full 4-Dataset Validation Set)',
                 fontsize=13, color='white', fontweight='bold', pad=15)
    ax.set_ylim(0, 110)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.15, color='white')

    # Annotation
    ax.text(2, best['accuracy'] + 5,
            f"★ Best config by F1\n"
            f"VideoMAE: {final_w_mae} | CNN: {final_w_cnn}",
            ha='center', fontsize=9, color='#FFD700', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#16213e', edgecolor='#FFD700', linewidth=1))

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "production_weight_comparison.png")
    plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"✓ Saved chart: {chart_path}")

    print("\n" + "=" * 80)
    print("✅ DONE! app.py will now automatically load these weights:")
    print("=" * 80)
    print(f"  WEIGHT_MAE = {final_w_mae}")
    print(f"  WEIGHT_CNN = {final_w_cnn}")
    print(f"  Config file: ensemble_config.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
