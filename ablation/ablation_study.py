"""
Ablation Study — Ensemble Weight Combinations
Tests all weight configurations on the Set A VALIDATION set to prove
that the final chosen weights are empirically optimal.

Strategy:
  Step 1 — Run each model individually on the validation set.
  Step 2 — Derive performance-proportional weights from their val accuracies:
              w_mae = acc_mae / (acc_mae + acc_cnn)
              w_cnn = acc_cnn / (acc_mae + acc_cnn)
  Step 3 — Compare auto-derived weights against a full manual grid search
            so the thesis can show the proportional approach is justified.

Usage:
    python ablation_study.py
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from transformers import (
    VideoMAEForVideoClassification, VideoMAEImageProcessor,
)
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ===================== Paths =====================
VIDEOMAE_PATH  = "models_setA/videomae_v8_combined/best"
CNN_PATH       = "models_setA/3dcnn_r2plus1d/best_model.pth"
SPLIT_FILE     = "dataset_splits_setA.json"
OUTPUT_DIR     = "result/ablation_study"
NUM_FRAMES     = 16
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


# ===================== Load Models =====================
def load_models():
    print(f"Device: {DEVICE}")
    print("Loading models...")

    # VideoMAE
    print("  [1/2] VideoMAE...")
    mae_processor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_PATH)
    mae_model = VideoMAEForVideoClassification.from_pretrained(VIDEOMAE_PATH).to(DEVICE)
    mae_model.eval()

    # 3D CNN
    print("  [2/2] 3D CNN R(2+1)D...")
    weights = R2Plus1D_18_Weights.DEFAULT
    cnn_model = r2plus1d_18(weights=weights)
    cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)
    cnn_model.load_state_dict(torch.load(CNN_PATH, weights_only=False))
    cnn_model = cnn_model.to(DEVICE)
    cnn_model.eval()

    return mae_model, mae_processor, cnn_model


# ===================== Frame Extraction =====================
def extract_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    return frames


# ===================== Get Raw Probabilities =====================
def get_raw_probs(val_samples, mae_model, mae_processor, cnn_model):
    y_true, probs_mae, probs_cnn = [], [], []

    print(f"\nRunning inference on {len(val_samples)} validation videos...")
    for video_path, label in tqdm(val_samples, desc="Inferencing"):
        frames = extract_frames(video_path)
        if frames is None:
            continue

        # VideoMAE — expects 224x224
        mae_inputs = mae_processor(frames, return_tensors="pt")
        with torch.no_grad():
            mae_out = mae_model(pixel_values=mae_inputs["pixel_values"].to(DEVICE))
            mae_p = torch.softmax(mae_out.logits, dim=-1).cpu().numpy()[0]
            probs_mae.append(float(mae_p[1]))

        # 3D CNN — expects 112x112, Kinetics-400 normalisation
        cnn_frames = [cv2.resize(f, (112, 112)) for f in frames]
        cnn_tensor = torch.FloatTensor(np.array(cnn_frames)).permute(3, 0, 1, 2) / 255.0
        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        std  = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
        cnn_tensor = (cnn_tensor - mean) / std
        with torch.no_grad():
            cnn_out = cnn_model(cnn_tensor.unsqueeze(0).to(DEVICE))
            cnn_p = torch.softmax(cnn_out, dim=-1).cpu().numpy()[0]
            probs_cnn.append(float(cnn_p[1]))

        y_true.append(label)

    return np.array(y_true), np.array(probs_mae), np.array(probs_cnn)


# ===================== Derive Proportional Weights =====================
def derive_proportional_weights(y_true, probs_mae, probs_cnn):
    """
    Compute individual model val accuracies, then assign weights
    proportional to each model's performance:
        w_mae = acc_mae / (acc_mae + acc_cnn)
        w_cnn = acc_cnn / (acc_mae + acc_cnn)
    This ensures stronger models contribute more to the ensemble.
    """
    acc_mae = accuracy_score(y_true, (probs_mae > 0.5).astype(int))
    acc_cnn = accuracy_score(y_true, (probs_cnn > 0.5).astype(int))

    total = acc_mae + acc_cnn
    w_mae = round(acc_mae / total, 4)
    w_cnn = round(acc_cnn / total, 4)

    print("\n" + "=" * 55)
    print("PERFORMANCE-PROPORTIONAL WEIGHT DERIVATION")
    print("=" * 55)
    print(f"  VideoMAE  val accuracy : {acc_mae*100:.2f}%")
    print(f"  3D CNN    val accuracy : {acc_cnn*100:.2f}%")
    print(f"  Derived weight — VideoMAE : {w_mae:.4f}")
    print(f"  Derived weight — 3D CNN   : {w_cnn:.4f}")
    print("=" * 55)

    return w_mae, w_cnn, acc_mae, acc_cnn


# ===================== Build Config List =====================
def build_configs(w_mae_auto, w_cnn_auto):
    """
    Build the full grid of configurations to compare.
    The auto-derived proportional weights are inserted and marked with ★.
    """
    # Round to nearest 0.1 for display label
    label = f"MAE({w_mae_auto:.2f}) + CNN({w_cnn_auto:.2f}) ★ [Auto]"

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

        # Auto-derived proportional weights (the final chosen config)
        (label,                        w_mae_auto, w_cnn_auto),
    ]
    return configs


# ===================== Evaluate One Config =====================
def evaluate_config(name, mae_w, cnn_w, y_true, probs_mae, probs_cnn):
    ensemble_prob = mae_w * probs_mae + cnn_w * probs_cnn
    y_pred = (ensemble_prob > 0.5).astype(int)

    acc  = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, zero_division=0) * 100
    rec  = recall_score(y_true, y_pred, zero_division=0) * 100
    f1   = f1_score(y_true, y_pred, zero_division=0) * 100

    return {
        "name": name,
        "weights": {"mae": mae_w, "cnn": cnn_w},
        "accuracy":  round(acc,  2),
        "precision": round(prec, 2),
        "recall":    round(rec,  2),
        "f1_score":  round(f1,   2),
    }


# ===================== Plot Results =====================
def plot_ablation(results, output_dir):
    names     = [r["name"] for r in results]
    accuracy  = [r["accuracy"] for r in results]
    f1        = [r["f1_score"] for r in results]
    recall    = [r["recall"] for r in results]
    precision = [r["precision"] for r in results]

    x     = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    b1 = ax.bar(x - 1.5*width, accuracy,  width, label='Accuracy',  color='#4FC3F7', alpha=0.9)
    b2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='#81C784', alpha=0.9)
    b3 = ax.bar(x + 0.5*width, recall,    width, label='Recall',    color='#FFB74D', alpha=0.9)
    b4 = ax.bar(x + 1.5*width, f1,        width, label='F1 Score',  color='#E57373', alpha=0.9)

    # Highlight chosen config (auto-derived) with gold border
    for bars in [b1, b2, b3, b4]:
        for i, bar in enumerate(bars):
            if "★" in names[i]:
                bar.set_edgecolor('#FFD700')
                bar.set_linewidth(2.5)
            else:
                bar.set_edgecolor('white')
                bar.set_linewidth(0.5)

    ax.set_xlabel('Ensemble Configuration', fontsize=12, color='white', fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, color='white', fontweight='bold')
    ax.set_title('Ablation Study: Ensemble Weight Configuration Comparison\n'
                 '(Evaluated on Set A Validation Set — Weights derived from individual model accuracy)',
                 fontsize=13, color='white', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=7.5, color='white')
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11, loc='upper right', facecolor='#16213e',
              edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.grid(axis='y', alpha=0.15, color='white')

    # Annotation on chosen config
    chosen_idx = next(i for i, n in enumerate(names) if "★" in n)
    ax.annotate('★ Auto-Derived\n  Proportional\n  Weights',
                xy=(chosen_idx, max(accuracy[chosen_idx], f1[chosen_idx]) + 3),
                fontsize=8, color='#FFD700', ha='center', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, "ablation_study.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"\nSaved chart: {path}")
    return path


# ===================== MAIN =====================
def main():
    print("=" * 70)
    print("ABLATION STUDY — PERFORMANCE-PROPORTIONAL ENSEMBLE WEIGHTS")
    print("Evaluated on: Set A Validation Set")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load VALIDATION split (NOT test — tuning only)
    with open(SPLIT_FILE, "r") as f:
        data = json.load(f)
    val_samples = data["val"]
    print(f"Validation set: {len(val_samples)} videos")

    # Load models
    mae_model, mae_processor, cnn_model = load_models()

    # Run inference once — get raw probabilities from both models
    y_true, probs_mae, probs_cnn = get_raw_probs(
        val_samples, mae_model, mae_processor, cnn_model
    )
    print(f"Processed {len(y_true)} videos successfully")

    # Step 1: Derive weights from individual model val accuracies
    w_mae_auto, w_cnn_auto, acc_mae, acc_cnn = derive_proportional_weights(
        y_true, probs_mae, probs_cnn
    )

    # Step 2: Build full config list (grid + auto)
    CONFIGS = build_configs(w_mae_auto, w_cnn_auto)

    # Step 3: Evaluate all configurations
    print("\n" + "=" * 85)
    print(f"{'Configuration':<42} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 85)

    results = []
    for name, mae_w, cnn_w in CONFIGS:
        r = evaluate_config(name, mae_w, cnn_w, y_true, probs_mae, probs_cnn)
        results.append(r)
        marker = " ← CHOSEN" if "★" in name else ""
        print(f"{name:<42} {r['accuracy']:>9.2f}% {r['precision']:>9.2f}% "
              f"{r['recall']:>9.2f}% {r['f1_score']:>9.2f}%{marker}")

    print("=" * 85)

    # Best config by F1
    best = max(results, key=lambda x: x["f1_score"])
    print(f"\nBest by F1   : {best['name']} → F1 = {best['f1_score']}%")
    auto_result = next(r for r in results if "★" in r["name"])
    print(f"Auto-Derived : {auto_result['name']} → F1 = {auto_result['f1_score']}%  "
          f"Acc = {auto_result['accuracy']}%")

    # Generate chart
    print("\nGenerating ablation chart...")
    plot_ablation(results, OUTPUT_DIR)

    # Save JSON
    out = {
        "evaluation_set": "Set A Validation (15%)",
        "description": (
            "Ablation study. Weights derived proportionally from each model's "
            "individual validation accuracy, then compared against a full 0.1-step grid search."
        ),
        "individual_model_val_accuracy": {
            "videomae": round(acc_mae * 100, 2),
            "3dcnn":    round(acc_cnn * 100, 2),
        },
        "auto_derived_weights": {
            "videomae": w_mae_auto,
            "3dcnn":    w_cnn_auto,
        },
        "results": results,
    }
    json_path = os.path.join(OUTPUT_DIR, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved JSON  : {json_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
