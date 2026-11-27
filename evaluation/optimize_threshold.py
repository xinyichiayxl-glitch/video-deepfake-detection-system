"""
Optimize Detection Threshold using Validation Set
Option 1: Test multiple thresholds, pick best F1
Option 2: ROC curve analysis, find optimal point

Usage: python optimize_threshold.py
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import json
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_curve, roc_auc_score)
import matplotlib.pyplot as plt

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
VIDEOMAE_PATH = "models/videomae_v8_combined/best"
CNN_PATH      = "models/3dcnn_r2plus1d/best_model.pth"
VAL_DIR       = "data/Validation_Dataset"   # Set B — all 4 datasets
NUM_FRAMES    = 16
WEIGHT_MAE    = 0.6
WEIGHT_CNN    = 0.4


def load_models():
    print("Loading models...")
    mae_processor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_PATH)
    mae_model = VideoMAEForVideoClassification.from_pretrained(VIDEOMAE_PATH).to(DEVICE)
    mae_model.eval()

    cnn_model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
    cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)
    cnn_model.load_state_dict(torch.load(CNN_PATH, map_location="cpu", weights_only=False))
    cnn_model.eval()
    return mae_model, mae_processor, cnn_model


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


def scan_validation_folder(val_dir):
    """Scan data/Validation_Dataset/ recursively — same logic as derive_ensemble_weights_from_setB.py."""
    import glob
    samples = []
    exts = ["mp4", "avi", "mov"]

    for ext in exts:
        for p in glob.glob(f"{val_dir}/real/**/*.{ext}", recursive=True):
            samples.append((p, 0))   # 0 = real
        for p in glob.glob(f"{val_dir}/fake/**/*.{ext}", recursive=True):
            samples.append((p, 1))   # 1 = fake

    real_count = sum(1 for _, l in samples if l == 0)
    fake_count = sum(1 for _, l in samples if l == 1)
    print(f"  Real videos : {real_count}")
    print(f"  Fake videos : {fake_count}")
    print(f"  Total       : {len(samples)}")
    return samples


def get_predictions(samples, mae_model, mae_processor, cnn_model):
    """Get ensemble fake probabilities for all samples."""
    y_true = []
    y_probs = []

    for video_path, label in tqdm(samples, desc="Predicting"):
        frames = extract_frames(video_path)
        if frames is None:
            continue

        resized = [cv2.resize(f, (224, 224)) for f in frames]

        # VideoMAE
        mae_inputs = mae_processor(resized, return_tensors="pt")
        with torch.no_grad():
            mae_out = mae_model(pixel_values=mae_inputs["pixel_values"].to(DEVICE))
            mae_p = torch.softmax(mae_out.logits, dim=-1).cpu().numpy()[0]
            p_mae = float(mae_p[1])

        # 3D CNN
        cnn_frames = [cv2.resize(f, (112, 112)) for f in frames]
        t = torch.FloatTensor(np.array(cnn_frames)).permute(3, 0, 1, 2) / 255.0
        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
        t = (t - mean) / std
        with torch.no_grad():
            cnn_out = cnn_model(t.unsqueeze(0))
            cnn_p = torch.softmax(cnn_out, dim=-1).cpu().numpy()[0]
            p_cnn = float(cnn_p[1])

        p_ens = WEIGHT_MAE * p_mae + WEIGHT_CNN * p_cnn
        y_true.append(label)
        y_probs.append(p_ens)

    return np.array(y_true), np.array(y_probs)


def main():
    print("=" * 60)
    print("THRESHOLD OPTIMIZATION")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Load Set B Validation Dataset (all 4 dataset types)
    # Uses data/Validation_Dataset/ — same pipeline as derive_ensemble_weights_from_setB.py
    print(f"Scanning: {VAL_DIR}")
    val_samples = scan_validation_folder(VAL_DIR)
    print(f"Validation set (Set B): {len(val_samples)} videos")

    mae_model, mae_processor, cnn_model = load_models()
    y_true, y_probs = get_predictions(val_samples, mae_model, mae_processor, cnn_model)
    print(f"Successfully processed {len(y_true)} videos")

    # ========== OPTION 1: Test Multiple Thresholds ==========
    print("\n" + "=" * 60)
    print("OPTION 1: Threshold Sweep")
    print("=" * 60)

    thresholds = np.arange(0.20, 0.80, 0.05)
    best_f1 = 0
    best_thresh_f1 = 0.5
    results = []

    print(f"\n{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 55)

    for t in thresholds:
        preds = (y_probs > t).astype(int)
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_thresh_f1 = t
            marker = " <-- BEST F1"

        print(f"{t:>10.2f} {acc*100:>9.1f}% {prec*100:>9.1f}% {rec*100:>9.1f}% {f1*100:>9.1f}%{marker}")
        results.append({"threshold": float(t), "accuracy": acc, "precision": prec,
                        "recall": rec, "f1": f1})

    print(f"\nBest threshold (by F1): {best_thresh_f1:.2f} (F1={best_f1*100:.1f}%)")

    # ========== OPTION 2: ROC Curve ==========
    print("\n" + "=" * 60)
    print("OPTION 2: ROC Curve Analysis")
    print("=" * 60)

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)

    # Youden's J statistic: optimal = max(TPR - FPR)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh_roc = float(roc_thresholds[best_idx])

    print(f"AUC Score: {auc:.4f}")
    print(f"Optimal threshold (Youden's J): {best_thresh_roc:.4f}")
    print(f"  At this threshold: TPR={tpr[best_idx]*100:.1f}%, FPR={fpr[best_idx]*100:.1f}%")

    # ========== Plot ROC Curve ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')

    # ROC Curve
    ax = axes[0]
    ax.set_facecolor('#1a1a2e')
    ax.plot(fpr, tpr, color='#00d2ff', lw=2, label=f'Ensemble (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    ax.scatter([fpr[best_idx]], [tpr[best_idx]], color='#ff4757', s=100, zorder=5,
               label=f'Optimal threshold = {best_thresh_roc:.3f}')
    ax.set_xlabel('False Positive Rate', color='white')
    ax.set_ylabel('True Positive Rate', color='white')
    ax.set_title('ROC Curve (Validation Set)', color='white', fontweight='bold')
    ax.legend(facecolor='#16213e', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # F1 vs Threshold
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a2e')
    f1_scores = [r["f1"] * 100 for r in results]
    thresh_vals = [r["threshold"] for r in results]
    ax2.plot(thresh_vals, f1_scores, color='#2ed573', lw=2, marker='o', markersize=4)
    ax2.axvline(x=best_thresh_f1, color='#ff4757', linestyle='--', lw=1.5,
                label=f'Best threshold = {best_thresh_f1:.2f}')
    ax2.axvline(x=0.50, color='#ffa502', linestyle=':', lw=1.5, label='Default (0.50)')
    ax2.set_xlabel('Threshold', color='white')
    ax2.set_ylabel('F1 Score (%)', color='white')
    ax2.set_title('F1 Score vs Threshold', color='white', fontweight='bold')
    ax2.legend(facecolor='#16213e', edgecolor='white', labelcolor='white')
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('#444')
    ax2.spines['left'].set_color('#444')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('threshold_optimization.png', dpi=150, facecolor=fig.get_facecolor())
    print(f"\nChart saved: threshold_optimization.png")

    # ========== Final Recommendation ==========
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"  Option 1 (Best F1):    threshold = {best_thresh_f1:.2f}")
    print(f"  Option 2 (ROC/Youden): threshold = {best_thresh_roc:.4f}")
    print(f"  Current (Default):     threshold = 0.50")
    print(f"\nSuggested: Use {best_thresh_f1:.2f} (F1-optimized)")

    # Save config
    config = {
        "optimized_threshold_f1": round(best_thresh_f1, 4),
        "optimized_threshold_roc": round(best_thresh_roc, 4),
        "default_threshold": 0.50,
        "auc_score": round(auc, 4),
        "method": "validation_set_optimization"
    }
    with open("result/threshold_optimization.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Results saved: result/threshold_optimization.json")


if __name__ == "__main__":
    main()
