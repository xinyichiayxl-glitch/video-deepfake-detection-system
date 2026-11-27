"""
Evaluate GenConViT VAE (Set A retrained) on the Set A test split.
Uses the same test videos as the other models for fair comparison.

Usage:
    python evaluate_genconvit_setA.py
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GENCONVIT_DIR = os.path.join(PROJECT_ROOT, "GenConViT")
os.chdir(GENCONVIT_DIR)
sys.path.insert(0, GENCONVIT_DIR)
from model.pred_func import load_genconvit, df_face, pred_vid, real_or_fake
os.chdir(PROJECT_ROOT)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_FILE = "dataset_splits_setA.json"
VAE_WEIGHT = "genconvit_vae_Mar_29_2026_07_41_01_acc_0.9186432361602783_inference"
NUM_FRAMES = 15


def main():
    print("=" * 60)
    print("GenConViT VAE - Set A Test Evaluation")
    print(f"Device: {DEVICE}")
    print(f"VAE Weight: {VAE_WEIGHT}")
    print("=" * 60)

    # Load test split
    with open(SPLIT_FILE, "r") as f:
        data = json.load(f)
    samples = data["test"]
    print(f"Test set: {len(samples)} videos\n")

    # Load model (must be in GenConViT dir for config.yaml)
    print("Loading GenConViT VAE model...")
    os.chdir(GENCONVIT_DIR)
    model = load_genconvit("vae", False, "genconvit_ed_inference", VAE_WEIGHT)
    os.chdir(PROJECT_ROOT)
    print("Model loaded.\n")

    y_true = []
    y_pred = []
    confidences = []
    skipped = 0

    for i, (video_path, label) in enumerate(samples):
        if (i + 1) % 50 == 0 or i == 0:
            acc_so_far = accuracy_score(y_true, y_pred) * 100 if len(y_true) > 0 else 0
            print(f"[{i+1}/{len(samples)}] Processing... (running acc: {acc_so_far:.1f}%)")

        try:
            df = df_face(video_path, NUM_FRAMES, "vae")
            if len(df) == 0:
                skipped += 1
                continue

            y, y_val = pred_vid(df, model)
            pred_label = real_or_fake(y)

            # Convert to numeric: FAKE=1, REAL=0
            pred_numeric = 1 if pred_label == "FAKE" else 0

            y_true.append(label)
            y_pred.append(pred_numeric)
            confidences.append(y_val)

        except Exception as e:
            skipped += 1
            if (i + 1) % 100 == 0:
                print(f"  [ERROR at {i+1}]: {str(e)[:80]}")

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, zero_division=0) * 100
    rec = recall_score(y_true, y_pred, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100

    print("\n" + "=" * 60)
    print("GENCONVIT VAE - SET A TEST RESULTS")
    print("=" * 60)
    print(f"Videos tested: {len(y_true)} / {len(samples)} (skipped: {skipped})")
    print(f"Accuracy:  {acc:.2f}%")
    print(f"Precision: {prec:.2f}%")
    print(f"Recall:    {rec:.2f}%")
    print(f"F1 Score:  {f1:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

    # Save results
    os.makedirs("result", exist_ok=True)
    result = {
        "model": "GenConViT VAE",
        "dataset": "Set A (FF++ + Celeb-DF)",
        "weight": VAE_WEIGHT,
        "total_test": len(samples),
        "tested": len(y_true),
        "skipped": skipped,
        "accuracy": round(acc, 2),
        "precision": round(prec, 2),
        "recall": round(rec, 2),
        "f1_score": round(f1, 2),
    }

    with open("result/genconvit_vae_setA_test.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: result/genconvit_vae_setA_test.json")


if __name__ == "__main__":
    main()
