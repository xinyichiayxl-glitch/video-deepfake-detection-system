import os
import sys
import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

RANDOM_SEED = 42
SPLIT_FILE = "dataset_splits_setA.json"

# We want roughly 50/50 real vs fake if possible, or stratify.
# In FF++, we have Original (Real) and 4 Fake types.
# In Celeb-DF, we have Celeb-Real/YouTube-Real (Real) and Celeb-synthesis (Fake).

def collect_setA_samples():
    """Collect ONLY FaceForensics++ and Celeb-DF samples"""
    samples = []
    print("Collecting SET A samples (FF++ and Celeb-DF only)...")
    
    def get_sorted_files(path, pattern="*.mp4", limit=None):
        if not path.exists(): return []
        files = sorted(list(path.glob(pattern))) 
        if limit: return files[:limit]
        return files

    # Celeb-DF
    print("  Adding Celeb-DF...")
    samples.extend([(str(v), 0) for v in get_sorted_files(Path("data/Celeb-DF/Celeb-real"), limit=500)])
    samples.extend([(str(v), 0) for v in get_sorted_files(Path("data/Celeb-DF/YouTube-real"), limit=300)])
    samples.extend([(str(v), 1) for v in get_sorted_files(Path("data/Celeb-DF/Celeb-synthesis"), limit=1500)])
        
    # FaceForensics++
    print("  Adding FaceForensics++...")
    ff_base = Path("data/FaceForensics/FaceForensics++_C23")
    ff_paths = [
        ("original", 0, 1000),              # Real
        ("Deepfakes", 1, 1000),            # Fake
        ("Face2Face", 1, 500),             # Fake
        ("FaceSwap", 1, 500),              # Fake
        ("NeuralTextures", 1, 500),        # Fake
        ("FaceShifter", 1, 500)            # Fake
    ]
    for folder, label, limit in ff_paths:
        p = ff_base / folder
        if p.exists():
            vids = get_sorted_files(p, "**/*.mp4", limit=limit)
            samples.extend([(str(v), label) for v in vids])
            
    return samples

def main():
    all_samples = collect_setA_samples()
    print(f"\nTotal SET A Samples Collected: {len(all_samples)}")
    
    # Stratify by label to keep Real/Fake ratio consistent
    labels = [l for _, l in all_samples]
    
    # The supervisor asked for 70/30 split. 
    # In Deep Learning, the 30% evaluation hold-out is split into Validation and Testing.
    # 70% Train, 15% Val, 15% Test.
    train, temp = train_test_split(all_samples, test_size=0.30, random_state=RANDOM_SEED, stratify=labels)
    
    temp_labels = [l for _, l in temp]
    val, test = train_test_split(temp, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_labels)
    
    split_data = {
        "metadata": {
            "total": len(all_samples),
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "seed": RANDOM_SEED,
            "description": "Set A: Celeb-DF and FaceForensics++ only for Baseline Comparison"
        },
        "train": train,
        "val": val,
        "test": test
    }
    
    with open(SPLIT_FILE, "w") as f:
        json.dump(split_data, f, indent=2)
        
    print("\n" + "="*50)
    print("✅ SET A DATASET (FF++ & CELEB-DF) SAVED")
    print("="*50)
    print(f"File: {SPLIT_FILE}")
    print(f"Total:      {len(all_samples)}")
    print(f"Training:   {len(train)} (70%)")
    print(f"Validation: {len(val)} (15%)")
    print(f"Testing:    {len(test)} (15%)")
    print("="*50)
    print("Ready for RETRAINING VideoMAE, 3D CNN, Video Swin, and GenConViT on Set A.")

if __name__ == "__main__":
    main()
