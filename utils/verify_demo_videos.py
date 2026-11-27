"""
Find the best videos for live demo — ones with confident correct predictions.
Tests a sample of fake and real videos, picks top 3 each with highest confidence.
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import os, glob, json, torch, numpy as np
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import cv2
from tqdm import tqdm

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_FRAMES = 16
VIDEOMAE_PATH = "models/videomae_v8_combined/best"
CNN_PATH      = "models/3dcnn_r2plus1d/best_model.pth"
W_MAE, W_CNN  = 0.6, 0.4

# Load ensemble config if exists
if os.path.exists("ensemble_config.json"):
    cfg = json.load(open("ensemble_config.json"))
    W_MAE = cfg.get("mae_weight", 0.6)
    W_CNN  = cfg.get("cnn_weight", 0.4)

print(f"Device: {DEVICE}")
print(f"Ensemble weights: VideoMAE={W_MAE}, CNN={W_CNN}")
print("Loading models...")

mae_proc  = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_PATH)
mae_model = VideoMAEForVideoClassification.from_pretrained(VIDEOMAE_PATH).to(DEVICE).eval()

cnn_model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)
cnn_model.load_state_dict(torch.load(CNN_PATH, map_location='cpu', weights_only=False))
cnn_model = cnn_model.to(DEVICE).eval()

print("Models loaded.\n")

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

def infer(frames):
    r224 = [cv2.resize(f, (224,224)) for f in frames]
    r112 = [cv2.resize(f, (112,112)) for f in frames]
    try:
        inp = mae_proc(r224, return_tensors='pt')
        with torch.no_grad():
            out = mae_model(pixel_values=inp['pixel_values'].to(DEVICE))
            p_mae = float(torch.softmax(out.logits, dim=-1).cpu()[0,1])
    except:
        p_mae = 0.5
    try:
        t = torch.FloatTensor(np.array(r112)).permute(3,0,1,2)/255.0
        mean = torch.tensor([0.43216,0.394666,0.37645]).view(3,1,1,1)
        std  = torch.tensor([0.22803,0.22145,0.216989]).view(3,1,1,1)
        t = (t-mean)/std
        with torch.no_grad():
            out = cnn_model(t.unsqueeze(0).to(DEVICE))
            p_cnn = float(torch.softmax(out, dim=-1).cpu()[0,1])
    except:
        p_cnn = 0.5
    score = W_MAE * p_mae + W_CNN * p_cnn
    verdict = "FAKE" if score > 0.5 else "REAL"
    confidence = score if verdict == "FAKE" else 1 - score
    return verdict, round(confidence*100, 1), round(score*100,1)

# --- Collect candidate videos ---
fake_sources = [
    "data/Test_Dataset/fake/FaceForensics_Deepfakes",
    "data/Test_Dataset/fake/Deepfake-Eval-2024",
    "data/Test_Dataset/fake/FaceForensics_Face2Face",
]
real_sources = [
    "data/Test_Dataset/real/FaceForensics_Original",
    "data/Test_Dataset/real/CelebDF",
]

fake_videos, real_videos = [], []
for src in fake_sources:
    vids = glob.glob(os.path.join(src, "*.mp4"))[:15]
    fake_videos.extend([(v, src.split("/")[-1]) for v in vids])

for src in real_sources:
    vids = glob.glob(os.path.join(src, "*.mp4"))[:15]
    real_videos.extend([(v, src.split("/")[-1]) for v in vids])

print(f"Testing {len(fake_videos)} fake candidates, {len(real_videos)} real candidates...\n")

fake_results, real_results = [], []

print("--- FAKE videos ---")
for vp, src in tqdm(fake_videos, desc="Fake"):
    frames = extract_frames(vp)
    if frames is None:
        continue
    verdict, conf, score = infer(frames)
    correct = verdict == "FAKE"
    fake_results.append({"path": vp, "source": src, "verdict": verdict,
                          "confidence": conf, "score": score, "correct": correct})

print("\n--- REAL videos ---")
for vp, src in tqdm(real_videos, desc="Real"):
    frames = extract_frames(vp)
    if frames is None:
        continue
    verdict, conf, score = infer(frames)
    correct = verdict == "REAL"
    real_results.append({"path": vp, "source": src, "verdict": verdict,
                          "confidence": conf, "score": score, "correct": correct})

# Sort by correct first, then highest confidence
fake_results.sort(key=lambda x: (x["correct"], x["confidence"]), reverse=True)
real_results.sort(key=lambda x: (x["correct"], x["confidence"]), reverse=True)

print("\n" + "="*70)
print("TOP FAKE VIDEOS (correct predictions, highest confidence first)")
print("="*70)
for r in fake_results[:5]:
    tag = "CORRECT" if r["correct"] else "WRONG"
    print(f"[{tag}] {r['confidence']}% confident | score={r['score']}% | {os.path.basename(r['path'])} | {r['source']}")

print("\n" + "="*70)
print("TOP REAL VIDEOS (correct predictions, highest confidence first)")
print("="*70)
for r in real_results[:5]:
    tag = "CORRECT" if r["correct"] else "WRONG"
    print(f"[{tag}] {r['confidence']}% confident | score={r['score']}% | {os.path.basename(r['path'])} | {r['source']}")

# Save for reference
out = {"fake_top5": fake_results[:5], "real_top5": real_results[:5]}
with open("result/demo_video_candidates.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved: result/demo_video_candidates.json")
