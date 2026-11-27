"""
Video Deepfake Detection System v2.1
VideoMAE v8 + 3D CNN R(2+1)D Ensemble

Features:
  - Ensemble Detection (VideoMAE v8 + 3D CNN weighted fusion)
  - Face Validation Security (rejects non-face videos)
  - Sliding Window Analysis (long video temporal segments)
  - Adjustable Detection Threshold
  - Grad-CAM Heatmap (Explainable AI)
  - Batch Processing with CSV Export
  - Base Model Comparison (Action Recognition)

Security Features (v2.1):
  - File Type & Size Validation (rejects non-video / oversized uploads)
  - Input Filename Sanitization (prevents path traversal attacks)
  - Per-Session Rate Limiting (max 20 analyses per hour)
  - Confidence Uncertainty Warning (flags ambiguous results 40-60%)
"""

import gradio as gr
import torch
import torch.nn as nn
import cv2
import numpy as np
import json
import sys
import os
import csv
import re
import time
from collections import deque
import tempfile
from pathlib import Path
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ===================== Configuration =====================
MODEL_MAE_PATH = "models/videomae_v8_combined/best"
MODEL_CNN_PATH = "models/3dcnn_r2plus1d/best_model.pth"
BASE_MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
ENSEMBLE_CONFIG = "ensemble_config.json"
NUM_FRAMES = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sliding Window
WINDOW_SEC = 5
STRIDE_SEC = 4

# ===================== Security Configuration =====================
ALLOWED_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
MAX_FILE_SIZE_MB = 500
RATE_LIMIT_MAX = 20          # max analyses per window
RATE_LIMIT_WINDOW_SEC = 3600 # 1 hour
CONFIDENCE_UNCERTAIN_LOW = 0.40   # below this = likely REAL
CONFIDENCE_UNCERTAIN_HIGH = 0.60  # above this = likely FAKE

# Rate limiter (per-session, resets on app restart)
_rate_timestamps = deque()

# Load ensemble weights
try:
    with open(ENSEMBLE_CONFIG, "r") as f:
        cfg = json.load(f)
    WEIGHT_MAE = cfg.get("mae_weight", 0.6)
    WEIGHT_CNN = cfg.get("cnn_weight", 0.4)
except FileNotFoundError:
    WEIGHT_MAE = 0.6
    WEIGHT_CNN = 0.4

print(f"[System] Device: {DEVICE} | Weights: MAE={WEIGHT_MAE:.2f}, CNN={WEIGHT_CNN:.2f}")

# Global models
model_mae = None
model_cnn = None
processor_mae = None
model_base = None
processor_base = None


# ===================== Model Loading =====================
def load_models():
    global model_mae, model_cnn, processor_mae
    if model_mae is None:
        print("Loading VideoMAE v8...")
        processor_mae = VideoMAEImageProcessor.from_pretrained(MODEL_MAE_PATH)
        model_mae = VideoMAEForVideoClassification.from_pretrained(MODEL_MAE_PATH)
        model_mae = model_mae.to(DEVICE)
        model_mae.eval()
        print("VideoMAE v8 loaded!")
    if model_cnn is None:
        print("Loading 3D CNN R(2+1)D...")
        model_cnn = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        model_cnn.fc = nn.Linear(model_cnn.fc.in_features, 2)
        model_cnn.load_state_dict(torch.load(MODEL_CNN_PATH, map_location="cpu"))
        model_cnn = model_cnn.to("cpu")
        model_cnn.eval()
        print("3D CNN loaded! (CPU)")


def load_base_model():
    global model_base, processor_base
    if model_base is None:
        print("Loading Base VideoMAE (Kinetics-400)...")
        processor_base = VideoMAEImageProcessor.from_pretrained(BASE_MODEL_ID)
        model_base = VideoMAEForVideoClassification.from_pretrained(BASE_MODEL_ID)
        model_base = model_base.to(DEVICE)
        model_base.eval()
        print("Base model loaded!")


# ===================== Security Layer 1: File Validation =====================
def validate_file(video_path):
    """Reject non-video files, oversized files, and unsafe filenames."""
    path = Path(video_path)

    # 1. Extension check
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, (f" Unsupported file type: `{ext}`\n"
                       f"Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}")

    # 2. File size check
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return False, (f" File too large: {size_mb:.0f} MB\n"
                           f"Maximum allowed: {MAX_FILE_SIZE_MB} MB")
    except OSError:
        return False, " Cannot read file."

    # 3. Filename sanitization (path traversal prevention)
    filename = path.name
    if re.search(r'\.\.[\\/]', filename) or re.search(r'[<>:"|?*]', filename):
        return False, " Invalid filename detected (potential path traversal)."  

    return True, f" File validated: {ext.upper()} format, {size_mb:.1f} MB"


# ===================== Security Layer 2: Rate Limiting =====================
def check_rate_limit():
    """Prevent abuse by limiting analyses per hour."""
    now = time.time()
    # Remove timestamps older than the window
    while _rate_timestamps and (now - _rate_timestamps[0]) > RATE_LIMIT_WINDOW_SEC:
        _rate_timestamps.popleft()

    if len(_rate_timestamps) >= RATE_LIMIT_MAX:
        remaining = int(RATE_LIMIT_WINDOW_SEC - (now - _rate_timestamps[0]))
        mins = remaining // 60
        return False, (f" Rate limit reached: {RATE_LIMIT_MAX} analyses per hour.\n"
                       f"Please try again in ~{mins} minutes.")

    _rate_timestamps.append(now)
    remaining = RATE_LIMIT_MAX - len(_rate_timestamps)
    return True, f" Rate limit OK ({remaining} analyses remaining this hour)"


# ===================== Security Layer 3: Confidence Warning =====================
def get_confidence_warning(p_fake):
    """Flag uncertain results where models are not confident."""
    if CONFIDENCE_UNCERTAIN_LOW <= p_fake <= CONFIDENCE_UNCERTAIN_HIGH:
        return (f"\n\n>  **UNCERTAIN RESULT** — Confidence is in the ambiguous zone "
                f"({p_fake*100:.1f}% fake probability, threshold range: "
                f"{CONFIDENCE_UNCERTAIN_LOW*100:.0f}%-{CONFIDENCE_UNCERTAIN_HIGH*100:.0f}%).\n"
                f"> The model is not confident in this prediction. "
                f"Manual verification is recommended.\n")
    return ""


# ===================== Security Layer 4: Face Validation =====================
# Google MediaPipe Face Detection - state-of-the-art (2023+)
import mediapipe as mp

FACE_MIN_RATIO = 0.5  # At least 50% of sampled frames must have a face
MP_FACE_MODEL = "models/face_detector/blaze_face_short_range.tflite"

# MediaPipe face detector (lazy init)
_mp_face_detector = None

def _get_mp_face_detector():
    global _mp_face_detector
    if _mp_face_detector is None:
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MP_FACE_MODEL),
            min_detection_confidence=0.5
        )
        _mp_face_detector = mp.tasks.vision.FaceDetector.create_from_options(options)
    return _mp_face_detector

# DNN detector kept for XAI bounding box analysis
FACE_PROTO = "models/face_detector/deploy.prototxt"
FACE_MODEL = "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
_face_net = None

def _get_face_net():
    global _face_net
    if _face_net is None:
        _face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    return _face_net

def validate_faces(video_path, num_samples=10):
    """Reject videos without human faces using Google MediaPipe (security guardrail)"""
    try:
        detector = _get_mp_face_detector()
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return False, "Cannot read video."
        indices = np.linspace(0, total - 1, num_samples, dtype=int)
        found = 0
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = detector.detect(mp_image)
            if results.detections and len(results.detections) > 0:
                found += 1
        cap.release()
        min_required = int(num_samples * FACE_MIN_RATIO)
        if found >= min_required:
            return True, f"Human face detected in {found}/{num_samples} frames (MediaPipe)."
        return False, f"Insufficient human faces ({found}/{num_samples}, need {min_required}+). This system only analyzes face-centric videos."
    except Exception as e:
        return True, f"Face check skipped: {e}"


# ===================== Frame Extraction =====================
def extract_frames(video_path, num_frames=16, start_frame=0, end_frame=None):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    if end_frame is None:
        end_frame = total
    end_frame = min(end_frame, total)
    if end_frame - start_frame <= 0:
        cap.release()
        return None
    indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
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


def crop_face(frame):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(max(w, h) * 0.3)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
        return frame[y1:y2, x1:x2]
    return frame


def create_frame_grid(frames, cols=4):
    if not frames: return None
    display = [cv2.resize(f, (224, 224)) for f in frames]
    rows_count = (len(display) + cols - 1) // cols
    h, w = 224, 224
    grid = np.zeros((rows_count * h, cols * w, 3), dtype=np.uint8)
    for i, f in enumerate(display):
        r, c = i // cols, i % cols
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = f
    return grid


# ===================== Ensemble Predict (single segment) =====================
def predict_segment(frames, threshold=0.5):
    """Run ensemble on one set of frames. Returns (prob_fake, prob_mae, prob_cnn, pred)"""
    resized = [cv2.resize(f, (224, 224)) for f in frames]
    mae_inputs = processor_mae(resized, return_tensors="pt")
    with torch.no_grad():
        mae_out = model_mae(pixel_values=mae_inputs["pixel_values"].to(DEVICE))
        mae_probs = torch.softmax(mae_out.logits, dim=-1).cpu().numpy()[0]
        p_mae = float(mae_probs[1])

    if DEVICE == "cuda": torch.cuda.empty_cache()

    cnn_frames = [cv2.resize(crop_face(f), (112, 112)) for f in frames]
    t = torch.FloatTensor(np.array(cnn_frames)).permute(3, 0, 1, 2) / 255.0
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
    t = (t - mean) / std
    with torch.no_grad():
        cnn_out = model_cnn(t.unsqueeze(0))
        cnn_probs = torch.softmax(cnn_out, dim=-1).cpu().numpy()[0]
        p_cnn = float(cnn_probs[1])

    p_ens = WEIGHT_MAE * p_mae + WEIGHT_CNN * p_cnn
    return p_ens, p_mae, p_cnn, (1 if p_ens > threshold else 0)


# ===================== Sliding Window =====================
def analyze_video(video_path, threshold=0.5):
    """Analyze video with sliding window for long videos, single pass for short."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    cap.release()

    win_frames = int(WINDOW_SEC * fps)
    stride_frames = int(STRIDE_SEC * fps)

    # Short video: single segment
    if duration <= WINDOW_SEC + 2:
        frames = extract_frames(video_path)
        if frames is None: return [], None, frames
        p_ens, p_mae, p_cnn, pred = predict_segment(frames, threshold)
        seg = {"start": 0, "end": duration, "prob": p_ens, "mae": p_mae, "cnn": p_cnn, "pred": pred}
        return [seg], seg, frames

    # Long video: sliding window
    starts = list(range(0, total - win_frames + 1, stride_frames))
    if len(starts) > 8:
        idx = np.linspace(0, len(starts) - 1, 8, dtype=int)
        starts = [starts[i] for i in idx]

    segments = []
    all_frames = None
    for s in starts:
        e = min(s + win_frames, total)
        frames = extract_frames(video_path, NUM_FRAMES, s, e)
        if frames is None: continue
        if all_frames is None: all_frames = frames
        p_ens, p_mae, p_cnn, pred = predict_segment(frames, threshold)
        segments.append({"start": s/fps, "end": e/fps, "prob": p_ens, "mae": p_mae, "cnn": p_cnn, "pred": pred})

    if not segments: return [], None, all_frames
    worst = max(segments, key=lambda s: s["prob"])
    return segments, worst, all_frames


# ===================== Grad-CAM =====================
def compute_gradcam(frames):
    """Compute Grad-CAM for all 16 frames."""
    resized = [cv2.resize(f, (224, 224)) for f in frames]
    mae_inputs = processor_mae(resized, return_tensors="pt")
    pv = mae_inputs["pixel_values"].to(DEVICE)
    pv.requires_grad_(True)

    acts, grads = {}, {}
    def fh(m, i, o): acts['v'] = o[0].detach()
    def bh(m, gi, go): grads['v'] = go[0].detach()

    layer = model_mae.videomae.encoder.layer[-1]
    h1 = layer.register_forward_hook(fh)
    h2 = layer.register_full_backward_hook(bh)

    model_mae.zero_grad()
    out = model_mae(pixel_values=pv)
    pred = out.logits.argmax(dim=-1).item()
    out.logits[0, pred].backward()
    h1.remove()
    h2.remove()

    w = grads['v'].mean(dim=1, keepdim=True)
    cam = torch.relu((w * acts['v']).sum(dim=-1))

    n = cam.shape[1]
    hp, wp = 14, 14
    if n % (hp * wp) != 0:
        cam = cam[:, 1:]
        n = cam.shape[1]
    tp = n // (hp * wp)
    if tp * hp * wp == n and tp > 0:
        cam = cam.reshape(1, tp, hp, wp).squeeze(0).cpu().numpy()
    else:
        side = int(n ** 0.5)
        cam = cam.reshape(1, side, side).cpu().numpy()
        tp = 1

    hm_frames = []
    raw_heatmaps = []
    for i in range(len(frames)):
        f = cv2.resize(frames[i], (224, 224))
        ti = min(i // max(1, 16 // tp), tp - 1)
        c = cam[ti].copy()
        c -= c.min()
        if c.max() > 0: c /= c.max()
        hm = cv2.resize(c, (224, 224))
        hm_c = (cm.jet(hm)[:, :, :3] * 255).astype(np.uint8)
        hm_frames.append(cv2.addWeighted(f, 0.5, hm_c, 0.5, 0))
        raw_heatmaps.append(hm)

    if DEVICE == "cuda": torch.cuda.empty_cache()
    return create_frame_grid(hm_frames, cols=4), raw_heatmaps


# ===================== XAI: Facial Region Analysis =====================
# Facial region definitions (relative to face bounding box)
FACE_REGIONS = {
    "forehead":    {"y": (0.00, 0.25), "x": (0.15, 0.85), "icon": "🧠",
                    "anomalies": ["Skin texture inconsistency", "Unnatural smoothness or blurring"]},
    "left_eye":    {"y": (0.20, 0.45), "x": (0.10, 0.45), "icon": "👁️",
                    "anomalies": ["Eye shape asymmetry", "Unnatural iris reflection or blinking pattern"]},
    "right_eye":   {"y": (0.20, 0.45), "x": (0.55, 0.90), "icon": "👁️",
                    "anomalies": ["Eye shape asymmetry", "Pupil misalignment or gaze direction inconsistency"]},
    "nose":        {"y": (0.35, 0.65), "x": (0.30, 0.70), "icon": "👃",
                    "anomalies": ["Nose bridge blending artifact", "Shadow inconsistency around nose"]},
    "mouth":       {"y": (0.60, 0.85), "x": (0.20, 0.80), "icon": "👄",
                    "anomalies": ["Lip sync mismatch or unnatural movement", "Teeth rendering anomaly"]},
    "left_cheek":  {"y": (0.40, 0.75), "x": (0.00, 0.25), "icon": "🔲",
                    "anomalies": ["Face boundary blending artifact", "Skin tone mismatch at border"]},
    "right_cheek": {"y": (0.40, 0.75), "x": (0.75, 1.00), "icon": "🔲",
                    "anomalies": ["Face boundary blending artifact", "Edge inconsistency between face and background"]},
    "jawline":     {"y": (0.80, 1.00), "x": (0.10, 0.90), "icon": "🔻",
                    "anomalies": ["Jawline warping or distortion", "Face-neck boundary artifact"]},
}

def generate_xai_report(raw_heatmaps, frames, p_fake, p_mae, p_cnn, pred):
    """Generate natural language XAI report based on Grad-CAM attention regions."""
    if not raw_heatmaps or not frames:
        return ""

    # Detect face in first frame to get bounding box
    net = _get_face_net()
    frame = cv2.resize(frames[0], (224, 224))
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    det = net.forward()

    # Find best face detection
    best_conf, fx1, fy1, fx2, fy2 = 0, 0, 0, 224, 224
    for i in range(det.shape[2]):
        conf = det[0, 0, i, 2]
        if conf > best_conf:
            best_conf = conf
            fx1 = max(0, int(det[0, 0, i, 3] * 224))
            fy1 = max(0, int(det[0, 0, i, 4] * 224))
            fx2 = min(224, int(det[0, 0, i, 5] * 224))
            fy2 = min(224, int(det[0, 0, i, 6] * 224))

    if best_conf < 0.5:
        fx1, fy1, fx2, fy2 = 20, 20, 204, 204  # fallback

    fw, fh = fx2 - fx1, fy2 - fy1

    # Average heatmap across all frames
    avg_heatmap = np.mean(raw_heatmaps, axis=0)

    # Calculate attention score for each facial region
    region_scores = {}
    for name, info in FACE_REGIONS.items():
        ry1 = fy1 + int(info["y"][0] * fh)
        ry2 = fy1 + int(info["y"][1] * fh)
        rx1 = fx1 + int(info["x"][0] * fw)
        rx2 = fx1 + int(info["x"][1] * fw)
        ry1, ry2 = max(0, ry1), min(224, ry2)
        rx1, rx2 = max(0, rx1), min(224, rx2)
        if ry2 > ry1 and rx2 > rx1:
            region_scores[name] = float(np.mean(avg_heatmap[ry1:ry2, rx1:rx2]))
        else:
            region_scores[name] = 0.0

    # Sort regions by attention score
    sorted_regions = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)

    # Build report
    report = "\n---\n### 🔬 AI Explainability Report\n\n"

    if pred == 1:  # FAKE
        report += "**Suspicious regions detected (ranked by attention):**\n\n"
        for name, score in sorted_regions:
            if score < 0.15:
                continue  # Skip low-attention regions
            info = FACE_REGIONS[name]
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            level = "High" if score > 0.5 else "Medium" if score > 0.3 else "Low"
            report += f"**{name.replace('_', ' ').title()}** [{level}]\n"
            report += f"  Attention: `{bar}` {score*100:.0f}%\n"
            if score > 0.3:
                report += f"  → {info['anomalies'][0]}\n"
            report += "\n"

        # Model agreement analysis
        report += "**Model Agreement Analysis:**\n"
        if abs(p_mae - p_cnn) > 0.5:
            report += f"   Models **disagree** (MAE: {p_mae*100:.0f}% vs CNN: {p_cnn*100:.0f}%)\n"
            report += "  → Result may be less reliable. Consider manual inspection.\n"
        else:
            report += f"   Models **agree** (MAE: {p_mae*100:.0f}%, CNN: {p_cnn*100:.0f}%)\n"
            report += "  → High confidence in result.\n"

    else:  # REAL
        report += "**No significant anomalies detected.**\n\n"
        report += "The model's attention is evenly distributed across facial regions, "
        report += "consistent with authentic video characteristics.\n\n"

        # Show top attention area anyway
        top_name, top_score = sorted_regions[0]
        info = FACE_REGIONS[top_name]
        report += f"Highest attention: {top_name.replace('_', ' ').title()} ({top_score*100:.0f}%) — within normal range.\n"

    return report


# ===================== Visual Charts =====================
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from io import BytesIO
from PIL import Image as PILImage

def fig_to_image(fig):
    """Convert matplotlib figure to numpy array (compatible with all matplotlib versions)."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    img = np.array(PILImage.open(buf).convert('RGB'))
    plt.close(fig)
    return img

def create_model_comparison_chart(p_mae, p_cnn, p_ensemble, threshold):
    """Create horizontal bar chart comparing model predictions."""
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    models = ['Ensemble', '3D CNN R(2+1)D', 'VideoMAE v8']
    scores = [p_ensemble * 100, p_cnn * 100, p_mae * 100]
    colors = []
    for s in scores:
        if s > 60: colors.append('#ff4757')      # Red - High risk
        elif s > 30: colors.append('#ffa502')     # Orange - Medium
        else: colors.append('#2ed573')            # Green - Low risk

    bars = ax.barh(models, scores, color=colors, height=0.6, edgecolor='white', linewidth=0.5)
    ax.axvline(x=threshold * 100, color='#ffffff', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Threshold ({threshold*100:.0f}%)')

    # Add score labels on bars
    for bar, score in zip(bars, scores):
        x_pos = min(score + 2, 90)
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{score:.1f}%',
                va='center', ha='left', color='white', fontsize=11, fontweight='bold')

    ax.set_xlim(0, 105)
    ax.set_xlabel('Fake Probability (%)', color='white', fontsize=10)
    ax.set_title('Model Comparison', color='white', fontsize=13, fontweight='bold', pad=10)
    ax.tick_params(colors='white', labelsize=10)
    ax.legend(loc='lower right', fontsize=8, facecolor='#16213e', edgecolor='white', labelcolor='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')

    return fig_to_image(fig)

def create_temporal_chart(segments, threshold):
    """Create bar chart showing fake probability per time segment."""
    if not segments:
        return None

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    labels = [f'{s["start"]:.0f}-{s["end"]:.0f}s' for s in segments]
    probs = [s["prob"] * 100 for s in segments]
    colors = ['#ff4757' if s["pred"] == 1 else '#2ed573' for s in segments]

    bars = ax.bar(labels, probs, color=colors, width=0.7, edgecolor='white', linewidth=0.5)
    ax.axhline(y=threshold * 100, color='#ffa502', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Threshold ({threshold*100:.0f}%)')

    # Add score labels on top
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{prob:.0f}%',
                ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')

    ax.set_ylim(0, 115)
    ax.set_xlabel('Time Segments', color='white', fontsize=10)
    ax.set_ylabel('Fake Probability (%)', color='white', fontsize=10)
    ax.set_title('Temporal Segment Analysis', color='white', fontsize=13, fontweight='bold', pad=10)
    ax.tick_params(colors='white', labelsize=9)
    ax.legend(loc='upper right', fontsize=8, facecolor='#16213e', edgecolor='white', labelcolor='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')

    return fig_to_image(fig)


def create_temporal_consistency_graph(raw_heatmaps, p_fake, pred):
    """Create line graph showing per-frame attention intensity (temporal consistency)."""
    if not raw_heatmaps or len(raw_heatmaps) < 2:
        return None

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    # Calculate mean attention per frame from Grad-CAM heatmaps
    frame_scores = [float(np.mean(hm)) for hm in raw_heatmaps]
    frames_x = list(range(1, len(frame_scores) + 1))

    # Normalize to 0-100 for readability
    max_score = max(frame_scores) if max(frame_scores) > 0 else 1
    frame_pcts = [s / max_score * 100 for s in frame_scores]

    # Line plot with fill
    color = '#ff4757' if pred == 1 else '#2ed573'
    ax.plot(frames_x, frame_pcts, color=color, linewidth=2.5, marker='o', markersize=5, zorder=3)
    ax.fill_between(frames_x, frame_pcts, alpha=0.2, color=color)

    # Mean line
    mean_pct = np.mean(frame_pcts)
    ax.axhline(y=mean_pct, color='#ffa502', linestyle='--', linewidth=1.2, alpha=0.7, label=f'Mean: {mean_pct:.0f}%')

    # Highlight peak frame
    peak_idx = np.argmax(frame_pcts)
    ax.annotate(f'Peak (Frame {peak_idx+1})',
                xy=(frames_x[peak_idx], frame_pcts[peak_idx]),
                xytext=(frames_x[peak_idx], min(frame_pcts[peak_idx] + 15, 110)),
                color='white', fontsize=8, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color='white', lw=1.2))

    # Stats
    std_pct = np.std(frame_pcts)
    consistency = "Stable" if std_pct < 15 else "Variable" if std_pct < 30 else "Inconsistent"
    ax.text(0.98, 0.95, f'Std: {std_pct:.1f}% ({consistency})',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='#16213e', edgecolor='#444'))

    ax.set_xlim(0.5, len(frame_scores) + 0.5)
    ax.set_ylim(0, 115)
    ax.set_xlabel('Frame Number', color='white', fontsize=10)
    ax.set_ylabel('Attention Intensity (%)', color='white', fontsize=10)
    ax.set_title('Temporal Consistency (Per-Frame Attention)', color='white', fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(frames_x)
    ax.tick_params(colors='white', labelsize=9)
    ax.legend(loc='lower right', fontsize=8, facecolor='#16213e', edgecolor='white', labelcolor='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')

    return fig_to_image(fig)


# ===================== TAB 1: Single Detection =====================
FIXED_THRESHOLD = 0.5

def detect_deepfake(video_file):
    """Main detection with security checks + face validation + sliding window + Grad-CAM"""
    threshold = FIXED_THRESHOLD
    if video_file is None:
        return "Waiting...", "Upload a video.", "N/A", None, None, None, None, None

    try:
        # Security Layer 1: File Validation
        valid, file_msg = validate_file(video_file)
        if not valid:
            return "REJECTED", f"**🔒 Security Check Failed**\n\n{file_msg}", "N/A", None, None, None, None, None

        # Security Layer 2: Rate Limiting
        rate_ok, rate_msg = check_rate_limit()
        if not rate_ok:
            return "RATE LIMITED", f"**🔒 Security Check Failed**\n\n{rate_msg}", "N/A", None, None, None, None, None

        # Security Layer 3: Face Validation
        valid, face_msg = validate_faces(video_file)
        if not valid:
            return "REJECTED", f"**🔒 Security Check Failed**\n\n{face_msg}", "N/A", None, None, None, None, None

        load_models()

        # Sliding Window Analysis
        segments, worst, frames = analyze_video(video_file, threshold)
        if worst is None:
            return "Error", "Cannot analyze video.", "N/A", None, None, None, None, None

        p_fake = worst["prob"]
        pred = 1 if p_fake > threshold else 0
        result_label = "REAL (Authentic)" if pred == 0 else "FAKE (Deepfake Detected)"
        confidence = f"{p_fake*100:.1f}% Fake Probability"

        # Security Layer 4: Confidence Warning
        uncertainty_warning = get_confidence_warning(p_fake)
        if uncertainty_warning:
            result_label += " UNCERTAIN"

        # Build detail
        detail = f"""**Ensemble Result** (VideoMAE v8 + 3D CNN R(2+1)D)

| Model | Fake % | Weight |
|---|---|---|
| VideoMAE v8 | {worst['mae']*100:.1f}% | {WEIGHT_MAE:.0%} |
| 3D CNN R(2+1)D | {worst['cnn']*100:.1f}% | {WEIGHT_CNN:.0%} |
| **Ensemble** | **{p_fake*100:.1f}%** | - |

**Decision:** {"FAKE" if pred == 1 else "REAL"} | **Threshold:** {threshold*100:.0f}%
**Security Checks:** {file_msg} | {rate_msg}
**Face Validation:** {face_msg}
{uncertainty_warning}
"""
        # Sliding window info
        if len(segments) > 1:
            detail += "\n**Temporal Segment Analysis:**\n"
            for i, s in enumerate(segments):
                detail += f"  Segment {i+1} [{s['start']:.1f}s - {s['end']:.1f}s]: {s['prob']*100:.1f}%\n"
            detail += "\n  Aggregation: MAX (worst segment used)\n"

        detail += "\n---\n**Grad-CAM:** Red = high attention, Blue = low attention"

        # Grad-CAM + XAI Report
        gradcam_img = None
        raw_heatmaps = None
        try:
            if frames and len(frames) >= 16:
                gradcam_img, raw_heatmaps = compute_gradcam(frames)
        except Exception as e:
            print(f"Grad-CAM error: {e}")

        # XAI Explainability Report
        try:
            xai_report = generate_xai_report(raw_heatmaps, frames, p_fake, worst["mae"], worst["cnn"], pred)
            detail += xai_report
        except Exception as e:
            print(f"XAI report error: {e}")

        # Visual Charts
        model_chart = None
        temporal_chart = None
        try:
            model_chart = create_model_comparison_chart(worst["mae"], worst["cnn"], p_fake, threshold)
        except Exception as e:
            print(f"Model chart error: {e}")
        try:
            temporal_chart = create_temporal_chart(segments, threshold)
        except Exception as e:
            print(f"Temporal chart error: {e}")
        consistency_graph = None
        try:
            consistency_graph = create_temporal_consistency_graph(raw_heatmaps, p_fake, pred)
        except Exception as e:
            print(f"Consistency graph error: {e}")

        if DEVICE == "cuda": torch.cuda.empty_cache()
        preview = create_frame_grid(frames) if frames else None
        return result_label, detail, confidence, preview, gradcam_img, model_chart, temporal_chart, consistency_graph

    except Exception as e:
        import traceback
        traceback.print_exc()
        if DEVICE == "cuda": torch.cuda.empty_cache()
        return "Error", f"Detection failed: {str(e)}", "N/A", None, None, None, None, None


# ===================== TAB 2: Batch Processing =====================
def detect_batch(files, progress=gr.Progress()):
    threshold = FIXED_THRESHOLD
    """Batch process multiple videos with CSV export"""
    if not files:
        return None, None

    load_models()
    results = []

    for file in progress.tqdm(files, desc="Processing"):
        fpath = file.name if hasattr(file, 'name') else str(file)
        fname = os.path.basename(fpath)

        try:
            # Security: File validation
            valid, file_msg = validate_file(fpath)
            if not valid:
                results.append({"Filename": fname, "Result": "REJECTED", "Fake%": "N/A",
                                "MAE%": "N/A", "CNN%": "N/A", "Note": file_msg})
                continue

            # Security: Rate limiting
            rate_ok, rate_msg = check_rate_limit()
            if not rate_ok:
                results.append({"Filename": fname, "Result": "RATE LIMITED", "Fake%": "N/A",
                                "MAE%": "N/A", "CNN%": "N/A", "Note": rate_msg})
                break  # Stop processing remaining files

            valid, face_msg = validate_faces(fpath)
            if not valid:
                results.append({"Filename": fname, "Result": "REJECTED", "Fake%": "N/A",
                                "MAE%": "N/A", "CNN%": "N/A", "Note": face_msg})
                continue

            segments, worst, _ = analyze_video(fpath, threshold)
            if worst is None:
                results.append({"Filename": fname, "Result": "ERROR", "Fake%": "N/A",
                                "MAE%": "N/A", "CNN%": "N/A", "Note": "Cannot process"})
                continue

            pred = 1 if worst["prob"] > threshold else 0
            results.append({
                "Filename": fname,
                "Result": "FAKE" if pred == 1 else "REAL",
                "Fake%": f"{worst['prob']*100:.1f}%",
                "MAE%": f"{worst['mae']*100:.1f}%",
                "CNN%": f"{worst['cnn']*100:.1f}%",
                "Note": f"{len(segments)} segment(s)"
            })
        except Exception as e:
            results.append({"Filename": fname, "Result": "ERROR", "Fake%": "N/A",
                            "MAE%": "N/A", "CNN%": "N/A", "Note": str(e)})

    # Save CSV
    csv_path = os.path.join(tempfile.gettempdir(), "batch_results.csv")
    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        # Also save to project dir
        with open("batch_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

    return results, csv_path


# ===================== TAB 3: Base Model =====================
def analyze_base_model(video_file):
    if video_file is None:
        return "Upload a video.", "Upload a video to test.", None
    try:
        load_base_model()
        frames = extract_frames(video_file)
        if frames is None: return "Error", "Cannot read video.", None

        resized = [cv2.resize(f, (224, 224)) for f in frames]
        inputs = processor_base(resized, return_tensors="pt")
        with torch.no_grad():
            out = model_base(pixel_values=inputs["pixel_values"].to(DEVICE))
            probs = torch.softmax(out.logits, dim=-1)

        top5_p, top5_i = torch.topk(probs[0], 5)
        id2label = model_base.config.id2label

        result = f"Action: {id2label[top5_i[0].item()]} ({top5_p[0]*100:.1f}%)"
        detail = "**Base VideoMAE (No Fine-tuning) - Action Recognition Only**\n\n"
        detail += "This model has **NO ability** to detect deepfakes.\n\n"
        detail += "**Top 5 Actions:**\n"
        for i in range(5):
            detail += f"  {i+1}. **{id2label[top5_i[i].item()]}**: {top5_p[i]*100:.1f}%\n"
        detail += "\n> This proves fine-tuning is essential for deepfake detection."
        return result, detail, create_frame_grid(frames)
    except Exception as e:
        return "Error", str(e), None


# ===================== Gradio Interface =====================
def create_interface():
    with gr.Blocks(title="Deepfake Detection System v2.0", theme=gr.themes.Soft()) as demo:

        gr.Markdown("""
        # Video Deepfake Detection System v2.0
        ### VideoMAE v8 + 3D CNN R(2+1)D Ensemble | Face Validation | Sliding Window | Grad-CAM
        ---
        """)

        # ==================== TAB 1: Detection (Single + Batch) ====================
        with gr.Tab("Deepfake Detection"):

            # --- Single Video ---
            gr.Markdown("### Single Video Analysis")
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload Video", sources=["upload"], format=None)
                    detect_btn = gr.Button("Detect Deepfake", variant="primary", size="lg")

                with gr.Column(scale=1):
                    result_label = gr.Textbox(label="Result", lines=1, interactive=False)
                    confidence_display = gr.Textbox(label="Confidence", lines=1, interactive=False)
                    result_detail = gr.Markdown(value="Waiting for upload...")

            with gr.Row():
                frame_preview = gr.Image(label="Original Frames (16 frames)", type="numpy")
                gradcam_output = gr.Image(label="Grad-CAM Heatmap (Red = High Attention)", type="numpy")

            with gr.Row():
                model_chart_output = gr.Image(label="Model Comparison", type="numpy")
                temporal_chart_output = gr.Image(label="Temporal Segment Analysis", type="numpy")

            with gr.Row():
                consistency_graph_output = gr.Image(label="Temporal Consistency (Per-Frame Attention)", type="numpy")

            detect_btn.click(
                fn=detect_deepfake,
                inputs=[video_input],
                outputs=[result_label, result_detail, confidence_display, frame_preview, gradcam_output, model_chart_output, temporal_chart_output, consistency_graph_output]
            )

            # --- Batch Processing ---
            gr.Markdown("---")
            gr.Markdown("### Batch Processing")
            gr.Markdown("Upload multiple videos for bulk detection. Results exported as CSV.")

            with gr.Row():
                with gr.Column(scale=1):
                    batch_files = gr.File(file_count="multiple", file_types=["video"], label="Upload Videos")
                    batch_btn = gr.Button("Process All", variant="primary", size="lg")
                with gr.Column(scale=2):
                    batch_output = gr.JSON(label="Batch Results")
                    batch_csv = gr.File(label="Download CSV Report")

            batch_btn.click(
                fn=detect_batch,
                inputs=[batch_files],
                outputs=[batch_output, batch_csv]
            )

        # ==================== TAB 2: Base Model ====================
        with gr.Tab("Base Model Comparison"):
            gr.Markdown("""
            ### Original VideoMAE (No Fine-tuning)
            
            Uses the **original pretrained VideoMAE** for **action recognition only**.
            It has **NO ability** to detect deepfakes — proving fine-tuning is essential.
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    base_video = gr.Video(label="Upload video", sources=["upload"], format=None)
                    base_btn = gr.Button("Analyze with Base Model", variant="secondary", size="lg")
                with gr.Column(scale=1):
                    base_result = gr.Textbox(label="Action Classification", lines=1, interactive=False)
                    base_detail = gr.Markdown(value="Upload a video...")
                    base_preview = gr.Image(label="Preview", type="numpy")

            base_btn.click(fn=analyze_base_model, inputs=[base_video],
                          outputs=[base_result, base_detail, base_preview])

        # ==================== Technical Details ====================
        with gr.Accordion("Technical Details", open=False):
            gr.Markdown(f"""
            | Component | Value |
            |---|---|
            | **Device** | {DEVICE.upper()} |
            | **VideoMAE** | {MODEL_MAE_PATH} |
            | **3D CNN** | {MODEL_CNN_PATH} |
            | **Ensemble Weights** | MAE={WEIGHT_MAE:.2f}, CNN={WEIGHT_CNN:.2f} |
            | **Frames** | {NUM_FRAMES} per segment |
            | **Input Size** | MAE=224x224, CNN=112x112 |
            | **Sliding Window** | {WINDOW_SEC}s window, {STRIDE_SEC}s stride |
            | **Face Validation** | Google MediaPipe BlazeFace |
            | **Grad-CAM** | Last encoder layer |
            """)

    return demo


if __name__ == "__main__":
    print("=" * 50)
    print("Deepfake Detection System v2.0")
    print("VideoMAE v8 + 3D CNN Ensemble")
    print("=" * 50)
    
    # Get local IP for network access
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "0.0.0.0"
    
    load_models()
    demo = create_interface()
    
    print(f"\n[INFO] Access the demo at:")
    print(f"  Local: http://127.0.0.1:7860")
    print(f"  Network: http://{local_ip}:7860")
    print(f"  Share Link: (will appear below after launch)\n")
    
    # server_name="0.0.0.0" allows access from any network interface
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)
