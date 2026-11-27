"""
Compute F1, Precision, Recall, Accuracy, AUC for all OLD models (trained on all 4 datasets)
on the OLD Test_Dataset (all 4 datasets mixed).
This generates the missing metrics for Experiment 2 in the thesis report.
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import json, torch, os, glob, numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor, SwinForImageClassification, AutoImageProcessor
from torchvision.models.video import r2plus1d_18
import cv2
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_FRAMES = 16

# ===================== Build test samples from old folder structure =====================
TEST_DIR = 'data/Test_Dataset'
test_samples = []

# Fake videos (label=1)
fake_dir = os.path.join(TEST_DIR, 'fake')
for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
    for vp in glob.glob(os.path.join(fake_dir, '**', ext), recursive=True):
        test_samples.append((vp, 1))

# Real videos (label=0)
real_dir = os.path.join(TEST_DIR, 'real')
for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
    for vp in glob.glob(os.path.join(real_dir, '**', ext), recursive=True):
        test_samples.append((vp, 0))

print(f'Test set: {len(test_samples)} videos ({sum(1 for _,l in test_samples if l==1)} fake, {sum(1 for _,l in test_samples if l==0)} real)')
print(f'Device: {DEVICE}')

def extract_frames(vp):
    cap = cv2.VideoCapture(str(vp))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    indices = np.linspace(0, total-1, NUM_FRAMES, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frames.append(np.zeros((224,224,3), dtype=np.uint8))
    cap.release()
    return frames

# ===================== Load OLD models (trained on all 4 datasets) =====================
print('\nLoading VideoMAE (old model)...')
mae_proc = VideoMAEImageProcessor.from_pretrained('models/videomae_v8_combined/best')
mae_model = VideoMAEForVideoClassification.from_pretrained('models/videomae_v8_combined/best').to(DEVICE).eval()

print('Loading 3D CNN (old model)...')
cnn_model = r2plus1d_18()
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)
cnn_model.load_state_dict(torch.load('models/3dcnn_r2plus1d/best_model.pth', map_location='cpu', weights_only=False))
cnn_model.eval()

print('Loading Video Swin (old model)...')
swin_proc = AutoImageProcessor.from_pretrained('microsoft/swin-base-simmim-window6-192')
swin_model = SwinForImageClassification.from_pretrained('microsoft/swin-base-simmim-window6-192', num_labels=2, ignore_mismatched_sizes=True)
swin_ckpt = torch.load('models/video_swin_deepfake/best/model.pth', map_location='cpu', weights_only=False)
# Strip 'base_model.' prefix from checkpoint keys
swin_state = {k.replace('base_model.', ''): v for k, v in swin_ckpt['model_state_dict'].items()}
swin_model.load_state_dict(swin_state, strict=False)
swin_model = swin_model.to(DEVICE).eval()

results = {
    'mae': {'y_true': [], 'y_prob': []},
    'cnn': {'y_true': [], 'y_prob': []},
    'swin': {'y_true': [], 'y_prob': []},
    'ens': {'y_true': [], 'y_prob': []}
}

print(f'\nProcessing {len(test_samples)} test videos...')
skipped = 0
for vp, label in tqdm(test_samples, desc="Evaluating (Set B - Old Models)"):
    frames = extract_frames(vp)
    if frames is None:
        skipped += 1
        continue
    resized224 = [cv2.resize(f, (224, 224)) for f in frames]
    resized112 = [cv2.resize(f, (112, 112)) for f in frames]

    # MAE
    try:
        inp = mae_proc(resized224, return_tensors='pt')
        with torch.no_grad():
            out = mae_model(pixel_values=inp['pixel_values'].to(DEVICE))
            p_mae = float(torch.softmax(out.logits, dim=-1).cpu()[0, 1])
    except:
        p_mae = 0.5

    # CNN
    try:
        t = torch.FloatTensor(np.array(resized112)).permute(3, 0, 1, 2) / 255.0
        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
        t = (t - mean) / std
        with torch.no_grad():
            out = cnn_model(t.unsqueeze(0))
            p_cnn = float(torch.softmax(out, dim=-1).cpu()[0, 1])
    except:
        p_cnn = 0.5

    # Swin (frame-level average, uses 192x192)
    try:
        swin_probs = []
        resized192 = [cv2.resize(f, (192, 192)) for f in frames]
        for fr in resized192[::4]:
            inp = swin_proc(images=fr, return_tensors='pt')
            with torch.no_grad():
                out = swin_model(**{k: v.to(DEVICE) for k, v in inp.items()})
                swin_probs.append(float(torch.softmax(out.logits, dim=-1).cpu()[0, 1]))
        p_swin = np.mean(swin_probs)
    except:
        p_swin = 0.5

    # Ensemble (equal weights as determined by validation)
    p_ens = 0.5 * p_mae + 0.5 * p_cnn

    for name, prob in [('mae', p_mae), ('cnn', p_cnn), ('swin', p_swin), ('ens', p_ens)]:
        results[name]['y_true'].append(label)
        results[name]['y_prob'].append(prob)

# ===================== Compute metrics =====================
print(f'\nSkipped {skipped} unreadable videos')
print('\n' + '=' * 80)
header = f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}"
print(header)
print('-' * 80)

model_names = {
    'mae': 'VideoMAE v8',
    'cnn': '3D CNN R(2+1)D',
    'swin': 'Video Swin',
    'ens': 'Ensemble (MAE+CNN)'
}
final = {}

for key in ['mae', 'cnn', 'swin', 'ens']:
    yt = np.array(results[key]['y_true'])
    yp = np.array(results[key]['y_prob'])
    preds = (yp > 0.5).astype(int)
    acc = accuracy_score(yt, preds)
    prec = precision_score(yt, preds, zero_division=0)
    rec = recall_score(yt, preds, zero_division=0)
    f1 = f1_score(yt, preds, zero_division=0)
    try:
        auc = roc_auc_score(yt, yp)
    except:
        auc = 0
    name = model_names[key]
    print(f'{name:<25} {acc*100:>9.2f}% {prec*100:>9.2f}% {rec*100:>9.2f}% {f1*100:>9.2f}% {auc:>9.4f}')
    final[key] = {
        'name': name,
        'accuracy': round(acc * 100, 2),
        'precision': round(prec * 100, 2),
        'recall': round(rec * 100, 2),
        'f1': round(f1 * 100, 2),
        'auc': round(auc, 4)
    }

print('=' * 80)

# Save results
os.makedirs('result', exist_ok=True)
with open('result/setB_all_model_metrics.json', 'w') as f:
    json.dump({
        'experiment': 'Experiment 2 - Old Models (Trained on all 4 datasets)',
        'test_dataset': 'Test_Dataset (all 4 datasets mixed)',
        'test_videos': len(test_samples) - skipped,
        'models': final
    }, f, indent=2)
print('\nSaved: result/setB_all_model_metrics.json')
