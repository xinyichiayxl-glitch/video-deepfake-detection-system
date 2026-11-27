"""
Compute F1, Precision, Recall, Accuracy, AUC for all models on test set.
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import json, torch, numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor, SwinForImageClassification, AutoImageProcessor
from torchvision.models.video import r2plus1d_18
import cv2
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_FRAMES = 16

with open('dataset_splits.json') as f:
    splits = json.load(f)
test_samples = splits['test']
print(f'Test set: {len(test_samples)} videos')
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

# Load models
print('\nLoading VideoMAE...')
mae_proc = VideoMAEImageProcessor.from_pretrained('models/videomae_v8_combined/best')
mae_model = VideoMAEForVideoClassification.from_pretrained('models/videomae_v8_combined/best').to(DEVICE).eval()

print('Loading 3D CNN...')
cnn_model = r2plus1d_18()
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)
cnn_model.load_state_dict(torch.load('models/3dcnn_r2plus1d/best_model.pth', map_location='cpu', weights_only=False))
cnn_model.eval()

print('Loading Video Swin...')
swin_proc = AutoImageProcessor.from_pretrained('models/video_swin_deepfake/best')
swin_model = SwinForImageClassification.from_pretrained('models/video_swin_deepfake/best').to(DEVICE).eval()

results = {
    'mae': {'y_true': [], 'y_prob': []},
    'cnn': {'y_true': [], 'y_prob': []},
    'swin': {'y_true': [], 'y_prob': []},
    'ens': {'y_true': [], 'y_prob': []}
}

print(f'\nProcessing {len(test_samples)} test videos...')
for vp, label in tqdm(test_samples, desc="Evaluating"):
    frames = extract_frames(vp)
    if frames is None:
        continue
    resized224 = [cv2.resize(f, (224, 224)) for f in frames]
    resized112 = [cv2.resize(f, (112, 112)) for f in frames]

    # MAE
    inp = mae_proc(resized224, return_tensors='pt')
    with torch.no_grad():
        out = mae_model(pixel_values=inp['pixel_values'].to(DEVICE))
        p_mae = float(torch.softmax(out.logits, dim=-1).cpu()[0, 1])

    # CNN
    t = torch.FloatTensor(np.array(resized112)).permute(3, 0, 1, 2) / 255.0
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
    t = (t - mean) / std
    with torch.no_grad():
        out = cnn_model(t.unsqueeze(0))
        p_cnn = float(torch.softmax(out, dim=-1).cpu()[0, 1])

    # Swin (frame-level average)
    swin_probs = []
    for fr in resized224[::4]:
        inp = swin_proc(images=fr, return_tensors='pt')
        with torch.no_grad():
            out = swin_model(**{k: v.to(DEVICE) for k, v in inp.items()})
            swin_probs.append(float(torch.softmax(out.logits, dim=-1).cpu()[0, 1]))
    p_swin = np.mean(swin_probs)

    # Ensemble
    p_ens = 0.6 * p_mae + 0.4 * p_cnn

    for name, prob in [('mae', p_mae), ('cnn', p_cnn), ('swin', p_swin), ('ens', p_ens)]:
        results[name]['y_true'].append(label)
        results[name]['y_prob'].append(prob)

# Compute metrics
print('\n' + '=' * 75)
header = f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}"
print(header)
print('-' * 75)

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
    print(f'{name:<22} {acc*100:>9.2f}% {prec*100:>9.2f}% {rec*100:>9.2f}% {f1*100:>9.2f}% {auc:>9.4f}')
    final[key] = {
        'name': name,
        'accuracy': round(acc * 100, 2),
        'precision': round(prec * 100, 2),
        'recall': round(rec * 100, 2),
        'f1': round(f1 * 100, 2),
        'auc': round(auc, 4)
    }

print('=' * 75)

with open('result/all_model_metrics.json', 'w') as f:
    json.dump(final, f, indent=2)
print('\nSaved: result/all_model_metrics.json')
