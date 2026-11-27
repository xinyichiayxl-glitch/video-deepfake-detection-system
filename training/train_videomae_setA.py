import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from pathlib import Path
from tqdm import tqdm
import json
import logging
import time

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    filename='train_videomae_setA.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "models_setA/videomae_v8_combined"
BASE_MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
NUM_FRAMES = 16
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 2e-5

def load_splits():
    with open("dataset_splits_setA.json", "r") as f:
        data = json.load(f)
    return data["train"], data["val"], data["test"]

class VideoMAEDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._extract_frames(video_path)
        
        # Processor handles reshaping and normalization
        inputs = self.processor(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0) # (num_frames, C, H, W)
        
        return pixel_values, torch.tensor(label, dtype=torch.long)
    
    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return [np.zeros((224, 224, 3), dtype=np.uint8)] * NUM_FRAMES
            
        indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        cap.release()
        return frames

def train_model():
    logging.info("Loading SET A Locked Datasets...")
    train_samples, val_samples, test_samples = load_splits()
    
    logging.info(f"Dataset Split:")
    logging.info(f"  Training:   {len(train_samples)}")
    logging.info(f"  Validation: {len(val_samples)}")
    logging.info(f"  Testing:    {len(test_samples)}")
    
    processor = VideoMAEImageProcessor.from_pretrained(BASE_MODEL_ID)
    
    train_ds = VideoMAEDataset(train_samples, processor)
    val_ds = VideoMAEDataset(val_samples, processor)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    logging.info("Loading VideoMAE Base Model...")
    model = VideoMAEForVideoClassification.from_pretrained(
        BASE_MODEL_ID, 
        num_labels=2, 
        ignore_mismatched_sizes=True
    )
    model = model.to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01) 
    criterion = nn.CrossEntropyLoss()
    
    Path(MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs).logits
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = val_correct / val_total
        logging.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Acc={val_acc*100:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_dir = os.path.join(MODEL_SAVE_PATH, "best")
            Path(best_dir).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            logging.info(f"Saved New Best Model! Acc: {val_acc*100:.2f}%")

if __name__ == "__main__":
    train_model()
