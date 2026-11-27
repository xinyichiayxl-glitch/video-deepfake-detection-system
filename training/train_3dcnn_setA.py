
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from pathlib import Path
from tqdm import tqdm
import json
import logging

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    filename='train_3dcnn_setA.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# ===================== Configuration =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "models_setA/3dcnn_r2plus1d"
NUM_FRAMES = 16
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4

# ===================== Dataset Logic =====================
def load_splits():
    with open("dataset_splits_setA.json", "r") as f:
        data = json.load(f)
    return data["train"], data["val"], data["test"]

class DeepfakeDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._extract_frames(video_path)
        frames = torch.FloatTensor(frames).permute(3, 0, 1, 2) 
        frames = frames / 255.0
        mean = torch.tensor([0.43216, 0.394666, 0.37645], dtype=torch.float32).view(3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989], dtype=torch.float32).view(3, 1, 1, 1)
        frames = (frames - mean) / std
        return frames, torch.tensor(label, dtype=torch.long)
    
    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return np.zeros((NUM_FRAMES, 112, 112, 3), dtype=np.uint8)
        indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (112, 112))
                frames.append(frame)
            else:
                frames.append(np.zeros((112, 112, 3), dtype=np.uint8))
        cap.release()
        return np.array(frames)

# ===================== Training =====================
def train_model():
    logging.info("=" * 50)
    logging.info("3D CNN R(2+1)D — SET A TRAINING (FF++ + Celeb-DF)")
    logging.info("=" * 50)
    
    train_samples, val_samples, test_samples = load_splits()
    logging.info(f"Training:   {len(train_samples)} (70%)")
    logging.info(f"Validation: {len(val_samples)} (15%)")
    logging.info(f"Testing:    {len(test_samples)} (15%)")
    
    train_loader = DataLoader(DeepfakeDataset(train_samples), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(DeepfakeDataset(val_samples), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(DeepfakeDataset(test_samples), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Fresh model from Kinetics-400 pretrained weights
    logging.info("Loading R(2+1)D from Kinetics-400 pretrained weights (FRESH)...")
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    Path(MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Train"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        avg_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = val_correct / val_total
        logging.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc*100:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/best_model.pth")
            logging.info(f"New Best! {val_acc*100:.2f}%")

    # Final Test
    logging.info("Loading best model for final test...")
    model.load_state_dict(torch.load(f"{MODEL_SAVE_PATH}/best_model.pth"))
    model.eval()
    test_correct = test_total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Final Test"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    final_acc = test_correct / test_total
    
    # Save results
    results = {
        "model": "3D CNN R(2+1)D",
        "dataset": "Set A (FF++ + Celeb-DF)",
        "best_val_acc": round(best_acc * 100, 2),
        "test_acc": round(final_acc * 100, 2),
        "epochs": EPOCHS,
        "split": "70/15/15"
    }
    with open(f"{MODEL_SAVE_PATH}/setA_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logging.info("=" * 50)
    logging.info(f"FINAL 3D CNN SET A RESULTS")
    logging.info(f"Val Accuracy:  {best_acc*100:.2f}%")
    logging.info(f"Test Accuracy: {final_acc*100:.2f}%")
    logging.info("=" * 50)

if __name__ == "__main__":
    train_model()
