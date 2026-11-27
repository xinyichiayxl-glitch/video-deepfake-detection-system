# -*- coding: utf-8 -*-
"""
Video Swin Transformer Training for Deepfake Detection
使用Hierarchical视频transformer，local window attention
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEImageProcessor, AutoModelForVideoClassification
import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm
import time

# Configuration
CONFIG = {
    "model_name": "microsoft/swin-base-simmim-window6-192",  # Swin Transformer预训练模型
    # Note: 真正的Video Swin可能需要从Microsoft官方repo下载
    # 这里使用Swin Transformer + 自定义temporal处理
    "num_labels": 2,  # Real (0) vs Fake (1)
    "num_frames": 16,
    "image_size": 192,  # Swin默认是192x192
    "batch_size": 4,
    "num_epochs": 15,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "warmup_epochs": 2,
    "save_dir": "models_setA/video_swin_deepfake",
    "class_weights": [6.0, 1.0],  # Real:Fake比例调整
}

def load_splits():
    with open("dataset_splits_setA.json", "r") as f:
        data = json.load(f)
    return data["train"], data["val"], data["test"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

class VideoDataset(Dataset):
    def __init__(self, samples, processor, num_frames=16):
        self.processor = processor
        self.num_frames = num_frames
        self.videos = samples
        
        real_count = sum(1 for _, label in self.videos if label == 0)
        fake_count = sum(1 for _, label in self.videos if label == 1)
        print(f"Loaded {len(self.videos)} videos. Real: {real_count}, Fake: {fake_count}")
    
    def __len__(self):
        return len(self.videos)
    
    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return None
        
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        cap.release()
        return frames
    
    def __getitem__(self, idx):
        video_path, label = self.videos[idx]
        
        frames = self.extract_frames(video_path)
        if frames is None:
            frames = [np.zeros((CONFIG["image_size"], CONFIG["image_size"], 3), dtype=np.uint8)] * self.num_frames
        
        # Process with Swin processor (processes each frame)
        # Convert list of frames to batch for processor
        processed_frames = []
        for frame in frames:
            inputs = self.processor(images=frame, return_tensors="pt")
            processed_frames.append(inputs["pixel_values"].squeeze(0))
        
        # Stack frames: (num_frames, C, H, W)
        pixel_values = torch.stack(processed_frames, dim=0)
        
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, dataloader, optimizer, scheduler, class_weights, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        predictions = outputs.logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def validate(model, dataloader, class_weights):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    # Calculate per-class accuracy
    real_correct = sum(1 for p, l in zip(all_preds, all_labels) if l == 0 and p == 0)
    real_total = sum(1 for l in all_labels if l == 0)
    fake_correct = sum(1 for p, l in zip(all_preds, all_labels) if l == 1 and p == 1)
    fake_total = sum(1 for l in all_labels if l == 1)
    
    real_acc = 100 * real_correct / real_total if real_total > 0 else 0
    fake_acc = 100 * fake_correct / fake_total if fake_total > 0 else 0
    
    return avg_loss, accuracy, real_acc, fake_acc

def main():
    print("="*80)
    print("Video Swin Transformer for Deepfake Detection")
    print("="*80)
    
    # Create save directory
    save_dir = Path(CONFIG["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)
    
    # Load processor and model
    print("\nLoading Swin Transformer model...")
    from transformers import AutoImageProcessor, SwinForImageClassification
    import torch.nn as nn
    
    # Load Swin Transformer for image classification
    base_model = SwinForImageClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=CONFIG["num_labels"],
        ignore_mismatched_sizes=True
    )
    processor = AutoImageProcessor.from_pretrained(CONFIG["model_name"])
    
    # Wrap in VideoSwin wrapper for temporal processing
    class VideoSwinWrapper(nn.Module):
        def __init__(self, base_model, num_frames=16):
            super().__init__()
            self.base_model = base_model
            self.num_frames = num_frames
            
        def forward(self, pixel_values, labels=None):
            # pixel_values: (batch, num_frames, C, H, W)
            batch_size = pixel_values.size(0)
            num_frames = pixel_values.size(1)
            
            # Process each frame through Swin
            frames = pixel_values.view(batch_size * num_frames, 
                                     pixel_values.size(2), 
                                     pixel_values.size(3), 
                                     pixel_values.size(4))
            
            outputs = self.base_model.swin(frames)
            pooled = outputs.pooler_output  # (batch*frames, hidden_dim)
            
            # Temporal aggregation: reshape and average
            hidden_dim = pooled.size(-1)
            pooled = pooled.view(batch_size, num_frames, hidden_dim)
            temporal_pooled = pooled.mean(dim=1)  # Average over frames
            
            # Classification
            logits = self.base_model.classifier(temporal_pooled)
            
            # Calculate loss if labels provided
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            
            # Return in HuggingFace format
            from transformers.modeling_outputs import SequenceClassifierOutput
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
            )
    
    model = VideoSwinWrapper(base_model, num_frames=CONFIG["num_frames"])
    model.to(DEVICE)
    
    print(f"Model loaded: {CONFIG['model_name']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Create datasets
    print("\nPreparing datasets from JSON split...")
    train_samples, val_samples, _ = load_splits()
    
    train_dataset = VideoDataset(train_samples, processor, CONFIG["num_frames"])
    val_dataset = VideoDataset(val_samples, processor, CONFIG["num_frames"])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    total_steps = len(train_loader) * CONFIG["num_epochs"]
    warmup_steps = len(train_loader) * CONFIG["warmup_epochs"]
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG["learning_rate"],
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps
    )
    
    # Class weights for imbalanced data
    class_weights = torch.FloatTensor(CONFIG["class_weights"])
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0
    training_history = []
    
    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, class_weights, epoch
        )
        
        # Validate
        val_loss, val_acc, real_acc, fake_acc = validate(model, val_loader, class_weights)
        
        epoch_time = time.time() - start_time
        
        # Log results
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Real Acc: {real_acc:.2f}%, Fake Acc: {fake_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save history
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "real_acc": real_acc,
            "fake_acc": fake_acc,
            "time": epoch_time
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  ⭐ New best validation accuracy: {val_acc:.2f}%")
            
            best_dir = save_dir / "best"
            best_dir.mkdir(exist_ok=True)
            
            # Save model weights
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'val_acc': val_acc,
                'real_acc': real_acc,
                'fake_acc': fake_acc
            }, best_dir / "model.pth")
            
            # Save processor
            processor.save_pretrained(best_dir)
            
            with open(best_dir / "metrics.json", "w") as f:
                json.dump({
                    "epoch": epoch + 1,
                    "val_acc": val_acc,
                    "real_acc": real_acc,
                    "fake_acc": fake_acc
                }, f, indent=2)
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_dir = save_dir / f"checkpoint_epoch_{epoch+1}"
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / "model.pth")
            processor.save_pretrained(checkpoint_dir)
    
    # Save final model
    final_dir = save_dir / "final"
    final_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), final_dir / "model.pth")
    processor.save_pretrained(final_dir)
    
    # Save training history
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: {save_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
