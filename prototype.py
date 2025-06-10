#!/usr/bin/env python3
"""
Minimal Working YOLO - 確実に動く最小実装
まずはこれで動作確認してから機能追加していく
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import time

# ===== 設定（超シンプル）=====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4  # 小さく始める
IMG_SIZE = 416  # 正方形でシンプルに
NUM_CLASSES = 15
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10

# パス設定
TRAIN_IMG_DIR = "/content/FLIR_YOLO_local/images/train"
TRAIN_LABEL_DIR = "/content/FLIR_YOLO_local/labels/train"
SAVE_DIR = "/content/drive/MyDrive/IR_obj_detection/minimal_yolo"

# ===== データセット（最小限）=====
class MinimalDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=416):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')][:1000]  # 最初は1000枚だけ
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 画像
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = cv2.imread(img_path, 0)  # グレースケール
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).float() / 255.0
        img = img.unsqueeze(0)  # [1, H, W]
        
        # ラベル（YOLO形式）
        label_path = os.path.join(self.label_dir, self.img_files[idx].replace('.jpg', '.txt'))
        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, cx, cy, w, h = map(float, parts)
                        targets.append([cls, cx, cy, w, h])
        
        return img, torch.tensor(targets) if targets else torch.zeros((0, 5))

# ===== モデル（超シンプル）=====
class MinimalYOLO(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.num_classes = num_classes
        
        # 超シンプルなBackbone（ダウンサンプリング5回でstride=32）
        self.features = nn.Sequential(
            # 416 -> 208
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 208 -> 104
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 104 -> 52
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 52 -> 26
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 26 -> 13
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Detection head（1つのスケールのみ、アンカーなし）
        self.detector = nn.Conv2d(256, 5 + num_classes, 1)
        
    def forward(self, x):
        x = self.features(x)
        x = self.detector(x)
        
        # [B, C, H, W] -> [B, H*W, 5+classes]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, H * W, C)
        
        return x, (H, W)  # グリッドサイズも返す

# ===== 損失関数（超シンプル）=====
class MinimalLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, predictions, targets, grid_size):
        """
        predictions: [B, H*W, 5+classes]
        targets: リスト of [N, 5] (class, cx, cy, w, h)
        """
        B = predictions.size(0)
        H, W = grid_size
        device = predictions.device
        
        # 予測値を分解
        pred_xy = predictions[..., :2].sigmoid()  # 0-1に正規化
        pred_wh = predictions[..., 2:4].exp()     # exp変換
        pred_conf = predictions[..., 4].sigmoid()  # objectness
        pred_cls = predictions[..., 5:].sigmoid()  # classes
        
        # 損失の初期化
        loss_xy = 0
        loss_wh = 0
        loss_conf = 0
        loss_cls = 0
        num_targets = 0
        
        # ターゲットマスク（どのグリッドにオブジェクトがあるか）
        obj_mask = torch.zeros(B, H * W, device=device)
        
        # バッチごとに処理
        for b in range(B):
            if len(targets[b]) == 0:
                continue
                
            for t in targets[b]:
                if torch.isnan(t).any() or torch.isinf(t).any():
                    continue
                    
                cls_id, cx, cy, w, h = t
                
                # グリッド座標
                gx = int(cx * W)
                gy = int(cy * H)
                
                if gx >= W or gy >= H or gx < 0 or gy < 0:
                    continue
                
                # グリッドインデックス
                gi = gy * W + gx
                
                # オブジェクトマスク
                obj_mask[b, gi] = 1
                
                # 相対座標（グリッド内の位置）
                tx = cx * W - gx
                ty = cy * H - gy
                tw = w * W
                th = h * H
                
                # 座標損失
                loss_xy += F.mse_loss(pred_xy[b, gi], torch.tensor([tx, ty], device=device))
                loss_wh += F.mse_loss(pred_wh[b, gi], torch.tensor([tw, th], device=device))
                
                # クラス損失
                cls_target = torch.zeros(self.num_classes, device=device)
                cls_target[int(cls_id)] = 1
                loss_cls += F.binary_cross_entropy(pred_cls[b, gi], cls_target)
                
                num_targets += 1
        
        # Confidence損失（正例と負例）
        loss_conf = F.binary_cross_entropy(pred_conf.view(-1), obj_mask.view(-1))
        
        # 合計（重み付け）
        if num_targets > 0:
            loss_xy /= num_targets
            loss_wh /= num_targets
            loss_cls /= num_targets
        
        total_loss = loss_xy * 5 + loss_wh * 5 + loss_conf + loss_cls
        
        return {
            'total': total_loss,
            'xy': loss_xy.item() if isinstance(loss_xy, torch.Tensor) else loss_xy,
            'wh': loss_wh.item() if isinstance(loss_wh, torch.Tensor) else loss_wh,
            'conf': loss_conf.item(),
            'cls': loss_cls.item() if isinstance(loss_cls, torch.Tensor) else loss_cls
        }

# ===== メイン処理 =====
def main():
    print("🚀 Starting Minimal YOLO Training")
    print(f"Device: {DEVICE}")
    
    # ディレクトリ作成
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # データセット
    print("\n📊 Loading dataset...")
    dataset = MinimalDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, IMG_SIZE)
    print(f"Dataset size: {len(dataset)}")
    
    # カスタムcollate関数
    def collate_fn(batch):
        imgs, targets = zip(*batch)
        imgs = torch.stack(imgs, 0)
        return imgs, targets
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          collate_fn=collate_fn, num_workers=2)
    print(f"Total batches: {len(dataloader)}")
    
    # モデル
    print("\n🤖 Creating model...")
    model = MinimalYOLO(NUM_CLASSES).to(DEVICE)
    
    # パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # テスト
    with torch.no_grad():
        test_input = torch.randn(2, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)
        test_output, grid_size = model(test_input)
        print(f"Test output shape: {test_output.shape}, Grid size: {grid_size}")
    
    # 損失関数とオプティマイザ
    criterion = MinimalLoss(NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 学習ループ
    print("\n🎯 Starting training...")
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        epoch_start = time.time()
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(DEVICE)
            
            # Forward
            predictions, grid_size = model(images)
            
            # Loss
            losses = criterion(predictions, targets, grid_size)
            loss = losses['total']
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 進捗表示
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} (xy:{losses['xy']:.3f} wh:{losses['wh']:.3f} "
                      f"conf:{losses['conf']:.3f} cls:{losses['cls']:.3f})")
        
        # エポック終了
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(dataloader)
        print(f"\n📈 Epoch {epoch+1} completed in {epoch_time:.1f}s - Avg Loss: {avg_loss:.4f}")
        
        # モデル保存（毎エポック）
        if epoch % 2 == 0 or epoch == NUM_EPOCHS - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }
            save_path = os.path.join(SAVE_DIR, f'model_epoch_{epoch+1}.pth')
            torch.save(checkpoint, save_path)
            print(f"💾 Model saved: {save_path}")
    
    print("\n✅ Training completed!")

if __name__ == "__main__":
    main()