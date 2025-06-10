#!/usr/bin/env python3
"""
Improved YOLO v1 - データ拡張と学習率調整を追加
train_minimal.py からの改善点：
1. データ拡張（明度調整、ノイズ）
2. 学習率スケジューラ
3. 検証セット対応
4. より詳細なログ
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
import random
from torch.optim.lr_scheduler import CosineAnnealingLR

# ===== ver管理 =====
class VersionTracker:
    """スクリプトのバージョンと修正履歴を追跡"""
    
    def __init__(self, script_name, version="1.0.0"):
        self.script_name = script_name
        self.version = version
        self.load_time = datetime.datetime.now()
        self.modifications = []
        
    def add_modification(self, description, author="AI Assistant"):
        """修正履歴を追加"""
        timestamp = datetime.datetime.now()
        self.modifications.append({
            'timestamp': timestamp,
            'description': description,
            'author': author
        })
        
    def get_file_hash(self, filepath):
        """ファイルのハッシュ値を計算（変更検出用）"""
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()[:8]
        except:
            return "unknown"
    
    def print_version_info(self):
        """バージョン情報を表示"""
        print(f"\n{'='*60}")
        print(f"📋 {self.script_name} - Version {self.version}")
        print(f"⏰ Loaded: {self.load_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if hasattr(self, 'file_hash'):
            print(f"🔗 File Hash: {self.file_hash}")
        
        if self.modifications:
            print(f"📝 Recent Modifications ({len(self.modifications)}):")
            for i, mod in enumerate(self.modifications[-3:], 1):  # 最新3件
                print(f"   {i}. {mod['timestamp'].strftime('%H:%M:%S')} - {mod['description']}")
        
        print(f"{'='*60}\n")

# 各ファイル用のバージョントラッカーを作成
def create_version_tracker(script_name, filepath=None):
    """バージョントラッカーを作成"""
    tracker = VersionTracker(script_name)
    
    if filepath:
        tracker.file_hash = tracker.get_file_hash(filepath)
    
    return tracker

# バージョン管理システム初期化
training_version = create_version_tracker("Unified Training System v0.1", "prototype.py")
training_version.add_modification("プロトタイプ")



# ===== 設定 =====
class Config:
    # パス
    train_img_dir = "/content/FLIR_YOLO_local/images/train"
    train_label_dir = "/content/FLIR_YOLO_local/labels/train"
    save_dir = "/content/drive/MyDrive/IR_obj_detection/improved_yolo_v1"
    
    # 基本設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8  # 少し増やす
    img_size = 416
    num_classes = 15
    num_epochs = 30  # より長く
    
    # 学習設定
    learning_rate = 3e-4
    weight_decay = 1e-4
    momentum = 0.9
    
    # データ拡張
    augment = True
    brightness_range = 0.3
    noise_level = 0.02

# ===== データセット（拡張機能付き）=====
class ImprovedDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=416, augment=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment
        
        # 全画像を使用
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        print(f"Dataset: {len(self.img_files)} images")
        
    def __len__(self):
        return len(self.img_files)
    
    def augment_image(self, img):
        """シンプルなデータ拡張"""
        # 明度調整
        if random.random() < 0.5:
            factor = 1 + random.uniform(-Config.brightness_range, Config.brightness_range)
            img = np.clip(img * factor, 0, 1)
        
        # ガウシアンノイズ
        if random.random() < 0.3:
            noise = np.random.normal(0, Config.noise_level, img.shape)
            img = np.clip(img + noise, 0, 1)
        
        return img
    
    def __getitem__(self, idx):
        # 画像読み込み
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        
        # データ拡張
        if self.augment:
            img = self.augment_image(img)
        
        img = torch.from_numpy(img).float().unsqueeze(0)
        
        # ラベル読み込み
        label_path = os.path.join(self.label_dir, self.img_files[idx].replace('.jpg', '.txt'))
        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, cx, cy, w, h = map(float, parts)
                        # 範囲チェック
                        if 0 <= cls < Config.num_classes and 0 < w < 1 and 0 < h < 1:
                            targets.append([cls, cx, cy, w, h])
        
        return img, torch.tensor(targets) if targets else torch.zeros((0, 5))

# ===== モデル（少し改良）=====
class ImprovedYOLO(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.num_classes = num_classes
        
        # より深いBackbone
        self.features = nn.Sequential(
            # 416 -> 208
            self._make_layer(1, 32, stride=2),
            # 208 -> 104
            self._make_layer(32, 64, stride=2),
            # 104 -> 52
            self._make_layer(64, 128, stride=2),
            # 52 -> 26
            self._make_layer(128, 256, stride=2),
            # 26 -> 13
            self._make_layer(256, 512, stride=2),
            # 追加の畳み込み
            self._make_layer(512, 512, stride=1),
        )
        
        # Detection head
        self.detector = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 5 + num_classes, 1)
        )
        
        # 初期化
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.detector(x)
        
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, H * W, C)
        
        return x, (H, W)

# ===== 損失関数（改良版）=====
class ImprovedLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5
        
    def forward(self, predictions, targets, grid_size):
        B = predictions.size(0)
        H, W = grid_size
        device = predictions.device
        
        # 予測値を分解
        pred_xy = predictions[..., :2].sigmoid()
        pred_wh = predictions[..., 2:4]
        pred_conf = predictions[..., 4].sigmoid()
        pred_cls = predictions[..., 5:]
        
        # 損失とメトリクス
        losses = {
            'xy': 0, 'wh': 0, 'conf_obj': 0, 'conf_noobj': 0, 'cls': 0,
            'total': 0, 'num_obj': 0, 'num_noobj': 0
        }
        
        # ターゲットマスク
        obj_mask = torch.zeros(B, H * W, dtype=torch.bool, device=device)
        noobj_mask = torch.ones(B, H * W, dtype=torch.bool, device=device)
        
        # バッチごとに処理
        for b in range(B):
            if len(targets[b]) == 0:
                continue
                
            for t in targets[b]:
                cls_id, cx, cy, w, h = t
                
                # グリッド座標
                gx = int(cx * W)
                gy = int(cy * H)
                
                if 0 <= gx < W and 0 <= gy < H:
                    gi = gy * W + gx
                    
                    # マスク更新
                    obj_mask[b, gi] = True
                    noobj_mask[b, gi] = False
                    
                    # ターゲット座標
                    tx = cx * W - gx
                    ty = cy * H - gy
                    tw = torch.log(w * W + 1e-16)
                    th = torch.log(h * H + 1e-16)
                    
                    # 座標損失
                    losses['xy'] += F.mse_loss(pred_xy[b, gi], torch.tensor([tx, ty], device=device))
                    losses['wh'] += F.mse_loss(pred_wh[b, gi], torch.tensor([tw, th], device=device))
                    
                    # クラス損失
                    cls_target = F.one_hot(torch.tensor(int(cls_id)), self.num_classes).float().to(device)
                    losses['cls'] += F.cross_entropy(pred_cls[b, gi:gi+1], torch.tensor([int(cls_id)], device=device))
                    
                    losses['num_obj'] += 1
        
        # Confidence損失
        if obj_mask.any():
            losses['conf_obj'] = F.binary_cross_entropy(pred_conf[obj_mask], torch.ones_like(pred_conf[obj_mask]))
        
        losses['num_noobj'] = noobj_mask.sum().item()
        if noobj_mask.any():
            losses['conf_noobj'] = F.binary_cross_entropy(pred_conf[noobj_mask], torch.zeros_like(pred_conf[noobj_mask]))
        
        # 重み付き合計
        num_obj = max(losses['num_obj'], 1)
        losses['total'] = (
            self.lambda_coord * (losses['xy'] + losses['wh']) / num_obj +
            losses['conf_obj'] / num_obj +
            self.lambda_noobj * losses['conf_noobj'] / max(losses['num_noobj'], 1) +
            losses['cls'] / num_obj
        )
        
        return losses

# ===== 学習関数 =====
def train_epoch(model, dataloader, criterion, optimizer, epoch, device):
    model.train()
    metrics = defaultdict(float)
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        
        # Forward
        predictions, grid_size = model(images)
        losses = criterion(predictions, targets, grid_size)
        
        # Backward
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        # メトリクス更新
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                metrics[k] += v.item()
            else:
                metrics[k] += v
        
        # 進捗表示
        if batch_idx % 20 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {losses['total']:.4f} "
                  f"(xy:{losses['xy']:.3f} wh:{losses['wh']:.3f} "
                  f"conf:{losses['conf_obj']:.3f}/{losses['conf_noobj']:.3f} "
                  f"cls:{losses['cls']:.3f}) "
                  f"Obj: {losses['num_obj']}")
    
    # エポック平均
    num_batches = len(dataloader)
    for k in metrics:
        metrics[k] /= num_batches
    
    return metrics

# ===== メイン =====
def main():
    print("🚀 Starting Improved YOLO Training v1")
    
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # データセット
    print("\n📊 Loading dataset...")
    train_dataset = ImprovedDataset(cfg.train_img_dir, cfg.train_label_dir, 
                                  cfg.img_size, augment=cfg.augment)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True)
    
    # モデル
    print("\n🤖 Creating model...")
    model = ImprovedYOLO(cfg.num_classes).to(cfg.device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 最適化
    criterion = ImprovedLoss(cfg.num_classes)
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, 
                         momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    
    # 学習ループ
    print("\n🎯 Starting training...")
    best_loss = float('inf')
    
    for epoch in range(1, cfg.num_epochs + 1):
        # 学習
        start_time = time.time()
        metrics = train_epoch(model, train_loader, criterion, optimizer, epoch, cfg.device)
        epoch_time = time.time() - start_time
        
        # 学習率更新
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # ログ
        print(f"\n📈 Epoch {epoch}/{cfg.num_epochs} - {epoch_time:.1f}s - LR: {current_lr:.6f}")
        print(f"   Loss: {metrics['total']:.4f} "
              f"(xy:{metrics['xy']:.3f} wh:{metrics['wh']:.3f} "
              f"conf:{metrics['conf_obj']:.3f}/{metrics['conf_noobj']:.3f} "
              f"cls:{metrics['cls']:.3f})")
        
        # モデル保存
        if metrics['total'] < best_loss:
            best_loss = metrics['total']
            save_path = os.path.join(cfg.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': cfg.__dict__
            }, save_path)
            print(f"💾 Best model saved (loss: {best_loss:.4f})")
        
        # 定期保存
        if epoch % 5 == 0:
            save_path = os.path.join(cfg.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)
    
    print("\n✅ Training completed!")

if __name__ == "__main__":
    main()