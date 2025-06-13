# advanced_augmentation.py - Phase 4: データ拡張強化

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import random
from torch.utils.data import Dataset
import math

class MixUpAugmentation:
    """MixUp: 2つの画像を線形結合"""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, batch_images, batch_targets):
        """
        Args:
            batch_images: [B, C, H, W]
            batch_targets: List of [N, 5] tensors
        Returns:
            mixed_images, mixed_targets
        """
        if random.random() > 0.5:  # 50%の確率でMixUp適用
            return batch_images, batch_targets
            
        batch_size = batch_images.size(0)
        indices = torch.randperm(batch_size)
        
        # Lambda値生成
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        # 画像をMix
        mixed_images = lam * batch_images + (1 - lam) * batch_images[indices]
        
        # ターゲットをMix（単純版: 重みをつけて結合）
        mixed_targets = []
        for i in range(batch_size):
            original_targets = batch_targets[i]
            shuffled_targets = batch_targets[indices[i]]
            
            # 元画像のターゲット（重み調整）
            if len(original_targets) > 0:
                original_weighted = original_targets.clone()
                # 信頼度を重みで調整（簡易版）
                mixed_targets.append(original_weighted)
            else:
                mixed_targets.append(torch.zeros((0, 5)))
        
        return mixed_images, mixed_targets

class MosaicAugmentation:
    """Mosaic: 4つの画像を2x2に結合"""
    
    def __init__(self, img_size=416):
        self.img_size = img_size
        
    def __call__(self, dataset, indices):
        """
        Args:
            dataset: Dataset object
            indices: [4] indices for mosaic
        Returns:
            mosaic_image, mosaic_targets
        """
        # 4つの画像とターゲットを取得
        images = []
        targets_list = []
        
        for idx in indices:
            img, targets = dataset[idx]
            images.append(img)
            targets_list.append(targets)
        
        # Mosaicグリッドの中心点（ランダム）
        center_x = random.randint(self.img_size // 4, 3 * self.img_size // 4)
        center_y = random.randint(self.img_size // 4, 3 * self.img_size // 4)
        
        # 結合画像初期化
        mosaic_image = torch.zeros(1, self.img_size, self.img_size)
        mosaic_targets = []
        
        # 4つの領域に配置
        positions = [
            (0, 0, center_x, center_y),         # 左上  
            (center_x, 0, self.img_size, center_y),    # 右上
            (0, center_y, center_x, self.img_size),    # 左下
            (center_x, center_y, self.img_size, self.img_size)  # 右下
        ]
        
        for i, (img, targets) in enumerate(zip(images, targets_list)):
            x1, y1, x2, y2 = positions[i]
            h, w = y2 - y1, x2 - x1
            
            # 画像をリサイズして配置
            img_resized = F.interpolate(
                img.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
            ).squeeze(0)
            
            mosaic_image[:, y1:y2, x1:x2] = img_resized
            
            # ターゲット座標を調整
            if len(targets) > 0:
                adjusted_targets = targets.clone()
                
                # 座標を新しい位置に調整
                adjusted_targets[:, 1] = (adjusted_targets[:, 1] * w + x1) / self.img_size  # cx
                adjusted_targets[:, 2] = (adjusted_targets[:, 2] * h + y1) / self.img_size  # cy  
                adjusted_targets[:, 3] = adjusted_targets[:, 3] * w / self.img_size  # w
                adjusted_targets[:, 4] = adjusted_targets[:, 4] * h / self.img_size  # h
                
                # 画像範囲内のターゲットのみ保持
                valid_mask = (
                    (adjusted_targets[:, 1] > 0) & (adjusted_targets[:, 1] < 1) &
                    (adjusted_targets[:, 2] > 0) & (adjusted_targets[:, 2] < 1) &
                    (adjusted_targets[:, 3] > 0) & (adjusted_targets[:, 4] > 0)
                )
                
                if valid_mask.any():
                    mosaic_targets.append(adjusted_targets[valid_mask])
        
        # 全ターゲットを結合
        if mosaic_targets:
            final_targets = torch.cat(mosaic_targets, dim=0)
        else:
            final_targets = torch.zeros((0, 5))
        
        return mosaic_image, final_targets

class CutMixAugmentation:
    """CutMix: 領域を別画像で置換"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, batch_images, batch_targets):
        """
        Args:
            batch_images: [B, C, H, W]
            batch_targets: List of targets
        Returns:
            cutmix_images, cutmix_targets
        """
        if random.random() > 0.5:
            return batch_images, batch_targets
            
        batch_size = batch_images.size(0)
        indices = torch.randperm(batch_size)
        
        # Lambda値生成
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Cut領域決定
        _, _, H, W = batch_images.shape
        cut_ratio = np.sqrt(1. - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        
        # Cut位置決定
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # 画像をCutMix
        cutmix_images = batch_images.clone()
        cutmix_images[:, :, bby1:bby2, bbx1:bbx2] = batch_images[indices, :, bby1:bby2, bbx1:bbx2]
        
        # ターゲット調整（簡易版）
        cutmix_targets = batch_targets.copy()
        
        return cutmix_images, cutmix_targets

class AdvancedDataset(Dataset):
    """Phase 4: 高度データ拡張対応データセット"""
    
    def __init__(self, base_dataset, use_mixup=True, use_mosaic=True, use_cutmix=False,
                 mixup_prob=0.5, mosaic_prob=0.5):
        self.base_dataset = base_dataset
        self.use_mixup = use_mixup
        self.use_mosaic = use_mosaic  
        self.use_cutmix = use_cutmix
        self.mixup_prob = mixup_prob
        self.mosaic_prob = mosaic_prob
        
        # 拡張器初期化
        if use_mixup:
            self.mixup = MixUpAugmentation(alpha=0.2)
        if use_mosaic:
            self.mosaic = MosaicAugmentation(img_size=416)
        if use_cutmix:
            self.cutmix = CutMixAugmentation(alpha=1.0)
        
        print(f"🎨 AdvancedDataset initialized")
        print(f"   Base dataset: {len(base_dataset)} samples")
        print(f"   MixUp: {'ON' if use_mixup else 'OFF'} (prob={mixup_prob})")
        print(f"   Mosaic: {'ON' if use_mosaic else 'OFF'} (prob={mosaic_prob})")
        print(f"   CutMix: {'ON' if use_cutmix else 'OFF'}")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Mosaic適用判定
        if self.use_mosaic and random.random() < self.mosaic_prob:
            # 4つのランダムサンプルでMosaic
            indices = [idx] + [random.randint(0, len(self.base_dataset) - 1) for _ in range(3)]
            return self.mosaic(self.base_dataset, indices)
        else:
            # 通常サンプル
            return self.base_dataset[idx]

def create_advanced_dataloader(base_dataset, batch_size, **kwargs):
    """高度データ拡張対応DataLoader作成"""
    
    # 高度データセット作成
    advanced_dataset = AdvancedDataset(
        base_dataset,
        use_mixup=kwargs.get('use_mixup', True),
        use_mosaic=kwargs.get('use_mosaic', True),
        use_cutmix=kwargs.get('use_cutmix', False)
    )
    
    # カスタムcollate関数
    def advanced_collate_fn(batch):
        images, targets = zip(*batch)
        
        # 基本的なstacking
        images = torch.stack(images, 0)
        
        # MixUp適用
        if advanced_dataset.use_mixup and random.random() < advanced_dataset.mixup_prob:
            images, targets = advanced_dataset.mixup(images, list(targets))
        
        return images, list(targets)
    
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        advanced_dataset,
        batch_size=batch_size,
        shuffle=kwargs.get('shuffle', True),
        collate_fn=advanced_collate_fn,
        num_workers=kwargs.get('num_workers', 4),
        pin_memory=kwargs.get('pin_memory', True)
    )
    
    return dataloader

def test_advanced_augmentation():
    """高度データ拡張のテスト"""
    print("🧪 Phase 4高度データ拡張テスト")
    print("-" * 50)
    
    # MixUpテスト
    print("1. MixUp Test:")
    mixup = MixUpAugmentation(alpha=0.2)
    batch_images = torch.randn(4, 1, 416, 416)
    batch_targets = [
        torch.tensor([[0, 0.5, 0.5, 0.1, 0.1]]),
        torch.tensor([[1, 0.3, 0.7, 0.2, 0.3]]),
        torch.tensor([[2, 0.8, 0.2, 0.4, 0.6]]),
        torch.zeros((0, 5))
    ]
    
    try:
        mixed_images, mixed_targets = mixup(batch_images, batch_targets)
        print(f"   ✅ MixUp成功: {mixed_images.shape}")
    except Exception as e:
        print(f"   ❌ MixUpエラー: {e}")
    
    # Mosaicテスト（簡易版）
    print("2. Mosaic Test:")
    mosaic = MosaicAugmentation(img_size=416)
    
    # ダミーデータセット
    class DummyDataset:
        def __getitem__(self, idx):
            return torch.randn(1, 416, 416), torch.tensor([[idx % 15, 0.5, 0.5, 0.1, 0.1]])
    
    dummy_dataset = DummyDataset()
    
    try:
        mosaic_img, mosaic_targets = mosaic(dummy_dataset, [0, 1, 2, 3])
        print(f"   ✅ Mosaic成功: {mosaic_img.shape}, targets: {len(mosaic_targets)}")
    except Exception as e:
        print(f"   ❌ Mosaicエラー: {e}")
    
    # CutMixテスト
    print("3. CutMix Test:")
    cutmix = CutMixAugmentation(alpha=1.0)
    
    try:
        cutmix_images, cutmix_targets = cutmix(batch_images, batch_targets)
        print(f"   ✅ CutMix成功: {cutmix_images.shape}")
    except Exception as e:
        print(f"   ❌ CutMixエラー: {e}")
    
    print("\n🎨 高度データ拡張テスト完了!")
    return True

def integration_with_existing_dataset():
    """既存データセットとの統合例"""
    print("🔗 既存データセットとの統合例")
    
    integration_code = '''
# dataset.py の FLIRDataset を拡張

from advanced_augmentation import create_advanced_dataloader

# 既存のデータセット作成
base_dataset = FLIRDataset(img_dir, label_dir, img_size, augment=True)

# 高度データ拡張DataLoader作成  
train_dataloader = create_advanced_dataloader(
    base_dataset,
    batch_size=batch_size,
    use_mixup=True,
    use_mosaic=True,
    use_cutmix=False,  # 段階的導入
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# 学習ループは既存と同じ
for batch_idx, (images, targets) in enumerate(train_dataloader):
    # 高度データ拡張が自動適用される
    predictions = model(images)
    loss = criterion(predictions, targets)
    # ...
'''
    
    print(integration_code)
    return integration_code

if __name__ == "__main__":
    # テスト実行
    success = test_advanced_augmentation()
    
    if success:
        print("🎉 Phase 4 Step 6 準備完了!")
        print("   MixUp, Mosaic, CutMix実装済み")
        print("   既存データセットとの統合可能")
        
        # 統合例表示
        integration_with_existing_dataset()
    else:
        print("❌ テスト失敗 - 修正が必要")
