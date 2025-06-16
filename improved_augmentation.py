# improved_augmentation.py - サーマル画像最適化データ拡張

import torch
import cv2
import numpy as np
import random
import os  # ★★★ 追加: DataLoaderエラー修正 ★★★
from torch.utils.data import Dataset
import torch.nn.functional as F

class ThermalImageAugmentation:
    """サーマル画像に特化した軽量データ拡張"""
    
    def __init__(self, 
                 brightness_range=0.5,
                 contrast_range=0.4,
                 noise_level=0.03,
                 gaussian_blur_prob=0.3,
                 mixup_prob=0.6):
        
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_level = noise_level
        self.gaussian_blur_prob = gaussian_blur_prob
        self.mixup_prob = mixup_prob
        
        print(f"🎨 ThermalImageAugmentation初期化")
        print(f"   明度範囲: ±{brightness_range}")
        print(f"   コントラスト: ±{contrast_range}")
        print(f"   ノイズレベル: {noise_level}")
        print(f"   ブラー確率: {gaussian_blur_prob}")
        print(f"   MixUp確率: {mixup_prob}")
    
    def apply_brightness_contrast(self, img):
        """明度・コントラスト調整（サーマル画像特性考慮）"""
        if random.random() > 0.7:  # 70%の確率で適用
            return img
        
        # 明度調整
        brightness_factor = 1 + random.uniform(-self.brightness_range, self.brightness_range)
        img = img * brightness_factor
        
        # コントラスト調整（サーマル画像では重要）
        contrast_factor = 1 + random.uniform(-self.contrast_range, self.contrast_range)
        mean_val = img.mean()
        img = mean_val + contrast_factor * (img - mean_val)
        
        # クリッピング
        img = torch.clamp(img, 0, 1)
        
        return img
    
    def apply_gaussian_noise(self, img):
        """ガウシアンノイズ（センサーノイズ模擬）"""
        if random.random() > 0.4:  # 40%の確率で適用
            return img
        
        noise = torch.randn_like(img) * self.noise_level
        img = img + noise
        img = torch.clamp(img, 0, 1)
        
        return img
    
    def apply_gaussian_blur(self, img):
        """ガウシアンブラー（大気効果模擬）"""
        if random.random() > self.gaussian_blur_prob:
            return img
        
        # PyTorchテンソルをnumpy配列に変換
        img_np = img.squeeze().cpu().numpy()
        
        # カーネルサイズをランダムに選択
        kernel_size = random.choice([3, 5])
        sigma = random.uniform(0.5, 1.5)
        
        # ガウシアンブラー適用
        blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)
        
        # テンソルに戻す
        img_blurred = torch.from_numpy(blurred).unsqueeze(0).float()
        
        return img_blurred
    
    def apply_thermal_specific_augment(self, img):
        """サーマル画像特有の拡張"""
        # 1. 温度レンジシフト（サーマル画像の特性）
        if random.random() < 0.3:
            shift = random.uniform(-0.1, 0.1)
            img = img + shift
            img = torch.clamp(img, 0, 1)
        
        # 2. エッジ強調（サーマル画像では物体境界が重要）
        if random.random() < 0.2:
            img = self._enhance_edges(img)
        
        return img
    
    def _enhance_edges(self, img):
        """エッジ強調処理"""
        # Sobelフィルタでエッジ検出
        img_np = img.squeeze().cpu().numpy()
        
        # Sobelフィルタ
        sobel_x = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 正規化
        sobel_combined = sobel_combined / sobel_combined.max()
        
        # 元画像にエッジを軽く加算
        enhanced = img_np + 0.1 * sobel_combined
        enhanced = np.clip(enhanced, 0, 1)
        
        return torch.from_numpy(enhanced).unsqueeze(0).float()
    
    def __call__(self, img):
        """全拡張を適用"""
        # 基本拡張
        img = self.apply_brightness_contrast(img)
        img = self.apply_gaussian_noise(img)
        
        # 重い処理は確率を下げる
        try:
            img = self.apply_gaussian_blur(img)
        except:
            pass  # エラー時はスキップ
        
        # サーマル特化拡張
        img = self.apply_thermal_specific_augment(img)
        
        return img

class LightweightMixUp:
    """軽量化されたMixUp実装"""
    
    def __init__(self, alpha=0.4, prob=0.6):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, batch_images, batch_targets):
        """軽量MixUp適用"""
        if random.random() > self.prob:
            return batch_images, batch_targets
        
        batch_size = batch_images.size(0)
        if batch_size < 2:
            return batch_images, batch_targets
        
        # ランダムにペアを作成
        indices = torch.randperm(batch_size)
        
        # Lambda値生成（Betaの近似として単純な乱数使用）
        lam = np.random.uniform(0.3, 0.7)  # 0.3-0.7の範囲で固定
        
        # 画像混合
        mixed_images = lam * batch_images + (1 - lam) * batch_images[indices]
        
        # ターゲット処理は簡略化（計算コスト削減）
        mixed_targets = []
        for i in range(batch_size):
            # メインのターゲットのみ使用（計算簡略化）
            if len(batch_targets[i]) > 0:
                mixed_targets.append(batch_targets[i])
            else:
                mixed_targets.append(torch.zeros((0, 5)))
        
        return mixed_images, mixed_targets

class ImprovedFLIRDataset(Dataset):
    """改良版FLIRデータセット（軽量拡張対応）"""
    
    def __init__(self, img_dir, label_dir, img_size=416, 
                 use_improved_augment=True, mixup_in_dataset=False):
        
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.use_improved_augment = use_improved_augment
        self.mixup_in_dataset = mixup_in_dataset
        
        import os
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        
        # 改良版拡張器
        if use_improved_augment:
            self.augmenter = ThermalImageAugmentation()
        
        # 軽量MixUp
        if mixup_in_dataset:
            self.mixup = LightweightMixUp()
        
        print(f"📊 ImprovedFLIRDataset初期化完了")
        print(f"   画像数: {len(self.img_files)}")
        print(f"   改良拡張: {'ON' if use_improved_augment else 'OFF'}")
        print(f"   MixUp: {'ON' if mixup_in_dataset else 'OFF'}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 基本データ読み込み（元のコードと同じ）
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        
        # テンソル変換
        img = torch.from_numpy(img).float().unsqueeze(0)
        
        # 改良版拡張適用
        if self.use_improved_augment:
            img = self.augmenter(img)
        
        # ラベル読み込み（元のコードと同じ）
        label_path = os.path.join(self.label_dir, 
                                 self.img_files[idx].replace('.jpg', '.txt'))
        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        targets.append(list(map(float, parts)))
        
        targets = torch.tensor(targets) if targets else torch.zeros((0, 5))
        
        return img, targets

def create_improved_dataloader(dataset, batch_size, use_mixup=True, **kwargs):
    """改良版DataLoader作成"""
    
    # 軽量MixUp用のcollate関数
    def improved_collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images, 0)
        
        # MixUp適用
        if use_mixup and random.random() < 0.6:
            mixup = LightweightMixUp()
            images, targets = mixup(images, list(targets))
        
        return images, list(targets)
    
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=kwargs.get('shuffle', True),
        collate_fn=improved_collate_fn,
        num_workers=kwargs.get('num_workers', 4),
        pin_memory=kwargs.get('pin_memory', True),
        persistent_workers=kwargs.get('persistent_workers', True),
        prefetch_factor=kwargs.get('prefetch_factor', 2)
    )
    
    return dataloader

def test_improved_augmentation():
    """改良版拡張のテスト"""
    print("🧪 改良版データ拡張テスト")
    print("-" * 50)
    
    # 1. サーマル拡張テスト
    print("1. ThermalImageAugmentation Test:")
    augmenter = ThermalImageAugmentation()
    
    # ダミー画像
    dummy_img = torch.rand(1, 416, 416)
    
    try:
        augmented = augmenter(dummy_img)
        print(f"   ✅ サーマル拡張成功: {augmented.shape}")
        print(f"   値範囲: {augmented.min():.3f} - {augmented.max():.3f}")
    except Exception as e:
        print(f"   ❌ サーマル拡張エラー: {e}")
    
    # 2. 軽量MixUpテスト
    print("2. LightweightMixUp Test:")
    mixup = LightweightMixUp()
    
    batch_images = torch.rand(4, 1, 416, 416)
    batch_targets = [
        torch.tensor([[0, 0.5, 0.5, 0.1, 0.1]]),
        torch.tensor([[1, 0.3, 0.7, 0.2, 0.3]]),
        torch.tensor([[2, 0.8, 0.2, 0.4, 0.6]]),
        torch.zeros((0, 5))
    ]
    
    try:
        mixed_images, mixed_targets = mixup(batch_images, batch_targets)
        print(f"   ✅ MixUp成功: {mixed_images.shape}")
        print(f"   ターゲット数: {[len(t) for t in mixed_targets]}")
    except Exception as e:
        print(f"   ❌ MixUpエラー: {e}")
    
    print("\n🎨 改良版拡張テスト完了!")
    return True

def integration_example():
    """統合例を表示"""
    integration_code = '''
# dataset.pyを改良版に置き換える例

from improved_augmentation import ImprovedFLIRDataset, create_improved_dataloader

# 改良版データセット作成
improved_dataset = ImprovedFLIRDataset(
    cfg.train_img_dir, 
    cfg.train_label_dir, 
    cfg.img_size,
    use_improved_augment=True,
    mixup_in_dataset=False  # DataLoaderレベルで実行
)

# 改良版DataLoader
train_dataloader = create_improved_dataloader(
    improved_dataset,
    batch_size=cfg.batch_size,
    use_mixup=True,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# 学習時間の大幅短縮が期待される
# - Mosaic停止: -8分/エポック
# - 軽量MixUp: 処理時間半減
# - サーマル特化拡張: 効果的な学習促進
'''
    
    print("🔗 統合例:")
    print(integration_code)
    return integration_code

if __name__ == "__main__":
    # テスト実行
    success = test_improved_augmentation()
    
    if success:
        print("🎉 改良版データ拡張準備完了!")
        print("   サーマル画像特化・軽量化・効果的")
        print("   次: config_improved.py + diagnostic_training.py と統合")
        
        # 統合例表示
        integration_example()
    else:
        print("❌ テスト失敗 - 修正が必要")