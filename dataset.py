# dataset.py
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import random # 追加

# ★★★ 共有VersionTrackerをインポート ★★★
from version_tracker import create_version_tracker, VersionTracker

# バージョン管理システム初期化
dataset_version = create_version_tracker("Dataset System v1.2", "dataset.py")
dataset_version.add_modification("FLIR データセット対応")
dataset_version.add_modification("collate_fn実装")
dataset_version.add_modification("データ拡張機能を追加 (Phase 2)") # 追加


class FLIRDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=416, augment=False): # augment引数を追加
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.augment = augment # 追加: データ拡張フラグをインスタンス変数に保存
        
        # configからデータ拡張のパラメータをインポート
        from config import Config # 追加
        self.brightness_range = Config.brightness_range # 追加
        self.noise_level = Config.noise_level # 追加
        
    def __len__(self):
        return len(self.img_files)

    # 追加: データ拡張ロジック
    def augment_image(self, img):
        """シンプルなデータ拡張"""
        # 明度調整
        if random.random() < 0.5:
            factor = 1 + random.uniform(-self.brightness_range, self.brightness_range)
            img = np.clip(img * factor, 0, 1)
        
        # ガウシアンノイズ
        if random.random() < 0.3:
            noise = np.random.normal(0, self.noise_level, img.shape)
            img = np.clip(img + noise, 0, 1)
        
        return img
    
    def __getitem__(self, idx):
        # 画像読み込み
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0 # dtypeをfloat32に指定

        # データ拡張の適用
        if self.augment: # 追加
            img = self.augment_image(img) # 追加
        
        img = torch.from_numpy(img).float() # .unsqueeze(0)はcollate_fnでstackする際に自動で行われるので不要だが、
        # 現在のtrain.pyのdataloader設定ではunsqueeze(0)が必要。一時的に残す。
        # 後でtrain.pyのdataloaderが正しく動作することを確認した上で削除検討
        img = img.unsqueeze(0) 
        
        # ラベル読み込み
        label_path = os.path.join(self.label_dir, 
                                 self.img_files[idx].replace('.jpg', '.txt'))
        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        targets.append(list(map(float, parts)))
        
        # ターゲットをテンソルに（空の場合も対応）
        targets = torch.tensor(targets) if targets else torch.zeros((0, 5))
        return img, targets

def collate_fn(batch):
    """カスタムcollate関数 - 異なるサイズのターゲットを処理"""

    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    # ターゲットはリストのまま返す（各画像で物体数が異なるため）
    return images, list(targets)