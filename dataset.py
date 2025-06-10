# dataset.py
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os

# ★★★ 共有VersionTrackerをインポート ★★★
from version_tracker import create_version_tracker, VersionTracker

# バージョン管理システム初期化
dataset_version = create_version_tracker("Dataset System v1.0", "dataset.py")
dataset_version.add_modification("FLIR データセット対応")
dataset_version.add_modification("collate_fn実装")


class FLIRDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=416):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 画像読み込み
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).float() / 255.0
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