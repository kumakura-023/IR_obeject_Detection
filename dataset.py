# dataset.py
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os

import datetime
import hashlib
# ===== ver管理 =====
class VersionTracker:
    """スクリプトのバージョンと修正履歴を追跡"""
    _all_trackers = {}

    def __init__(self, script_name, version="1.0.0"):
        self.script_name = script_name
        self.version = version
        self.load_time = datetime.datetime.now()
        self.modifications = []
        
        VersionTracker._all_trackers[script_name] = self
        
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
training_version = create_version_tracker("Unified Training System v0.0", "dataset.py")
training_version.add_modification("プロトタイプ")

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
    #ver取得
    training_version.print_version_info()

    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    # ターゲットはリストのまま返す（各画像で物体数が異なるため）
    return images, list(targets)