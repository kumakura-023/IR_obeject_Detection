# model.py
import torch
import torch.nn as nn

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
training_version = create_version_tracker("Unified Training System v0.0", "model.py")
training_version.add_modification("プロトタイプ")

class SimpleYOLO(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.num_classes = num_classes
        
        # Backbone
        self.features = nn.Sequential(
            self._make_layer(1, 32, 2),      # 416->208
            self._make_layer(32, 64, 2),     # 208->104
            self._make_layer(64, 128, 2),    # 104->52
            self._make_layer(128, 256, 2),   # 52->26
            self._make_layer(256, 512, 2),   # 26->13
        )
        
        # Detection Head
        self.detector = nn.Conv2d(512, 5 + num_classes, 1)
        
    def _make_layer(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.detector(x)
        
        # [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, H * W, C)
        
        return x, (H, W)