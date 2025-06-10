# model.py
import torch
import torch.nn as nn

# ★★★ 共有VersionTrackerをインポート ★★★
from version_tracker import create_version_tracker, VersionTracker

# バージョン管理システム初期化
model_version = create_version_tracker("Model System v1.0", "model.py")
model_version.add_modification("SimpleYOLO実装")
model_version.add_modification("検出ヘッド最適化")


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