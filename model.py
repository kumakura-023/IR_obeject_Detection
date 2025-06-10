# model.py
import torch
import torch.nn as nn

# ★★★ 共有VersionTrackerをインポート ★★★
from version_tracker import create_version_tracker, VersionTracker

# バージョン管理システム初期化
model_version = create_version_tracker("Model System v1.0", "model.py")
model_version.add_modification("SimpleYOLO実装")
model_version.add_modification("検出ヘッド最適化")
model_version.add_modification("モデル改良 (Backboneを深く、検出ヘッド強化, 重み初期化) (Phase 2)") # 追加


class SimpleYOLO(nn.Module): # クラス名はSimpleYOLOのままでOK
    def __init__(self, num_classes=15):
        super().__init__()
        self.num_classes = num_classes
        
        # Backbone (より深いBackboneに変更)
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
            # 追加の畳み込み層 (ストライド1)
            self._make_layer(512, 512, stride=1),
        )
        
        # Detection Head (改良)
        self.detector = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True), # ReLUではなくLeakyReLUを適用
            nn.Conv2d(256, 5 + num_classes, 1)
        )

        # 重みの初期化を追加
        self._initialize_weights()
        
    def _make_layer(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    # 重みの初期化メソッドを追加
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
        
        # [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, H * W, C)
        
        return x, (H, W)