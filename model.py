# model.py - 修正版
import torch
import torch.nn as nn

# ★★★ 共有VersionTrackerをインポート ★★★
from version_tracker import create_version_tracker, VersionTracker

# バージョン管理システム初期化
model_version = create_version_tracker("Model System v1.3", "model.py")
model_version.add_modification("SimpleYOLO実装")
model_version.add_modification("検出ヘッド最適化")
model_version.add_modification("モデル改良 (Backboneを深く、検出ヘッド強化, 重み初期化) (Phase 2)")
model_version.add_modification("メソッド名エラー修正 - 不足メソッド追加")

class SimpleYOLO(nn.Module):
    def __init__(self, num_classes=15, use_phase2_enhancements=False):
        super().__init__()
        self.num_classes = num_classes
        self.use_phase2_enhancements = use_phase2_enhancements
        
        if use_phase2_enhancements:
            print(f"🚀 Phase2 Enhanced SimpleYOLO使用")
            # 拡張版は後で実装
            self.features = self._build_enhanced_backbone()
            self.detector = self._build_enhanced_detector()
        else:
            print(f"📚 標準SimpleYOLO使用")
            # 標準版を構築
            self.features = self._build_standard_backbone()
            self.detector = self._build_standard_detector()
        
        # 重みの初期化
        self._initialize_weights()
        
        # パラメータ数表示
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   📊 総パラメータ数: {total_params:,}")
    
    def _build_standard_backbone(self):
        """標準バックボーン構築"""
        return nn.Sequential(
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
    
    def _build_enhanced_backbone(self):
        """拡張バックボーン構築（Phase2用）"""
        # まずは標準版と同じにしておく
        return self._build_standard_backbone()
    
    def _build_standard_detector(self, num_anchors=1):
        """標準検出ヘッド構築"""
        return nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, num_anchors * (5 + self.num_classes), 1)
        )
    
    def _build_enhanced_detector(self):
        """拡張検出ヘッド構築（Phase2用）"""
        # まずは標準版と同じにしておく
        return self._build_standard_detector(num_anchors=1)
    
    def _make_layer(self, in_ch, out_ch, stride):
        """基本的な畳み込みレイヤー作成"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def _initialize_weights(self):
        """重みの初期化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 信頼度バイアスを低めに設定（偽検出対策）
        try:
            # 最後の層の信頼度チャンネル（インデックス4）を調整
            if hasattr(self.detector[-1], 'bias') and self.detector[-1].bias is not None:
                with torch.no_grad():
                    out_channels = self.detector[-1].out_channels
                    channels_per_anchor = 5 + self.num_classes
                    
                    # 各アンカーの信頼度バイアスを-2.0に設定
                    for i in range(0, out_channels, channels_per_anchor):
                        if i + 4 < out_channels:
                            self.detector[-1].bias[i + 4] = -2.0
                    
                    print(f"   ✅ 信頼度バイアス初期化: -2.0")
        except Exception as e:
            print(f"   ⚠️ 信頼度バイアス初期化スキップ: {e}")
    
    def forward(self, x):
        """前向き計算"""
        # バックボーン特徴抽出
        x = self.features(x)
        
        # 検出ヘッド
        x = self.detector(x)
        
        # 形状変換: [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, H * W, C)
        
        return x, (H, W)

# テスト用関数
def test_simple_yolo():
    """SimpleYOLO動作テスト"""
    print("🧪 SimpleYOLO動作テスト")
    print("-" * 40)
    
    # 標準版テスト
    print("1. 標準版テスト:")
    model_standard = SimpleYOLO(num_classes=15, use_phase2_enhancements=False)
    
    # 拡張版テスト
    print("\n2. 拡張版テスト:")
    model_enhanced = SimpleYOLO(num_classes=15, use_phase2_enhancements=True)
    
    # Forward テスト
    print("\n3. Forward テスト:")
    test_input = torch.randn(2, 1, 416, 416)
    
    with torch.no_grad():
        # 標準版
        output_std, grid_std = model_standard(test_input)
        print(f"   標準版出力: {output_std.shape}, グリッド: {grid_std}")
        
        # 拡張版
        output_enh, grid_enh = model_enhanced(test_input)
        print(f"   拡張版出力: {output_enh.shape}, グリッド: {grid_enh}")
        
        # 信頼度チェック
        conf_std = torch.sigmoid(output_std[..., 4])
        conf_enh = torch.sigmoid(output_enh[..., 4])
        
        print(f"\n4. 信頼度チェック:")
        print(f"   標準版 - 平均: {conf_std.mean():.4f}, 最大: {conf_std.max():.4f}")
        print(f"   拡張版 - 平均: {conf_enh.mean():.4f}, 最大: {conf_enh.max():.4f}")
        
        # 成功判定
        success = True
        if conf_std.mean() > 0.5 or conf_enh.mean() > 0.5:
            print("   ⚠️ 平均信頼度が高すぎる（偽検出の可能性）")
            success = False
        else:
            print("   ✅ 信頼度は適切な範囲")
        
        return success

if __name__ == "__main__":
    success = test_simple_yolo()
    if success:
        print("\n🎉 SimpleYOLO修正版テスト成功!")
    else:
        print("\n❌ まだ調整が必要")