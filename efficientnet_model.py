import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b1

# EfficientNet-B1 Backbone（チャンネル数修正版）
class EfficientNetBackbone(nn.Module):
    def __init__(self, in_channels=1, pretrained=True):
        super().__init__()
        # EfficientNet-B1をロード
        efficientnet = efficientnet_b1(pretrained=pretrained)
        
        # 1ch対応：最初のconv層を変更
        self.conv_stem = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        if pretrained:
            # RGBの重みを輝度変換で1chに適応
            rgb_weights = efficientnet.features[0][0].weight.data
            self.conv_stem.weight.data = (0.299 * rgb_weights[:, 0:1, :, :] + 
                                        0.587 * rgb_weights[:, 1:2, :, :] + 
                                        0.114 * rgb_weights[:, 2:3, :, :])
        
        # EfficientNetの特徴を保存
        self.features = efficientnet.features
        self.features[0][0] = self.conv_stem  # 最初の層を置き換え
        
        # EfficientNetの構造を調査してデバッグ
        print(f">> EfficientNet-B1 backbone initialized")
        self._print_feature_info()

    def _print_feature_info(self):
        """各レイヤーの出力チャンネル数を調査"""
        print("\n=== EfficientNet-B1 Layer Structure ===")
        x = torch.randn(1, 1, 640, 512)
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            print(f"Layer {i}: {layer.__class__.__name__} -> output shape: {x.shape}")

    def forward(self, x):
        # print(f">> Backbone input: {x.shape}")
        
        # EfficientNet-B1の実際の構造に基づいて特徴を抽出
        # MBConvブロックの出力を使用
        features = []
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # EfficientNet-B1の特徴抽出ポイント（実際のチャンネル数に基づく）
            if i == 1:  # 最初のMBConvブロック後（16チャンネル）
                feat1 = x
                # print(f">> Feature 1 (Layer {i}): {feat1.shape}")
            elif i == 3:  # 中間のMBConvブロック後（40チャンネル）
                feat2 = x  
                # print(f">> Feature 2 (Layer {i}): {feat2.shape}")
            elif i == 5:  # より深いMBConvブロック後（80チャンネル）
                feat3 = x
                # print(f">> Feature 3 (Layer {i}): {feat3.shape}")
        
        return feat1, feat2, feat3

# 動的FPN実装（チャンネル数を自動調整）
class AdaptiveFPN(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        
        # EfficientNet-B1の実際のチャンネル数に対応
        # 実行時に動的に決定されるが、典型的な値を設定
        self.in_channels = [16, 40, 80]  # EfficientNet-B1の典型的なチャンネル数
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in self.in_channels
        ])
        
        # Output convolutions
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in self.in_channels
        ])
        
        print(f">> AdaptiveFPN initialized for channels: {self.in_channels}")

    def forward(self, feat1, feat2, feat3):
        # print(f">> FPN input shapes: {feat1.shape}, {feat2.shape}, {feat3.shape}")
        
        # 実際の入力チャンネル数を確認
        actual_channels = [feat1.size(1), feat2.size(1), feat3.size(1)]
        
        # 期待値と異なる場合は動的に調整
        if actual_channels != self.in_channels:
            print(f">> Adjusting FPN for actual channels: {actual_channels}")
            self.in_channels = actual_channels
            
            # レイヤーを再作成
            self.lateral_convs = nn.ModuleList([
                nn.Conv2d(in_ch, self.out_channels, 1).to(feat1.device) 
                for in_ch in actual_channels
            ])
            self.output_convs = nn.ModuleList([
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1).to(feat1.device)
                for _ in actual_channels
            ])
        
        # Lateral connections
        p1 = self.lateral_convs[0](feat1)
        p2 = self.lateral_convs[1](feat2)
        p3 = self.lateral_convs[2](feat3)
        
        # Top-down pathway（サイズを動的に調整）
        h2, w2 = p2.shape[2], p2.shape[3]
        h1, w1 = p1.shape[2], p1.shape[3]
        
        # p3をp2のサイズにリサイズ
        p2 = p2 + F.interpolate(p3, size=(h2, w2), mode='nearest')
        
        # p2をp1のサイズにリサイズ  
        p1 = p1 + F.interpolate(p2, size=(h1, w1), mode='nearest')
        
        # Final output convolutions
        p1 = self.output_convs[0](p1)
        p2 = self.output_convs[1](p2)
        p3 = self.output_convs[2](p3)
        
        # print(f">> FPN output shapes: {p1.shape}, {p2.shape}, {p3.shape}")
        
        return p1, p2, p3

# Detection Head（修正版）
class SafeDetectionHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=15, num_anchors=3):
        super().__init__()
        self.num_outputs = 5 + num_classes
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        # Shared convolutions
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # 各スケール用のHead（1つのHeadを共有）
        self.detection_head = nn.Conv2d(in_channels, num_anchors * self.num_outputs, 1)
        
        # 重み初期化
        self._initialize_weights()

    def _initialize_weights(self):
        """重みとバイアスの初期化"""
        print("🔧 Initializing SafeDetectionHead weights...")
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    # 通常の初期化
                    nn.init.constant_(m.bias, 0)
        
        # Detection head のObjectness biasを特別に初期化
        if hasattr(self, 'detection_head') and self.detection_head.bias is not None:
            bias_shape = self.detection_head.bias.shape[0]
            outputs_per_anchor = self.num_outputs  # 5 + num_classes
            
            print(f"   Setting objectness bias to -4.0 for {self.num_anchors} anchors")
            
            # Objectness biasを-2.0に変更（sigmoid(-2.0) ≈ 0.12）
        for anchor_idx in range(self.num_anchors):
            obj_idx = anchor_idx * outputs_per_anchor + 4
            if obj_idx < bias_shape:
                self.detection_head.bias.data[obj_idx] = -2.0  # -4.0 → -2.0
        
        print("✅ Weight initialization completed")

    def _reshape_output(self, x):
        B, _, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        return x.view(B, H * W * self.num_anchors, self.num_outputs)

    def forward(self, p1, p2, p3):
        # Shared processing
        p1 = self.shared_conv(p1)
        p2 = self.shared_conv(p2) 
        p3 = self.shared_conv(p3)
        
        # Head outputs（同じheadを使用）
        out_p1 = self._reshape_output(self.detection_head(p1))
        out_p2 = self._reshape_output(self.detection_head(p2))
        out_p3 = self._reshape_output(self.detection_head(p3))
        
        # Concatenate all outputs
        out = torch.cat([out_p1, out_p2, out_p3], dim=1)

        # ★★★ 根本的な修正：活性化関数を適用せず、生のlogitsを返す ★★★
        # 損失計算に必要なのは生の予測値のため、ここでは何もせずそのまま返す。
        return out

# EfficientNet Detection Model（修正版）
class EfficientNetDetectionModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=15, num_anchors=3, pretrained_backbone=True):
        super().__init__()
        print("🚀 Creating EfficientNetDetectionModel...")
        
        self.backbone = EfficientNetBackbone(in_channels=in_channels, pretrained=pretrained_backbone)
        self.neck = AdaptiveFPN(out_channels=256)
        self.head = SafeDetectionHead(in_channels=256, num_classes=num_classes, num_anchors=num_anchors)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)
        
        print("✅ EfficientNetDetectionModel created successfully")

    def forward(self, x):
        # print(f">> Model input: {x.shape}")
        
        # Backbone特徴抽出
        feat1, feat2, feat3 = self.backbone(x)
        
        # Dropout適用
        feat1 = self.dropout(feat1)
        feat2 = self.dropout(feat2) 
        feat3 = self.dropout(feat3)
        
        # Neck処理
        p1, p2, p3 = self.neck(feat1, feat2, feat3)
        
        # Detection head
        output = self.head(p1, p2, p3)
        # print(f">> Model output: {output.shape}")
        
        return output

# モデル作成関数
def create_efficientnet_model(num_classes=15, pretrained=True):
    """EfficientNetベースの検出モデルを作成"""
    print(f"🏗️ Creating EfficientNet model (classes={num_classes}, pretrained={pretrained})")
    
    model = EfficientNetDetectionModel(
        in_channels=1,
        num_classes=num_classes,
        num_anchors=3,
        pretrained_backbone=pretrained
    )
    
    # パラメータ数確認
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   vs ResNet50 (~25M): {total_params/25_000_000:.1f}x")
    
    return model

# デバッグ用テスト関数
def debug_efficientnet_structure():
    """EfficientNetの構造を詳細に調査"""
    print("=== Debugging EfficientNet-B1 Structure ===")
    
    # EfficientNet-B1をロード
    efficientnet = efficientnet_b1(pretrained=False)
    
    # テスト入力
    x = torch.randn(1, 3, 640, 512)
    
    print("\n=== Layer-by-layer analysis ===")
    features = []
    
    # 各レイヤーを通過させながら形状を記録
    for i, layer in enumerate(efficientnet.features):
        x = layer(x)
        print(f"Layer {i} ({layer.__class__.__name__}): {x.shape}")
        features.append((i, x.shape))
    
    print("\n=== Suitable feature extraction points ===")
    # 適切な特徴抽出ポイントを提案
    for i, shape in features:
        if shape[1] in [16, 24, 40, 80, 112, 192, 320]:  # EfficientNetの典型的なチャンネル数
            print(f"Layer {i}: channels={shape[1]}, spatial={shape[2]}x{shape[3]}")

# テスト関数
def test_model_creation():
    """モデル作成をテスト"""
    print("🧪 Testing model creation...")
    
    try:
        model = create_efficientnet_model(num_classes=15)
        
        # テスト用入力
        x = torch.randn(2, 1, 640, 512)
        print(f"📥 Input shape: {x.shape}")
        
        with torch.no_grad():
            output = model(x)
            print(f"📤 Output shape: {output.shape}")
            print(f"✅ Model test successful!")
            
        return model
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# メイン実行部分
if __name__ == "__main__":
    print("=== Testing Corrected EfficientNet Model ===")
    
    # まず構造を調査
    debug_efficientnet_structure()
    
    print("\n\n=== Creating Detection Model ===")
    model = test_model_creation()
    
    if model is not None:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")