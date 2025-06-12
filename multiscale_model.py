# multiscale_model.py - Step 2: マルチスケール検出実装

import torch
import torch.nn as nn
import torch.nn.functional as F

# Step 1で生成されたアンカーをインポート
ANCHORS = {
    'small':  [(7, 11), (14, 28), (22, 65)],      # 52x52 grid
    'medium': [(42, 35), (76, 67), (46, 126)],    # 26x26 grid  
    'large':  [(127, 117), (88, 235), (231, 218)] # 13x13 grid
}

class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature extraction"""
    
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        
        # Top-down pathway (3x3 conv for feature fusion)
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) 
            for _ in in_channels_list
        ])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize FPN weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        Args:
            features: List of feature maps [C3, C4, C5] from backbone
        Returns:
            fpn_features: List of FPN feature maps [P3, P4, P5]
        """
        # Apply lateral connections
        laterals = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, features)]
        
        # Top-down pathway with feature fusion
        fpn_features = []
        
        # Start from the highest level (smallest feature map)
        prev_feat = laterals[-1]
        fpn_features.append(self.fpn_convs[-1](prev_feat))
        
        # Propagate to lower levels
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample previous feature
            upsampled = F.interpolate(prev_feat, scale_factor=2, mode='nearest')
            
            # Element-wise addition
            fused = laterals[i] + upsampled
            
            # Apply 3x3 conv
            fpn_feat = self.fpn_convs[i](fused)
            fpn_features.append(fpn_feat)
            
            prev_feat = fused
        
        # Reverse to get [P3, P4, P5] order
        fpn_features.reverse()
        
        return fpn_features

class DetectionHead(nn.Module):
    """Detection head for one scale"""
    
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Shared convolutions
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Detection outputs
        # Each anchor predicts: [x, y, w, h, confidence, class1, class2, ..., classN]
        self.detection_conv = nn.Conv2d(256, num_anchors * (5 + num_classes), 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize detection head weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize confidence bias to -2.0 (for sigmoid activation)
        # This helps with training stability
        with torch.no_grad():
            bias = self.detection_conv.bias.view(self.num_anchors, -1)
            bias[:, 4].fill_(-2.0)  # confidence bias
    
    def forward(self, x):
        """
        Args:
            x: Feature map [B, C, H, W]
        Returns:
            detections: [B, H*W*num_anchors, 5+num_classes]
        """
        # Apply shared convolutions
        x = self.shared_conv(x)
        
        # Apply detection convolution
        detections = self.detection_conv(x)
        
        # Reshape: [B, num_anchors*(5+C), H, W] -> [B, H, W, num_anchors, 5+C]
        B, _, H, W = detections.shape
        detections = detections.view(B, self.num_anchors, 5 + self.num_classes, H, W)
        detections = detections.permute(0, 3, 4, 1, 2).contiguous()
        
        # Flatten: [B, H, W, num_anchors, 5+C] -> [B, H*W*num_anchors, 5+C]
        detections = detections.view(B, H * W * self.num_anchors, 5 + self.num_classes)
        
        return detections

class MultiScaleBackbone(nn.Module):
    """Improved backbone for multi-scale detection"""
    
    def __init__(self):
        super().__init__()
        
        # Stage 1: 416 -> 208
        self.stage1 = nn.Sequential(
            self._make_layer(1, 32, stride=2),
            self._make_layer(32, 32, stride=1),
        )
        
        # Stage 2: 208 -> 104  
        self.stage2 = nn.Sequential(
            self._make_layer(32, 64, stride=2),
            self._make_layer(64, 64, stride=1),
        )
        
        # Stage 3: 104 -> 52 (P3 output)
        self.stage3 = nn.Sequential(
            self._make_layer(64, 128, stride=2),
            self._make_layer(128, 128, stride=1),
            self._make_layer(128, 128, stride=1),
        )
        
        # Stage 4: 52 -> 26 (P4 output)
        self.stage4 = nn.Sequential(
            self._make_layer(128, 256, stride=2),
            self._make_layer(256, 256, stride=1),
            self._make_layer(256, 256, stride=1),
        )
        
        # Stage 5: 26 -> 13 (P5 output)
        self.stage5 = nn.Sequential(
            self._make_layer(256, 512, stride=2),
            self._make_layer(512, 512, stride=1),
            self._make_layer(512, 512, stride=1),
        )
        
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def _init_weights(self):
        """Initialize backbone weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Input image [B, 1, 416, 416]
        Returns:
            features: List of feature maps [C3, C4, C5]
        """
        # Forward through stages
        x = self.stage1(x)  # [B, 32, 208, 208]
        x = self.stage2(x)  # [B, 64, 104, 104]
        
        c3 = self.stage3(x)  # [B, 128, 52, 52]  -> P3
        c4 = self.stage4(c3) # [B, 256, 26, 26]  -> P4  
        c5 = self.stage5(c4) # [B, 512, 13, 13]  -> P5
        
        return [c3, c4, c5]

class MultiScaleYOLO(nn.Module):
    """Multi-scale YOLO with FPN"""
    
    def __init__(self, num_classes=15, anchors=None):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors or ANCHORS
        
        # Backbone
        self.backbone = MultiScaleBackbone()
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork([128, 256, 512], out_channels=256)
        
        # Detection heads for each scale
        self.head_small = DetectionHead(256, num_classes, 3)   # 52x52
        self.head_medium = DetectionHead(256, num_classes, 3)  # 26x26
        self.head_large = DetectionHead(256, num_classes, 3)   # 13x13
        
        print(f"🤖 MultiScaleYOLO initialized")
        print(f"   Classes: {num_classes}")
        print(f"   Anchors: {len(self.anchors)} scales")
        print(f"   Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        """
        Args:
            x: Input images [B, 1, 416, 416]
        Returns:
            outputs: Dict with keys ['small', 'medium', 'large']
            Each value: [B, H*W*3, 5+num_classes]
        """
        # Extract features from backbone
        features = self.backbone(x)  # [C3, C4, C5]
        
        # Apply FPN
        fpn_features = self.fpn(features)  # [P3, P4, P5]
        
        # Apply detection heads
        outputs = {
            'small': self.head_small(fpn_features[0]),   # P3: [B, 52*52*3, 20]
            'medium': self.head_medium(fpn_features[1]), # P4: [B, 26*26*3, 20]  
            'large': self.head_large(fpn_features[2])    # P5: [B, 13*13*3, 20]
        }
        
        return outputs

def test_multiscale_model():
    """Test the multi-scale model"""
    print("🧪 Step 2: MultiScaleYOLO テスト開始")
    print("-" * 50)
    
    # Create model
    model = MultiScaleYOLO(num_classes=15)
    model.eval()
    
    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 1, 416, 416)
    
    print(f"📊 入力テスト:")
    print(f"   入力形状: {x.shape}")
    print(f"   メモリ使用量: {x.numel() * x.element_size() / 1024**2:.2f} MB")
    
    # Forward pass
    with torch.no_grad():
        try:
            outputs = model(x)
            
            print(f"\n✅ Forward pass成功!")
            print(f"📊 出力形状:")
            
            total_predictions = 0
            for scale, output in outputs.items():
                B, N, C = output.shape
                grid_size = int((N // 3) ** 0.5)
                total_predictions += N
                
                print(f"   {scale:6s}: {output.shape} -> {grid_size}x{grid_size} grid, 3 anchors")
            
            print(f"   総予測数: {total_predictions:,}")
            
            # Memory usage
            model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
            print(f"\n📊 メモリ使用量:")
            print(f"   モデル: {model_memory:.2f} MB")
            print(f"   入力: {x.numel() * x.element_size() / 1024**2:.2f} MB")
            
            total_memory = model_memory + x.numel() * x.element_size() / 1024**2
            print(f"   合計予想: {total_memory:.2f} MB")
            
            # Parameter count by component
            backbone_params = sum(p.numel() for p in model.backbone.parameters())
            fpn_params = sum(p.numel() for p in model.fpn.parameters())
            head_small_params = sum(p.numel() for p in model.head_small.parameters())
            head_medium_params = sum(p.numel() for p in model.head_medium.parameters())
            head_large_params = sum(p.numel() for p in model.head_large.parameters())
            head_params = head_small_params + head_medium_params + head_large_params
            
            print(f"\n📊 パラメータ分布:")
            print(f"   Backbone: {backbone_params:,}")
            print(f"   FPN: {fpn_params:,}")
            print(f"   Heads: {head_params:,}")
            
            return True, outputs
            
        except Exception as e:
            print(f"❌ Forward passエラー: {e}")
            import traceback
            traceback.print_exc()
            return False, None

def compare_with_old_model():
    """Compare with original SimpleYOLO"""
    print(f"\n🔍 旧モデルとの比較:")
    print("-" * 30)
    
    try:
        # Import old model
        from model import SimpleYOLO
        
        old_model = SimpleYOLO(15)
        new_model = MultiScaleYOLO(15)
        
        old_params = sum(p.numel() for p in old_model.parameters())
        new_params = sum(p.numel() for p in new_model.parameters())
        
        print(f"   旧モデル: {old_params:,} parameters")
        print(f"   新モデル: {new_params:,} parameters")
        print(f"   増加率: {new_params/old_params:.1f}x")
        
        # Test both models
        x = torch.randn(1, 1, 416, 416)
        
        with torch.no_grad():
            old_output, old_grid = old_model(x)
            new_output = new_model(x)
            
            old_predictions = old_output.shape[1]
            new_predictions = sum(out.shape[1] for out in new_output.values())
            
            print(f"   旧予測数: {old_predictions:,}")
            print(f"   新予測数: {new_predictions:,}")
            print(f"   予測数比: {new_predictions/old_predictions:.1f}x")
            
    except ImportError:
        print("   旧モデルが見つかりません（正常）")

def run_step2():
    """Run Step 2"""
    print("🚀 Step 2: マルチスケール検出実装開始")
    print("=" * 60)
    
    try:
        # Test model
        success, outputs = test_multiscale_model()
        
        if not success:
            return False
        
        # Compare models
        compare_with_old_model()
        
        print("\n" + "=" * 60)
        print("✅ Step 2完了!")
        print("=" * 60)
        print(f"📊 実装結果:")
        print(f"   ✅ マルチスケール検出: 3スケール (52x52, 26x26, 13x13)")
        print(f"   ✅ Feature Pyramid Network: 実装済み")
        print(f"   ✅ 3つの検出ヘッド: 各スケール3アンカー")
        print(f"   ✅ 動作確認: エラーなし")
        
        # Memory estimation for training
        print(f"\n📊 学習時メモリ予想 (batch_size=160):")
        base_memory = 100  # モデル自体
        input_memory = 160 * 1 * 416 * 416 * 4 / 1024**2  # 入力データ
        gradient_memory = base_memory  # 勾配（モデルと同程度）
        
        total_memory = base_memory + input_memory + gradient_memory
        print(f"   予想使用量: {total_memory:.0f} MB = {total_memory/1024:.1f} GB")
        print(f"   現在使用量: 6.6 GB")
        print(f"   予想合計: {6.6 + total_memory/1024:.1f} GB")
        
        if total_memory/1024 + 6.6 < 12:
            print("   ✅ T4メモリ内で動作可能")
        else:
            print("   ⚠️ バッチサイズ調整が必要かも")
        
        print(f"\n📝 次のステップ:")
        print(f"   1. 新モデルのメモリ使用量確認")
        print(f"   2. バッチサイズ調整（必要に応じて）")
        print(f"   3. Step 3: アンカーベース損失関数実装")
        
        return True
        
    except Exception as e:
        print(f"❌ Step 2でエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False

# ===== 使用例 =====
if __name__ == "__main__":
    # Step 2実行
    success = run_step2()
    
    if success:
        print("🎉 Step 2成功! Step 3の準備完了")
    else:
        print("❌ Step 2失敗 - エラーを確認してください")