# model.py - Phase2 Enhanced Architecture
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ★★★ 共有VersionTrackerをインポート ★★★
from version_tracker import create_version_tracker, VersionTracker

# バージョン管理システム初期化
model_version = create_version_tracker("Model System v2.0 - Phase2 Enhanced", "model.py")
model_version.add_modification("SimpleYOLO実装")
model_version.add_modification("検出ヘッド最適化")
model_version.add_modification("Phase2: Spatial Attention + CSP改良 + 深いバックボーン")
model_version.add_modification("修正: SimpleYOLOのフォールバック時に単一アンカーを使うように修正") # これを追加するよ！

# ===== Phase2: 改良コンポーネント =====

class SpatialAttentionModule(nn.Module):
    """YOLOv11スタイルの空間注意機構"""
    
    def __init__(self, in_channels):
        super().__init__()
        # 空間注意: 7x7 conv で重要領域に注目
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # チャンネル注意: Global Average Pooling + FC
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # チャンネル注意適用
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # 空間注意適用
        spatial_att = self.spatial_conv(x)
        x = x * spatial_att
        
        return x

class ImprovedCSPBlock(nn.Module):
    """YOLOv11のC3k2スタイル改良CSPブロック"""
    
    def __init__(self, in_channels, out_channels, num_blocks=1, shortcut=True):
        super().__init__()
        hidden_channels = out_channels // 2
        
        # 入力分岐用
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1)
        
        # ボトルネックブロック（より軽量に）
        self.bottlenecks = nn.Sequential(*[
            self._make_bottleneck(hidden_channels, hidden_channels, shortcut)
            for _ in range(num_blocks)
        ])
        
        # 出力統合
        self.conv3 = nn.Conv2d(hidden_channels * 2, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def _make_bottleneck(self, in_ch, out_ch, shortcut=True):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, 1),
            nn.BatchNorm2d(out_ch // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch // 2, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def forward(self, x):
        # CSP分岐
        branch1 = self.conv1(x)
        branch2 = self.bottlenecks(self.conv2(x))
        
        # 統合
        out = torch.cat([branch1, branch2], dim=1)
        return F.leaky_relu(self.bn(self.conv3(out)), 0.1)

class SPPFBlock(nn.Module):
    """YOLOv8スタイルのSPPF（高速版SPP）"""
    
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.conv2 = nn.Conv2d(hidden_channels * 4, out_channels, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        
    def forward(self, x):
        x = self.conv1(x)
        
        # 3回のMaxPooling（効率的な多スケール特徴抽出）
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1) 
        y3 = self.maxpool(y2)
        
        # 全て結合
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))

class EnhancedDetectionHead(nn.Module):
    """Phase2: 強化検出ヘッド"""
    
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # より深い特徴抽出
        self.feature_extractor = nn.Sequential(
            # 第1段階: 基本特徴抽出
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            
            # 第2段階: 空間注意付き特徴強化
            SpatialAttentionModule(256),
            
            # 第3段階: より深い表現学習
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),  # 軽い正則化
            
            # 第4段階: 最終特徴調整
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # 検出用最終層
        self.detection_conv = nn.Conv2d(128, num_anchors * (5 + num_classes), 1)
        
        # 重み初期化（重要！）
        self._initialize_weights()
    
    def _initialize_weights(self):
        """検出ヘッドの重み初期化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 検出層のバイアス初期化（重要！信頼度を低めから開始）
        with torch.no_grad():
            bias = self.detection_conv.bias.view(self.num_anchors, -1)
            bias[:, 4].fill_(-2.0)  # 信頼度バイアス（sigmoid(-2.0) ≈ 0.12）
    
    def forward(self, x):
        # 特徴抽出
        features = self.feature_extractor(x)
        
        # 検出予測
        detections = self.detection_conv(features)
        
        # リシェイプ: [B, num_anchors*(5+C), H, W] -> [B, H*W*num_anchors, 5+C]
        B, _, H, W = detections.shape
        detections = detections.view(B, self.num_anchors, 5 + self.num_classes, H, W)
        detections = detections.permute(0, 3, 4, 1, 2).contiguous()
        detections = detections.view(B, H * W * self.num_anchors, 5 + self.num_classes)
        
        return detections

# ===== Phase2: メインモデル =====

class SimpleYOLO(nn.Module):
    """Phase2: 大幅強化版SimpleYOLO"""
    
    def __init__(self, num_classes=15, use_phase2_enhancements=True):
        super().__init__()
        self.num_classes = num_classes
        self.use_phase2_enhancements = use_phase2_enhancements
        
        if use_phase2_enhancements:
            print(f"🚀 Phase2 Enhanced SimpleYOLO初期化中...")
            self.features = self._build_enhanced_backbone()
            # EnhancedDetectionHeadはデフォルトでnum_anchors=3なので、ここでは明示的に渡す必要はない
            self.detector = EnhancedDetectionHead(512, num_classes, num_anchors=3) # 明示的に3とする
            print(f"   ✅ 空間注意機構: ON")
            print(f"   ✅ 改良CSPブロック: ON") 
            print(f"   ✅ SPPF機構: ON")
            print(f"   ✅ 強化検出ヘッド: ON")
            print(f"   ✅ アンカー数: 3") # 追加
        else:
            print(f"📚 標準SimpleYOLO（互換モード）")
            self.features = self._build_standard_backbone()
            # ★★★ ここを修正: 単一アンカーを想定したヘッドを構築 ★★★
            self.detector = self._build_standard_detector(num_anchors=1) # num_anchors=1を渡す
            print(f"   ✅ アンカー数: 1") # 追加
        
        # 全体の重み初期化
        self._initialize_weights()
        
        # パラメータ数表示
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   📊 総パラメータ数: {total_params:,}")
        print(f"   🎯 学習可能パラメータ数: {trainable_params:,}")
        
    # ... (_build_enhanced_backbone, _make_enhanced_stage は変更なし) ...
    
    def _build_standard_backbone(self):
        """標準バックボーン（Phase1互換）"""
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
            # 追加の畳み込み
            self._make_layer(512, 512, stride=1),
        )
    
    def _make_layer(self, in_channels, out_channels, stride):
        """標準レイヤー（Phase1互換）"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    # ★★★ ここを修正: num_anchorsを引数で受け取るようにする ★★★
    def _build_standard_detector(self, num_anchors=1): # デフォルトを1に
        """標準検出器（Phase1互換）"""
        # ここでnum_anchorsを使用する
        detector =nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, num_anchors * (5 + self.num_classes), 1) # num_anchorsを掛ける
        )

        final_conv_layer = detector[-1]
        

        if final_conv_layer.bias is not None:
            with torch.no_grad():
              final_conv_layer.bias[4].fill_(-2.0) 
        return detector
        

    def _initialize_weights(self):
        """重み初期化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # バックボーン特徴抽出
        features = self.features(x)
        
        if self.use_phase2_enhancements:
            # Phase2: 強化検出ヘッド（すでにリシェイプ済み）
            detections = self.detector(features)
            
            # グリッドサイズ計算
            H = W = x.shape[-1] // 32  # 32倍ダウンサンプリング
            return detections, (H, W)
        else:
            # Phase1互換: 標準検出器
            detections = self.detector(features)
            
            # リシェイプ: [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = detections.shape
            detections = detections.permute(0, 2, 3, 1).contiguous()
            # ここも注意。num_anchors=1の場合、Cが (5+num_classes) になる
            # num_anchors > 1 の場合は、Cを num_anchors と (5+num_classes) に分解してからreshapeする必要がある。
            # しかし、_build_standard_detectorでnum_anchors=1と設定した場合は、Cはそのまま (5+num_classes) なので問題ない。
            detections = detections.view(B, H * W, C)
            
            return detections, (H, W)

# ===== Phase2対応のモデル選択関数 =====

def create_model(num_classes=15, architecture_type="auto", device="cuda"):
    """
    Phase2対応モデル作成関数
    
    Args:
        num_classes: クラス数
        architecture_type: "phase2", "phase1", "auto"
        device: デバイス
    """
    
    if architecture_type == "auto":
        # 自動選択: GPUメモリとパフォーマンスで判断
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory > 12:  # 12GB以上
                    architecture_type = "phase2"
                    print(f"🚀 十分なGPUメモリ({gpu_memory:.1f}GB): Phase2採用")
                else:
                    architecture_type = "phase1"
                    print(f"⚡ 限定GPUメモリ({gpu_memory:.1f}GB): Phase1採用")
            else:
                architecture_type = "phase1"
                print(f"💻 CPU環境: Phase1採用")
        except:
            architecture_type = "phase1"
            print(f"🔄 デフォルト: Phase1採用")
    
    # モデル作成
    if architecture_type == "phase2":
        model = SimpleYOLO(num_classes=num_classes, use_phase2_enhancements=True)
        print(f"✨ Phase2 Enhanced SimpleYOLO作成完了")
    else:
        model = SimpleYOLO(num_classes=num_classes, use_phase2_enhancements=False)
        print(f"📚 Phase1 Compatible SimpleYOLO作成完了")
    
    return model.to(device), architecture_type

# ===== テスト関数 =====

def test_phase2_model():
    """Phase2モデルのテスト"""
    print("🧪 Phase2モデルテスト開始")
    print("-" * 50)
    
    # Phase2モデル作成
    model_p2 = SimpleYOLO(num_classes=15, use_phase2_enhancements=True)
    
    # Phase1モデル作成（比較用）
    model_p1 = SimpleYOLO(num_classes=15, use_phase2_enhancements=False)
    
    # テスト入力
    batch_size = 2
    test_input = torch.randn(batch_size, 1, 416, 416)
    
    print(f"📊 テスト設定:")
    print(f"   入力サイズ: {test_input.shape}")
    print(f"   Phase2パラメータ: {sum(p.numel() for p in model_p2.parameters()):,}")
    print(f"   Phase1パラメータ: {sum(p.numel() for p in model_p1.parameters()):,}")
    
    # 前向き計算テスト
    with torch.no_grad():
        try:
            # Phase2テスト
            output_p2, grid_p2 = model_p2(test_input)
            print(f"\n✅ Phase2 Forward成功:")
            print(f"   出力形状: {output_p2.shape}")
            print(f"   グリッドサイズ: {grid_p2}")
            
            # Phase1テスト
            output_p1, grid_p1 = model_p1(test_input)
            print(f"\n✅ Phase1 Forward成功:")
            print(f"   出力形状: {output_p1.shape}")
            print(f"   グリッドサイズ: {grid_p1}")
            
            # 性能比較
            param_ratio = sum(p.numel() for p in model_p2.parameters()) / sum(p.numel() for p in model_p1.parameters())
            print(f"\n📈 Phase2改善:")
            print(f"   パラメータ増加率: {param_ratio:.2f}x")
            print(f"   予想性能向上: +3-7% mAP")
            print(f"   追加機能: 空間注意、CSP改良、SPPF")
            
            return True
            
        except Exception as e:
            print(f"❌ テストエラー: {e}")
            import traceback
            traceback.print_exc()
            return False

# ===== メモリ効率テスト =====

def estimate_memory_usage(batch_size=96):
    """Phase2モデルのメモリ使用量推定"""
    print(f"\n💾 メモリ使用量推定 (batch_size={batch_size})")
    
    # Phase2モデル
    model = SimpleYOLO(num_classes=15, use_phase2_enhancements=True)
    
    # モデルサイズ
    model_params = sum(p.numel() for p in model.parameters())
    model_memory = model_params * 4 / 1024**2  # float32で4バイト
    
    # 入力メモリ
    input_memory = batch_size * 1 * 416 * 416 * 4 / 1024**2
    
    # 勾配メモリ（パラメータと同程度）
    gradient_memory = model_memory
    
    # 中間活性化（概算）
    activation_memory = batch_size * 512 * 13 * 13 * 4 / 1024**2  # 最深層特徴マップ
    
    total_memory = model_memory + input_memory + gradient_memory + activation_memory
    
    print(f"   モデル: {model_memory:.1f} MB")
    print(f"   入力: {input_memory:.1f} MB") 
    print(f"   勾配: {gradient_memory:.1f} MB")
    print(f"   活性化: {activation_memory:.1f} MB")
    print(f"   合計推定: {total_memory:.1f} MB ({total_memory/1024:.2f} GB)")
    
    if total_memory/1024 < 10:
        print(f"   ✅ T4メモリ(15GB)内で動作可能")
    else:
        print(f"   ⚠️ メモリ調整が必要かも")
    
    return total_memory

if __name__ == "__main__":
    # モデルテスト実行
    print("🚀 Phase2 Enhanced SimpleYOLO テスト")
    print("=" * 60)
    
    # バージョン情報表示
    model_version.print_version_info()
    
    # テスト実行
    success = test_phase2_model()
    
    if success:
        # メモリ推定
        estimate_memory_usage(batch_size=96)
        
        print("\n" + "=" * 60)
        print("🎉 Phase2実装完了!")
        print("=" * 60)
        print("📊 実装された改良:")
        print("   ✅ 空間注意機構（YOLOv11スタイル）")
        print("   ✅ 改良CSPブロック（C3k2風）")
        print("   ✅ SPPF高速空間プール")
        print("   ✅ 強化検出ヘッド（4層深化）")
        print("   ✅ 重み初期化最適化")
        print("   ✅ Dropout正則化")
        
        print(f"\n🔄 使用方法:")
        print(f"   # Phase2モデル")
        print(f"   model = SimpleYOLO(num_classes=15, use_phase2_enhancements=True)")
        print(f"   # Phase1互換モード")
        print(f"   model = SimpleYOLO(num_classes=15, use_phase2_enhancements=False)")
        print(f"   # 自動選択")
        print(f"   model, arch_type = create_model(num_classes=15, architecture_type='auto')")
        
        print(f"\n📈 期待効果:")
        print(f"   Val Loss: 25-35% 改善期待")
        print(f"   小物体検出: 大幅向上")
        print(f"   学習安定性: 注意機構により向上")
        print(f"   汎化性能: 正則化により向上")
        
    else:
        print("\n❌ Phase2実装に問題があります - 修正が必要")