# train_phase3_integrated.py - マルチスケールYOLO統合版
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import os

from config import Config

# ★★★ データセット関連修正 ★★★
try:
    # 改良版を優先
    from improved_augmentation import ImprovedFLIRDataset, create_improved_dataloader
    USE_IMPROVED = True
    print("✅ 改良版データセットを使用")
except ImportError:
    # フォールバック
    from dataset import FLIRDataset, collate_fn
    USE_IMPROVED = False
    print("📚 標準データセットを使用")

# ★★★ Phase 3 新アーキテクチャをインポート ★★★
from multiscale_model import MultiScaleYOLO
from advanced_losses import AdvancedMultiScaleLoss
from post_processing import AdvancedPostProcessor, SoftNMS
from diagnostic_training import DiagnosticTrainer

# ★★★ フォールバック用（従来アーキテクチャ） ★★★
from model import SimpleYOLO
from loss import YOLOLoss

# ★★★ 共有VersionTrackerをインポート ★★★
from version_tracker import (
    create_version_tracker, 
    VersionTracker, 
    show_all_project_versions,
    debug_version_status,
    get_version_count
)

# バージョン管理システム初期化
training_version = create_version_tracker("Training System v3.0 - Diagnostic Integrated", "train.py")
training_version.add_modification("診断機能完全統合")
training_version.add_modification("改良版データセット対応")
training_version.add_modification("エラーハンドリング強化")

# ===== Phase 3: EMAクラス実装 =====
class EMAModel:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# ===== アーキテクチャ選択関数 =====
def create_model_and_loss(cfg):
    """Phase 3アーキテクチャ or フォールバック選択"""
    
    if getattr(cfg, 'use_multiscale_architecture', True):
        print("🚀 Phase 3: マルチスケールアーキテクチャを使用")
        try:
            # Step 1のアンカー（実際にはStep 1で生成されたものを使用）
            anchors = {
                'small':  [(7, 11), (14, 28), (22, 65)],      # 52x52 grid
                'medium': [(42, 35), (76, 67), (46, 126)],    # 26x26 grid  
                'large':  [(127, 117), (88, 235), (231, 218)] # 13x13 grid
            }
            
            model = MultiScaleYOLO(num_classes=cfg.num_classes, anchors=anchors)
            criterion = AdvancedMultiScaleLoss(
                anchors=anchors, 
                num_classes=cfg.num_classes,
                use_ciou=getattr(cfg, 'use_ciou', True),
                use_focal=getattr(cfg, 'use_focal', True),
                use_label_smoothing=getattr(cfg, 'use_label_smoothing', False)
            )
            
            print(f"   ✅ MultiScaleYOLO: {sum(p.numel() for p in model.parameters()):,} parameters")
            print(f"   ✅ AdvancedMultiScaleLoss: 3スケール対応")
            
            return model, criterion, "multiscale"
            
        except Exception as e:
            print(f"⚠️ マルチスケール初期化失敗: {e}")
            print("📚 フォールバックモードに切り替えます")
    
    # フォールバック: 従来アーキテクチャ
    print("📚 フォールバック: 従来アーキテクチャを使用")
    model = SimpleYOLO(cfg.num_classes, use_phase2_enhancements=False)
    criterion = YOLOLoss(cfg.num_classes)
    
    print(f"   ✅ SimpleYOLO: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   ✅ YOLOLoss: 単一スケール")
    
    return model, criterion, "fallback"

# ===== Phase 3: データ分割関数 =====
def create_train_val_split(dataset, val_split=0.15):
    """訓練・検証データの分割"""
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 再現性のため
    )
    
    print(f"📊 データ分割完了:")
    print(f"   Train: {train_size} images ({100*(1-val_split):.1f}%)")
    print(f"   Validation: {val_size} images ({100*val_split:.1f}%)")
    return train_dataset, val_dataset

# ===== Phase 3: 検証関数（マルチスケール対応） =====
def validate_model(model, val_dataloader, criterion, device, architecture_type):
    """検証実行（アーキテクチャ自動判定）"""
    model.eval()
    total_val_loss = 0
    val_batches = 0
    
    with torch.no_grad():
        for images, targets in val_dataloader:
            images = images.to(device, non_blocking=True)
            
            if architecture_type == "multiscale":
                # マルチスケール: 辞書形式の出力
                predictions = model(images)
                loss = criterion(predictions, targets)
            else:
                # 従来: タプル形式の出力
                predictions, grid_size = model(images)
                loss = criterion(predictions, targets, grid_size)
            
            total_val_loss += loss.item()
            val_batches += 1
    
    avg_val_loss = total_val_loss / val_batches if val_batches > 0 else float('inf')
    return avg_val_loss

def validate_model_with_postprocessing(model, val_dataloader, criterion, device, architecture_type, use_advanced_postprocessing=True):
    """後処理を含む詳細検証"""
    model.eval()
    total_val_loss = 0
    val_batches = 0
    
    # 後処理システム初期化
    if use_advanced_postprocessing:
        post_processor = AdvancedPostProcessor(
            use_soft_nms=True,
            use_tta=False,      # 検証時は時間効率重視
            use_multiscale=False,
            conf_threshold=0.25,  # 改良設定
            iou_threshold=0.5
        )
        
        detection_stats = {
            'total_detections': 0,
            'high_conf_detections': 0,  # conf > 0.7
            'processed_detections': 0
        }
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_dataloader):
            images = images.to(device, non_blocking=True)
            
            # 通常の損失計算
            if architecture_type == "multiscale":
                predictions = model(images)
                loss = criterion(predictions, targets)
            else:
                predictions, grid_size = model(images)
                loss = criterion(predictions, targets, grid_size)
            
            total_val_loss += loss.item()
            val_batches += 1
            
            # 後処理テスト（サンプリング検証）
            if use_advanced_postprocessing and batch_idx % 10 == 0:  # 10バッチに1回
                try:
                    # 1枚目の画像で後処理テスト
                    single_image = images[0:1]
                    single_pred = {k: v[0:1] for k, v in predictions.items()} if isinstance(predictions, dict) else predictions[0:1]
                    
                    # 後処理実行
                    processed_detections = post_processor.process_predictions(
                        model, single_image, single_pred
                    )
                    
                    # 統計更新
                    detection_stats['total_detections'] += len(processed_detections)
                    high_conf = sum(1 for det in processed_detections if det['score'] > 0.7)
                    detection_stats['high_conf_detections'] += high_conf
                    detection_stats['processed_detections'] += 1
                    
                except Exception as e:
                    pass  # 後処理エラーは無視
    
    avg_val_loss = total_val_loss / val_batches if val_batches > 0 else float('inf')
    
    # 後処理統計を表示
    if use_advanced_postprocessing and detection_stats['processed_detections'] > 0:
        avg_detections = detection_stats['total_detections'] / detection_stats['processed_detections']
        avg_high_conf = detection_stats['high_conf_detections'] / detection_stats['processed_detections']
        
        print(f"📊 後処理統計 (サンプル{detection_stats['processed_detections']}枚):")
        print(f"   平均検出数: {avg_detections:.1f}/画像")
        print(f"   高信頼度検出: {avg_high_conf:.1f}/画像 (conf>0.7)")
    
    return avg_val_loss

# ===== データローダーセットアップ（修正版） =====
def setup_dataloaders(cfg):
    """データローダーセットアップ（改良版対応）"""
    
    if USE_IMPROVED:
        # 改良版データセット使用
        print("🎨 改良版データセット使用中...")
        full_dataset = ImprovedFLIRDataset(
            cfg.train_img_dir, 
            cfg.train_label_dir, 
            cfg.img_size,
            use_improved_augment=True,
            mixup_in_dataset=False  # DataLoaderレベルで処理
        )
        
        # 検証分割
        if cfg.validation_split > 0:
            train_dataset, val_dataset = create_train_val_split(full_dataset, cfg.validation_split)
        else:
            train_dataset = full_dataset
            val_dataset = None
        
        # 改良版DataLoader
        train_dataloader = create_improved_dataloader(
            train_dataset,
            batch_size=cfg.batch_size,
            use_mixup=getattr(cfg, 'use_mixup', True),
            shuffle=True,
            num_workers=2,  # ★★★ 修正: 4 → 2 (警告対策) ★★★
            pin_memory=getattr(cfg, 'pin_memory', True),
            persistent_workers=False,  # ★★★ 修正: True → False (安定性向上) ★★★
            prefetch_factor=2  # ★★★ 修正: デフォルト値明示 ★★★
        )
        
        val_dataloader = None
        if val_dataset:
            val_dataloader = create_improved_dataloader(
                val_dataset,
                batch_size=cfg.batch_size,
                use_mixup=False,  # 検証時はMixUpなし
                shuffle=False,
                num_workers=2,  # ★★★ 修正: 4 → 2 ★★★
                pin_memory=getattr(cfg, 'pin_memory', True),
                persistent_workers=False  # ★★★ 修正: 安定性向上 ★★★
            )
    
    else:
        # 標準データセット使用
        print("📚 標準データセット使用中...")
        full_dataset = FLIRDataset(
            cfg.train_img_dir, 
            cfg.train_label_dir, 
            cfg.img_size, 
            augment=getattr(cfg, 'augment', True)
        )
        
        # 検証分割
        if cfg.validation_split > 0:
            train_dataset, val_dataset = create_train_val_split(full_dataset, cfg.validation_split)
        else:
            train_dataset = full_dataset
            val_dataset = None
        
        # DataLoader作成
        num_workers = 2  # ★★★ 修正: 4 → 2 (警告対策) ★★★
        pin_memory = getattr(cfg, 'pin_memory', True)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False  # ★★★ 修正: 安定性向上 ★★★
        )
        
        val_dataloader = None
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=False  # ★★★ 修正: 安定性向上 ★★★
            )
    
    print(f"   Train batches: {len(train_dataloader)}")
    if val_dataloader:
        print(f"   Validation batches: {len(val_dataloader)}")
    
    return train_dataloader, val_dataloader

# ===== Phase 3: マルチスケール対応学習ループ（診断統合版） =====
def phase3_integrated_training_loop(model, train_dataloader, val_dataloader, criterion, cfg, architecture_type):
    """Phase 3統合学習ループ（診断機能完全統合）"""
    
    # ★★★ 診断機能初期化 ★★★
    diagnostics = DiagnosticTrainer(
        save_dir=os.path.join(cfg.save_dir, "diagnostics")
    )
    
    # オプティマイザ設定
    if getattr(cfg, 'optimizer_type', 'AdamW') == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=getattr(cfg, 'weight_decay', 2e-4),
            betas=getattr(cfg, 'betas', (0.9, 0.999)),
            eps=getattr(cfg, 'eps', 1e-8)
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    # EMA初期化
    ema = None
    if getattr(cfg, 'use_ema', True):
        ema = EMAModel(model, decay=getattr(cfg, 'ema_decay', 0.9995))
        print(f"🔄 EMA initialized with decay {cfg.ema_decay}")
    
    # スケジューラ設定
    scheduler = None
    if getattr(cfg, 'use_scheduler', True):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=cfg.num_epochs,
            eta_min=getattr(cfg, 'min_lr', cfg.learning_rate / 250)
        )
    
    # Early Stopping設定
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 学習統計
    train_losses = []
    val_losses = []
    learning_rates = []
    
    print(f"🚀 Phase 3統合学習開始（診断機能付き）")
    print(f"   Architecture: {architecture_type}")
    print(f"   Optimizer: {getattr(cfg, 'optimizer_type', 'AdamW')}")
    print(f"   EMA: {'ON' if getattr(cfg, 'use_ema', True) else 'OFF'}")
    print(f"   Validation: {'ON' if val_dataloader else 'OFF'}")
    print(f"   Diagnostics: {diagnostics.save_dir}")
    
    for epoch in range(cfg.num_epochs):
        epoch_start = time.time()
        
        # ★★★ エポック診断開始 ★★★
        diagnostics.start_epoch_diagnosis(epoch + 1)
        
        # ===== ウォームアップ処理 =====
        warmup_epochs = getattr(cfg, 'warmup_epochs', 0)
        if epoch < warmup_epochs:
            warmup_lr = cfg.learning_rate * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"🔥 Warmup Epoch {epoch+1}: LR = {warmup_lr:.6f}")
        
        # ===== 訓練フェーズ =====
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        # 進捗トラッカー初期化（利用可能な場合）
        try:
            from progress import MultiScaleProgressTracker
            progress_tracker = MultiScaleProgressTracker(len(train_dataloader), print_interval=getattr(cfg, 'print_interval', 5))
            progress_tracker.start_epoch(epoch + 1, cfg.num_epochs)
            use_progress_tracker = True
        except ImportError:
            use_progress_tracker = False
            print_interval = getattr(cfg, 'print_interval', 5)
        
        for batch_idx, (images, targets) in enumerate(train_dataloader):
            images = images.to(cfg.device, non_blocking=True)
            
            # Forward（アーキテクチャ別）
            if architecture_type == "multiscale":
                predictions = model(images)
                loss = criterion(predictions, targets)
                
                # ★★★ 診断情報取得 ★★★
                loss_components = None
                try:
                    # 詳細情報付きで再計算（診断用）
                    if hasattr(criterion, 'return_components') and batch_idx % 20 == 0:
                        criterion.return_components = True
                        _, loss_components = criterion(predictions, targets)
                        criterion.return_components = False
                except:
                    pass  # エラー時は無視
                
            else:
                predictions, grid_size = model(images)
                loss = criterion(predictions, targets, grid_size)
                loss_components = None
            
            # ★★★ バッチ診断実行 ★★★
            try:
                diagnostics.log_batch_diagnosis(
                    batch_idx, images, targets, predictions, loss_components
                )
            except Exception as e:
                if batch_idx % 50 == 0:  # エラーログを減らす
                    print(f"   ⚠️ 診断エラー (batch {batch_idx}): {e}")
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # 勾配クリッピング
            if hasattr(cfg, 'gradient_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            
            optimizer.step()
            
            # EMA更新
            if ema:
                ema.update()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # 進捗表示
            current_lr = optimizer.param_groups[0]['lr']
            if use_progress_tracker:
                # 詳細進捗表示
                if architecture_type == "multiscale" and loss_components:
                    progress_tracker.update_batch_multiscale(
                        batch_idx, loss.item(), current_lr, loss_components
                    )
                else:
                    progress_tracker.update_batch(batch_idx, loss.item(), current_lr)
            else:
                # 簡易進捗表示
                if batch_idx % print_interval == 0:
                    print(f"   Batch {batch_idx:4d}/{len(train_dataloader)}: Loss={loss.item():.4f}, LR={current_lr:.6f}")
        
        avg_train_loss = epoch_loss / batch_count
        
        # ===== 検証フェーズ =====
        val_loss = float('inf')
        if val_dataloader and epoch % getattr(cfg, 'validate_every', 1) == 0:
            # EMAモデルで検証
            if ema:
                ema.apply_shadow()
            
            # 詳細検証（5エポックに1回）
            if epoch % 5 == 0 and getattr(cfg, 'use_advanced_postprocessing', True):
                print(f"🔧 後処理込み詳細検証実行中...")
                val_loss = validate_model_with_postprocessing(
                    model, val_dataloader, criterion, cfg.device, architecture_type
                )
            else:
                # 通常検証（軽量）
                val_loss = validate_model(
                    model, val_dataloader, criterion, cfg.device, architecture_type
                )
            
            if ema:
                ema.restore()
        
        # スケジューラ更新
        if scheduler and epoch >= warmup_epochs:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # 統計記録
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        learning_rates.append(current_lr)
        
        # ★★★ エポック診断終了 & 改善提案 ★★★
        suggestions = diagnostics.end_epoch_diagnosis(epoch + 1, val_loss, model)
        
        # ログ表示
        val_str = f"Val: {val_loss:6.4f}" if val_loss != float('inf') else "Val: ----"
        print(f"\n📈 Epoch [{epoch+1:2d}/{cfg.num_epochs}] "
              f"Train: {avg_train_loss:6.4f} {val_str} "
              f"Time: {epoch_time:4.1f}s LR: {current_lr:.6f}")
        
        # 改善提案表示
        if suggestions:
            print(f"💡 診断提案:")
            for suggestion in suggestions[:2]:  # 上位2件のみ
                print(f"   {suggestion['type']}: {suggestion['suggestion']}")
        
        # GPU使用量表示（最初の数エポック）
        if cfg.device.type == 'cuda' and epoch < 3:
            memory_used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"   GPU Memory: {memory_used:.2f}GB")
        
        # Early Stopping & Best Model Saving
        current_loss = val_loss if val_loss != float('inf') else avg_train_loss
        min_improvement = getattr(cfg, 'min_improvement', 0.005)
        
        if current_loss < best_val_loss - min_improvement:
            best_val_loss = current_loss
            patience_counter = 0
            
            # EMAモデルを保存
            if ema:
                ema.apply_shadow()
            
            save_best_model_integrated(model, optimizer, ema, epoch, best_val_loss, cfg, architecture_type)
            
            if ema:
                ema.restore()
            
            print(f"🎉 New best loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            patience = getattr(cfg, 'patience', 10)
            print(f"⏳ No improvement for {patience_counter}/{patience} epochs")
            
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered")
                break
        
        # 定期保存
        save_interval = getattr(cfg, 'save_interval', 2)
        if (epoch + 1) % save_interval == 0:
            save_checkpoint_integrated(model, optimizer, epoch, avg_train_loss, cfg, architecture_type)
        
        # メモリクリーンアップ
        if cfg.device.type == 'cuda':
            empty_cache_every = getattr(cfg, 'empty_cache_every_n_batch', 50)
            if (epoch + 1) % (empty_cache_every // 10) == 0:  # エポックベースで調整
                torch.cuda.empty_cache()
    
    # ★★★ 最終診断レポート ★★★
    diagnostics.generate_final_report()
    
    print(f"\n✅ Phase 3統合学習完了!")
    print(f"🏆 Best Loss: {best_val_loss:.4f}")
    
    return train_losses, val_losses, learning_rates, best_val_loss

# ===== 統合版保存関数 =====
def save_best_model_integrated(model, optimizer, ema, epoch, loss, cfg, architecture_type):
    """統合版ベストモデル保存"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'architecture_type': architecture_type,
        'ema_state_dict': ema.shadow if ema else None,
        'config': {
            'batch_size': cfg.batch_size,
            'learning_rate': cfg.learning_rate,
            'ema_decay': getattr(cfg, 'ema_decay', None),
            'validation_split': getattr(cfg, 'validation_split', 0.15),
            'use_multiscale': getattr(cfg, 'use_multiscale_architecture', True)
        },
        'training_stats': {
            'gpu_memory_peak': torch.cuda.max_memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0,
            'parameters': sum(p.numel() for p in model.parameters())
        },
        'version_info': VersionTracker.get_all_trackers()
    }
    
    save_path = os.path.join(cfg.save_dir, f'phase3_integrated_{architecture_type}_loss_{loss:.4f}.pth')
    torch.save(checkpoint, save_path)
    print(f"💾 Phase 3 integrated model saved: {save_path}")

def save_checkpoint_integrated(model, optimizer, epoch, loss, cfg, architecture_type):
    """統合版チェックポイント保存"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'architecture_type': architecture_type,
        'config_info': {
            'batch_size': cfg.batch_size,
            'learning_rate': cfg.learning_rate,
            'optimizer_type': getattr(cfg, 'optimizer_type', 'AdamW')
        },
        'version_info': VersionTracker.get_all_trackers()
    }
    
    save_path = os.path.join(cfg.save_dir, f'checkpoint_integrated_epoch_{epoch+1}.pth')
    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint saved: {save_path}")

def plot_training_progress(losses, lrs, val_losses=None, save_path="training_progress_integrated.png"):
    """学習進捗を可視化（統合版）"""
    try:
        import matplotlib.pyplot as plt
        
        if val_losses:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Training Loss plot
        ax1.plot(losses, 'b-', linewidth=2, label='Train Loss')
        if val_losses:
            valid_val_losses = [loss for loss in val_losses if loss != float('inf')]
            valid_epochs = [i for i, loss in enumerate(val_losses) if loss != float('inf')]
            if valid_val_losses:
                ax1.plot(valid_epochs, valid_val_losses, 'r-', linewidth=2, label='Val Loss')
        
        ax1.set_title('Phase 3 Integrated Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        if val_losses:
            ax1.legend()
        
        # Learning Rate plot
        ax2.plot(lrs, 'r-', linewidth=2)
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Validation Loss separate plot
        if val_losses:
            valid_val_losses = [loss for loss in val_losses if loss != float('inf')]
            valid_epochs = [i for i, loss in enumerate(val_losses) if loss != float('inf')]
            if valid_val_losses:
                ax3.plot(valid_epochs, valid_val_losses, 'g-', linewidth=2)
                ax3.set_title('Validation Loss')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Val Loss')
                ax3.grid(True, alpha=0.3)
                ax3.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()  # メモリ節約
        
        print(f"📊 Training progress saved to {save_path}")
    except ImportError:
        print("⚠️ matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"⚠️ 可視化エラー: {e}")

# ===== GPU環境チェック =====
def comprehensive_gpu_check():
    """包括的なGPU環境チェック"""
    print("\n🔍 GPU環境詳細チェック")
    print("="*60)
    
    print(f"1. CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name(0)}")
        
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   GPU memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
    else:
        print("   ❌ CUDA not available!")
        return False
    
    print(f"2. PyTorch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"3. Selected device: {device}")
    
    return True

# ===== メイン関数 =====
def main():
    print("🚀 Starting Phase 3 Integrated YOLO Training (Diagnostic Enhanced)")
    
    # プロジェクト全体のバージョン情報を表示
    print("\n" + "="*80)
    print("📋 プロジェクト全体バージョン確認")
    print("="*80)
    VersionTracker.print_all_versions()

    # 設定とGPU確認
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # 診断ディレクトリ作成
    diagnostic_dir = os.path.join(cfg.save_dir, "diagnostics")
    os.makedirs(diagnostic_dir, exist_ok=True)
    
    # GPU環境詳細チェック
    if not comprehensive_gpu_check():
        print("❌ GPU使用不可 - CPU学習に切り替えます")
        cfg.device = torch.device('cpu')
        cfg.batch_size = max(cfg.batch_size // 4, 1)
        print(f"   CPU用にバッチサイズを {cfg.batch_size} に調整")
    
    print(f"\n📋 学習設定:")
    print(f"   Device: {cfg.device}")
    print(f"   Batch size: {cfg.batch_size}")
    print(f"   Image size: {cfg.img_size}")
    print(f"   Classes: {cfg.num_classes}")
    print(f"   Learning rate: {cfg.learning_rate:.0e}")
    print(f"   EMA: {getattr(cfg, 'use_ema', True)}")
    print(f"   Validation Split: {getattr(cfg, 'validation_split', 0.15)}")
    print(f"   Diagnostics: ON")
    
    # Phase 3統合: データセット & 検証分割
    print("\n📊 Loading dataset with validation split...")
    try:
        train_dataloader, val_dataloader = setup_dataloaders(cfg)
        
        print(f"   Total train batches: {len(train_dataloader)}")
        if val_dataloader:
            print(f"   Total validation batches: {len(val_dataloader)}")
    except Exception as e:
        print(f"❌ データローダー作成エラー: {e}")
        print("   標準データセットでリトライ中...")
        
        # フォールバック
        global USE_IMPROVED
        USE_IMPROVED = False
        train_dataloader, val_dataloader = setup_dataloaders(cfg)
    
    # Phase 3統合: モデル&損失関数（アーキテクチャ自動選択）
    print("\n🤖 Creating Phase 3 integrated model...")
    model, criterion, architecture_type = create_model_and_loss(cfg)
    
    # GPUに移動
    print(f"   Moving model to {cfg.device}...")
    model = model.to(cfg.device)
    
    if cfg.device.type == 'cuda':
        model = model.float()
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    model_device = next(model.parameters()).device
    print(f"   Model device confirmed: {model_device}")
    
    # Phase 3統合学習実行
    print("\n🚀 Phase 3統合学習を開始（診断機能付き）")
    print(f"🎯 目標: Val Loss 43.45 → 25-30")
    
    try:
        train_losses, val_losses, lrs, best_loss = phase3_integrated_training_loop(
            model, train_dataloader, val_dataloader, criterion, cfg, architecture_type
        )
        
        # 結果可視化
        try:
            plot_training_progress(train_losses, lrs, val_losses)
        except Exception as e:
            print(f"⚠️ 可視化エラー: {e}")
        
        print(f"\n🎯 Phase 3統合学習完了!")
        print(f"🏆 Best Loss: {best_loss:.4f}")
        print(f"🔧 Architecture: {architecture_type}")
        
        # 目標達成判定
        print(f"\n📊 目標達成判定:")
        if best_loss < 30.0:
            print("🎉 Step 1目標達成! (Val Loss < 30.0)")
            if best_loss < 20.0:
                print("🏆 予想を上回る成果! (Val Loss < 20.0)")
        elif best_loss < 35.0:
            print("🟡 部分的改善 (Val Loss < 35.0) - 継続推奨")
        else:
            print("🔴 目標未達成 - 方針転換検討")
            print("   推奨: 学習率をさらに2倍、またはアンカー見直し")
        
        # 具体的な改善提案
        improvement_ratio = 43.45 / best_loss if best_loss > 0 else 1
        print(f"\n💡 改善結果:")
        print(f"   改善率: {improvement_ratio:.1f}x (43.45 → {best_loss:.4f})")
        print(f"   次目標: Val Loss < {best_loss * 0.7:.1f}")
        
        if best_loss > 35.0:
            print(f"\n🚨 緊急改善案:")
            print(f"   1. 学習率を2倍に ({cfg.learning_rate:.0e} → {cfg.learning_rate*2:.0e})")
            print(f"   2. バッチサイズを半分に ({cfg.batch_size} → {cfg.batch_size//2})")
            print(f"   3. アンカーサイズの全面見直し")
        
    except Exception as e:
        print(f"❌ 学習エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ Training completed!")
    
    # 最終統計
    if cfg.device.type == 'cuda':
        final_memory = torch.cuda.memory_allocated(0) / 1024**3
        max_memory = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"📊 最終GPU統計:")
        print(f"   現在のメモリ使用量: {final_memory:.2f}GB")
        print(f"   最大メモリ使用量: {max_memory:.2f}GB")
    
    # 学習結果サマリー
    print(f"\n🎯 学習結果サマリー:")
    print(f"   Architecture: {architecture_type}")
    print(f"   最終Loss: {best_loss:.4f}")
    print(f"   改良拡張: {'ON' if USE_IMPROVED else 'OFF'}")
    print(f"   診断機能: ON")
    print(f"   EMA使用: {'Yes' if getattr(cfg, 'use_ema', True) else 'No'}")
    print(f"   検証分割: {'Yes' if getattr(cfg, 'validation_split', 0.15) > 0 else 'No'}")
    print(f"   保存先: {cfg.save_dir}")
    print(f"   診断ログ: {diagnostic_dir}")

if __name__ == "__main__":
    main()# train_phase3_integrated.py - マルチスケールYOLO統合版
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import os

from config import Config

# ★★★ データセット関連修正 ★★★
try:
    # 改良版を優先
    from improved_augmentation import ImprovedFLIRDataset, create_improved_dataloader
    USE_IMPROVED = True
    print("✅ 改良版データセットを使用")
except ImportError:
    # フォールバック
    from dataset import FLIRDataset, collate_fn
    USE_IMPROVED = False
    print("📚 標準データセットを使用")

# ★★★ Phase 3 新アーキテクチャをインポート ★★★
from multiscale_model import MultiScaleYOLO
from advanced_losses import AdvancedMultiScaleLoss
from post_processing import AdvancedPostProcessor, SoftNMS
from diagnostic_training import DiagnosticTrainer

# ★★★ フォールバック用（従来アーキテクチャ） ★★★
from model import SimpleYOLO
from loss import YOLOLoss

# ★★★ 共有VersionTrackerをインポート ★★★
from version_tracker import (
    create_version_tracker, 
    VersionTracker, 
    show_all_project_versions,
    debug_version_status,
    get_version_count
)

# バージョン管理システム初期化
training_version = create_version_tracker("Training System v3.0 - Diagnostic Integrated", "train.py")
training_version.add_modification("診断機能完全統合")
training_version.add_modification("改良版データセット対応")
training_version.add_modification("エラーハンドリング強化")

# ===== Phase 3: EMAクラス実装 =====
class EMAModel:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# ===== アーキテクチャ選択関数 =====
def create_model_and_loss(cfg):
    """Phase 3アーキテクチャ or フォールバック選択"""
    
    if getattr(cfg, 'use_multiscale_architecture', True):
        print("🚀 Phase 3: マルチスケールアーキテクチャを使用")
        try:
            # Step 1のアンカー（実際にはStep 1で生成されたものを使用）
            anchors = {
                'small':  [(7, 11), (14, 28), (22, 65)],      # 52x52 grid
                'medium': [(42, 35), (76, 67), (46, 126)],    # 26x26 grid  
                'large':  [(127, 117), (88, 235), (231, 218)] # 13x13 grid
            }
            
            model = MultiScaleYOLO(num_classes=cfg.num_classes, anchors=anchors)
            criterion = AdvancedMultiScaleLoss(
                anchors=anchors, 
                num_classes=cfg.num_classes,
                use_ciou=getattr(cfg, 'use_ciou', True),
                use_focal=getattr(cfg, 'use_focal', True),
                use_label_smoothing=getattr(cfg, 'use_label_smoothing', False)
            )
            
            print(f"   ✅ MultiScaleYOLO: {sum(p.numel() for p in model.parameters()):,} parameters")
            print(f"   ✅ AdvancedMultiScaleLoss: 3スケール対応")
            
            return model, criterion, "multiscale"
            
        except Exception as e:
            print(f"⚠️ マルチスケール初期化失敗: {e}")
            print("📚 フォールバックモードに切り替えます")
    
    # フォールバック: 従来アーキテクチャ
    print("📚 フォールバック: 従来アーキテクチャを使用")
    model = SimpleYOLO(cfg.num_classes, use_phase2_enhancements=False) 
    criterion = YOLOLoss(cfg.num_classes)
    
    print(f"   ✅ SimpleYOLO: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   ✅ YOLOLoss: 単一スケール")
    
    return model, criterion, "fallback"

# ===== Phase 3: データ分割関数 =====
def create_train_val_split(dataset, val_split=0.15):
    """訓練・検証データの分割"""
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 再現性のため
    )
    
    print(f"📊 データ分割完了:")
    print(f"   Train: {train_size} images ({100*(1-val_split):.1f}%)")
    print(f"   Validation: {val_size} images ({100*val_split:.1f}%)")
    return train_dataset, val_dataset

# ===== Phase 3: 検証関数（マルチスケール対応） =====
def validate_model(model, val_dataloader, criterion, device, architecture_type):
    """検証実行（アーキテクチャ自動判定）"""
    model.eval()
    total_val_loss = 0
    val_batches = 0
    
    with torch.no_grad():
        for images, targets in val_dataloader:
            images = images.to(device, non_blocking=True)
            
            if architecture_type == "multiscale":
                # マルチスケール: 辞書形式の出力
                predictions = model(images)
                loss = criterion(predictions, targets)
            else:
                # 従来: タプル形式の出力
                predictions, grid_size = model(images)
                loss = criterion(predictions, targets, grid_size)
            
            total_val_loss += loss.item()
            val_batches += 1
    
    avg_val_loss = total_val_loss / val_batches if val_batches > 0 else float('inf')
    return avg_val_loss

def validate_model_with_postprocessing(model, val_dataloader, criterion, device, architecture_type, use_advanced_postprocessing=True):
    """後処理を含む詳細検証"""
    model.eval()
    total_val_loss = 0
    val_batches = 0
    
    # 後処理システム初期化
    if use_advanced_postprocessing:
        post_processor = AdvancedPostProcessor(
            use_soft_nms=True,
            use_tta=False,      # 検証時は時間効率重視
            use_multiscale=False,
            conf_threshold=0.25,  # 改良設定
            iou_threshold=0.5
        )
        
        detection_stats = {
            'total_detections': 0,
            'high_conf_detections': 0,  # conf > 0.7
            'processed_detections': 0
        }
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_dataloader):
            images = images.to(device, non_blocking=True)
            
            # 通常の損失計算
            if architecture_type == "multiscale":
                predictions = model(images)
                loss = criterion(predictions, targets)
            else:
                predictions, grid_size = model(images)
                loss = criterion(predictions, targets, grid_size)
            
            total_val_loss += loss.item()
            val_batches += 1
            
            # 後処理テスト（サンプリング検証）
            if use_advanced_postprocessing and batch_idx % 10 == 0:  # 10バッチに1回
                try:
                    # 1枚目の画像で後処理テスト
                    single_image = images[0:1]
                    single_pred = {k: v[0:1] for k, v in predictions.items()} if isinstance(predictions, dict) else predictions[0:1]
                    
                    # 後処理実行
                    processed_detections = post_processor.process_predictions(
                        model, single_image, single_pred
                    )
                    
                    # 統計更新
                    detection_stats['total_detections'] += len(processed_detections)
                    high_conf = sum(1 for det in processed_detections if det['score'] > 0.7)
                    detection_stats['high_conf_detections'] += high_conf
                    detection_stats['processed_detections'] += 1
                    
                except Exception as e:
                    pass  # 後処理エラーは無視
    
    avg_val_loss = total_val_loss / val_batches if val_batches > 0 else float('inf')
    
    # 後処理統計を表示
    if use_advanced_postprocessing and detection_stats['processed_detections'] > 0:
        avg_detections = detection_stats['total_detections'] / detection_stats['processed_detections']
        avg_high_conf = detection_stats['high_conf_detections'] / detection_stats['processed_detections']
        
        print(f"📊 後処理統計 (サンプル{detection_stats['processed_detections']}枚):")
        print(f"   平均検出数: {avg_detections:.1f}/画像")
        print(f"   高信頼度検出: {avg_high_conf:.1f}/画像 (conf>0.7)")
    
    return avg_val_loss

# ===== データローダーセットアップ（修正版） =====
def setup_dataloaders(cfg):
    """データローダーセットアップ（改良版対応）"""
    
    if USE_IMPROVED:
        # 改良版データセット使用
        print("🎨 改良版データセット使用中...")
        full_dataset = ImprovedFLIRDataset(
            cfg.train_img_dir, 
            cfg.train_label_dir, 
            cfg.img_size,
            use_improved_augment=True,
            mixup_in_dataset=False  # DataLoaderレベルで処理
        )
        
        # 検証分割
        if cfg.validation_split > 0:
            train_dataset, val_dataset = create_train_val_split(full_dataset, cfg.validation_split)
        else:
            train_dataset = full_dataset
            val_dataset = None
        
        # 改良版DataLoader
        train_dataloader = create_improved_dataloader(
            train_dataset,
            batch_size=cfg.batch_size,
            use_mixup=getattr(cfg, 'use_mixup', True),
            shuffle=True,
            num_workers=2,  # ★★★ 修正: 4 → 2 (警告対策) ★★★
            pin_memory=getattr(cfg, 'pin_memory', True),
            persistent_workers=False,  # ★★★ 修正: True → False (安定性向上) ★★★
            prefetch_factor=2  # ★★★ 修正: デフォルト値明示 ★★★
        )
        
        val_dataloader = None
        if val_dataset:
            val_dataloader = create_improved_dataloader(
                val_dataset,
                batch_size=cfg.batch_size,
                use_mixup=False,  # 検証時はMixUpなし
                shuffle=False,
                num_workers=2,  # ★★★ 修正: 4 → 2 ★★★
                pin_memory=getattr(cfg, 'pin_memory', True),
                persistent_workers=False  # ★★★ 修正: 安定性向上 ★★★
            )
    
    else:
        # 標準データセット使用
        print("📚 標準データセット使用中...")
        full_dataset = FLIRDataset(
            cfg.train_img_dir, 
            cfg.train_label_dir, 
            cfg.img_size, 
            augment=getattr(cfg, 'augment', True)
        )
        
        # 検証分割
        if cfg.validation_split > 0:
            train_dataset, val_dataset = create_train_val_split(full_dataset, cfg.validation_split)
        else:
            train_dataset = full_dataset
            val_dataset = None
        
        # DataLoader作成
        num_workers = 2  # ★★★ 修正: 4 → 2 (警告対策) ★★★
        pin_memory = getattr(cfg, 'pin_memory', True)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False  # ★★★ 修正: 安定性向上 ★★★
        )
        
        val_dataloader = None
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=False  # ★★★ 修正: 安定性向上 ★★★
            )
    
    print(f"   Train batches: {len(train_dataloader)}")
    if val_dataloader:
        print(f"   Validation batches: {len(val_dataloader)}")
    
    return train_dataloader, val_dataloader

# ===== Phase 3: マルチスケール対応学習ループ（診断統合版） =====
def phase3_integrated_training_loop(model, train_dataloader, val_dataloader, criterion, cfg, architecture_type):
    """Phase 3統合学習ループ（診断機能完全統合）"""
    
    # ★★★ 診断機能初期化 ★★★
    diagnostics = DiagnosticTrainer(
        save_dir=os.path.join(cfg.save_dir, "diagnostics")
    )
    
    # オプティマイザ設定
    if getattr(cfg, 'optimizer_type', 'AdamW') == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=getattr(cfg, 'weight_decay', 2e-4),
            betas=getattr(cfg, 'betas', (0.9, 0.999)),
            eps=getattr(cfg, 'eps', 1e-8)
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    # EMA初期化
    ema = None
    if getattr(cfg, 'use_ema', True):
        ema = EMAModel(model, decay=getattr(cfg, 'ema_decay', 0.9995))
        print(f"🔄 EMA initialized with decay {cfg.ema_decay}")
    
    # スケジューラ設定
    scheduler = None
    if getattr(cfg, 'use_scheduler', True):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=cfg.num_epochs,
            eta_min=getattr(cfg, 'min_lr', cfg.learning_rate / 250)
        )
    
    # Early Stopping設定
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 学習統計
    train_losses = []
    val_losses = []
    learning_rates = []
    
    print(f"🚀 Phase 3統合学習開始（診断機能付き）")
    print(f"   Architecture: {architecture_type}")
    print(f"   Optimizer: {getattr(cfg, 'optimizer_type', 'AdamW')}")
    print(f"   EMA: {'ON' if getattr(cfg, 'use_ema', True) else 'OFF'}")
    print(f"   Validation: {'ON' if val_dataloader else 'OFF'}")
    print(f"   Diagnostics: {diagnostics.save_dir}")
    
    for epoch in range(cfg.num_epochs):
        epoch_start = time.time()
        
        # ★★★ エポック診断開始 ★★★
        diagnostics.start_epoch_diagnosis(epoch + 1)
        
        # ===== ウォームアップ処理 =====
        warmup_epochs = getattr(cfg, 'warmup_epochs', 0)
        if epoch < warmup_epochs:
            warmup_lr = cfg.learning_rate * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"🔥 Warmup Epoch {epoch+1}: LR = {warmup_lr:.6f}")
        
        # ===== 訓練フェーズ =====
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        # 進捗トラッカー初期化（利用可能な場合）
        try:
            from progress import MultiScaleProgressTracker
            progress_tracker = MultiScaleProgressTracker(len(train_dataloader), print_interval=getattr(cfg, 'print_interval', 5))
            progress_tracker.start_epoch(epoch + 1, cfg.num_epochs)
            use_progress_tracker = True
        except ImportError:
            use_progress_tracker = False
            print_interval = getattr(cfg, 'print_interval', 5)
        
        for batch_idx, (images, targets) in enumerate(train_dataloader):
            images = images.to(cfg.device, non_blocking=True)
            
            # Forward（アーキテクチャ別）
            if architecture_type == "multiscale":
                predictions = model(images)
                loss = criterion(predictions, targets)
                
                # ★★★ 診断情報取得 ★★★
                loss_components = None
                try:
                    # 詳細情報付きで再計算（診断用）
                    if hasattr(criterion, 'return_components') and batch_idx % 20 == 0:
                        criterion.return_components = True
                        _, loss_components = criterion(predictions, targets)
                        criterion.return_components = False
                except:
                    pass  # エラー時は無視
                
            else:
                predictions, grid_size = model(images)
                loss = criterion(predictions, targets, grid_size)
                loss_components = None
            
            # ★★★ バッチ診断実行 ★★★
            try:
                diagnostics.log_batch_diagnosis(
                    batch_idx, images, targets, predictions, loss_components
                )
            except Exception as e:
                if batch_idx % 50 == 0:  # エラーログを減らす
                    print(f"   ⚠️ 診断エラー (batch {batch_idx}): {e}")
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # 勾配クリッピング
            if hasattr(cfg, 'gradient_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            
            optimizer.step()
            
            # EMA更新
            if ema:
                ema.update()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # 進捗表示
            current_lr = optimizer.param_groups[0]['lr']
            if use_progress_tracker:
                # 詳細進捗表示
                if architecture_type == "multiscale" and loss_components:
                    progress_tracker.update_batch_multiscale(
                        batch_idx, loss.item(), current_lr, loss_components
                    )
                else:
                    progress_tracker.update_batch(batch_idx, loss.item(), current_lr)
            else:
                # 簡易進捗表示
                if batch_idx % print_interval == 0:
                    print(f"   Batch {batch_idx:4d}/{len(train_dataloader)}: Loss={loss.item():.4f}, LR={current_lr:.6f}")
        
        avg_train_loss = epoch_loss / batch_count
        
        # ===== 検証フェーズ =====
        val_loss = float('inf')
        if val_dataloader and epoch % getattr(cfg, 'validate_every', 1) == 0:
            # EMAモデルで検証
            if ema:
                ema.apply_shadow()
            
            # 詳細検証（5エポックに1回）
            if epoch % 5 == 0 and getattr(cfg, 'use_advanced_postprocessing', True):
                print(f"🔧 後処理込み詳細検証実行中...")
                val_loss = validate_model_with_postprocessing(
                    model, val_dataloader, criterion, cfg.device, architecture_type
                )
            else:
                # 通常検証（軽量）
                val_loss = validate_model(
                    model, val_dataloader, criterion, cfg.device, architecture_type
                )
            
            if ema:
                ema.restore()
        
        # スケジューラ更新
        if scheduler and epoch >= warmup_epochs:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # 統計記録
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        learning_rates.append(current_lr)
        
        # ★★★ エポック診断終了 & 改善提案 ★★★
        suggestions = diagnostics.end_epoch_diagnosis(epoch + 1, val_loss, model)
        
        # ログ表示
        val_str = f"Val: {val_loss:6.4f}" if val_loss != float('inf') else "Val: ----"
        print(f"\n📈 Epoch [{epoch+1:2d}/{cfg.num_epochs}] "
              f"Train: {avg_train_loss:6.4f} {val_str} "
              f"Time: {epoch_time:4.1f}s LR: {current_lr:.6f}")
        
        # 改善提案表示
        if suggestions:
            print(f"💡 診断提案:")
            for suggestion in suggestions[:2]:  # 上位2件のみ
                print(f"   {suggestion['type']}: {suggestion['suggestion']}")
        
        # GPU使用量表示（最初の数エポック）
        if cfg.device.type == 'cuda' and epoch < 3:
            memory_used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"   GPU Memory: {memory_used:.2f}GB")
        
        # Early Stopping & Best Model Saving
        current_loss = val_loss if val_loss != float('inf') else avg_train_loss
        min_improvement = getattr(cfg, 'min_improvement', 0.005)
        
        if current_loss < best_val_loss - min_improvement:
            best_val_loss = current_loss
            patience_counter = 0
            
            # EMAモデルを保存
            if ema:
                ema.apply_shadow()
            
            save_best_model_integrated(model, optimizer, ema, epoch, best_val_loss, cfg, architecture_type)
            
            if ema:
                ema.restore()
            
            print(f"🎉 New best loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            patience = getattr(cfg, 'patience', 10)
            print(f"⏳ No improvement for {patience_counter}/{patience} epochs")
            
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered")
                break
        
        # 定期保存
        save_interval = getattr(cfg, 'save_interval', 2)
        if (epoch + 1) % save_interval == 0:
            save_checkpoint_integrated(model, optimizer, epoch, avg_train_loss, cfg, architecture_type)
        
        # メモリクリーンアップ
        if cfg.device.type == 'cuda':
            empty_cache_every = getattr(cfg, 'empty_cache_every_n_batch', 50)
            if (epoch + 1) % (empty_cache_every // 10) == 0:  # エポックベースで調整
                torch.cuda.empty_cache()
    
    # ★★★ 最終診断レポート ★★★
    diagnostics.generate_final_report()
    
    print(f"\n✅ Phase 3統合学習完了!")
    print(f"🏆 Best Loss: {best_val_loss:.4f}")
    
    return train_losses, val_losses, learning_rates, best_val_loss

# ===== 統合版保存関数 =====
def save_best_model_integrated(model, optimizer, ema, epoch, loss, cfg, architecture_type):
    """統合版ベストモデル保存"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'architecture_type': architecture_type,
        'ema_state_dict': ema.shadow if ema else None,
        'config': {
            'batch_size': cfg.batch_size,
            'learning_rate': cfg.learning_rate,
            'ema_decay': getattr(cfg, 'ema_decay', None),
            'validation_split': getattr(cfg, 'validation_split', 0.15),
            'use_multiscale': getattr(cfg, 'use_multiscale_architecture', True)
        },
        'training_stats': {
            'gpu_memory_peak': torch.cuda.max_memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0,
            'parameters': sum(p.numel() for p in model.parameters())
        },
        'version_info': VersionTracker.get_all_trackers()
    }
    
    save_path = os.path.join(cfg.save_dir, f'phase3_integrated_{architecture_type}_loss_{loss:.4f}.pth')
    torch.save(checkpoint, save_path)
    print(f"💾 Phase 3 integrated model saved: {save_path}")

def save_checkpoint_integrated(model, optimizer, epoch, loss, cfg, architecture_type):
    """統合版チェックポイント保存"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'architecture_type': architecture_type,
        'config_info': {
            'batch_size': cfg.batch_size,
            'learning_rate': cfg.learning_rate,
            'optimizer_type': getattr(cfg, 'optimizer_type', 'AdamW')
        },
        'version_info': VersionTracker.get_all_trackers()
    }
    
    save_path = os.path.join(cfg.save_dir, f'checkpoint_integrated_epoch_{epoch+1}.pth')
    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint saved: {save_path}")

def plot_training_progress(losses, lrs, val_losses=None, save_path="training_progress_integrated.png"):
    """学習進捗を可視化（統合版）"""
    try:
        import matplotlib.pyplot as plt
        
        if val_losses:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Training Loss plot
        ax1.plot(losses, 'b-', linewidth=2, label='Train Loss')
        if val_losses:
            valid_val_losses = [loss for loss in val_losses if loss != float('inf')]
            valid_epochs = [i for i, loss in enumerate(val_losses) if loss != float('inf')]
            if valid_val_losses:
                ax1.plot(valid_epochs, valid_val_losses, 'r-', linewidth=2, label='Val Loss')
        
        ax1.set_title('Phase 3 Integrated Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        if val_losses:
            ax1.legend()
        
        # Learning Rate plot
        ax2.plot(lrs, 'r-', linewidth=2)
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Validation Loss separate plot
        if val_losses:
            valid_val_losses = [loss for loss in val_losses if loss != float('inf')]
            valid_epochs = [i for i, loss in enumerate(val_losses) if loss != float('inf')]
            if valid_val_losses:
                ax3.plot(valid_epochs, valid_val_losses, 'g-', linewidth=2)
                ax3.set_title('Validation Loss')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Val Loss')
                ax3.grid(True, alpha=0.3)
                ax3.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()  # メモリ節約
        
        print(f"📊 Training progress saved to {save_path}")
    except ImportError:
        print("⚠️ matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"⚠️ 可視化エラー: {e}")

# ===== GPU環境チェック =====
def comprehensive_gpu_check():
    """包括的なGPU環境チェック"""
    print("\n🔍 GPU環境詳細チェック")
    print("="*60)
    
    print(f"1. CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name(0)}")
        
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   GPU memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
    else:
        print("   ❌ CUDA not available!")
        return False
    
    print(f"2. PyTorch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"3. Selected device: {device}")
    
    return True

# ===== メイン関数 =====
def main():
    print("🚀 Starting Phase 3 Integrated YOLO Training (Diagnostic Enhanced)")
    
    # プロジェクト全体のバージョン情報を表示
    print("\n" + "="*80)
    print("📋 プロジェクト全体バージョン確認")
    print("="*80)
    VersionTracker.print_all_versions()

    # 設定とGPU確認
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # 診断ディレクトリ作成
    diagnostic_dir = os.path.join(cfg.save_dir, "diagnostics")
    os.makedirs(diagnostic_dir, exist_ok=True)
    
    # GPU環境詳細チェック
    if not comprehensive_gpu_check():
        print("❌ GPU使用不可 - CPU学習に切り替えます")
        cfg.device = torch.device('cpu')
        cfg.batch_size = max(cfg.batch_size // 4, 1)
        print(f"   CPU用にバッチサイズを {cfg.batch_size} に調整")
    
    print(f"\n📋 学習設定:")
    print(f"   Device: {cfg.device}")
    print(f"   Batch size: {cfg.batch_size}")
    print(f"   Image size: {cfg.img_size}")
    print(f"   Classes: {cfg.num_classes}")
    print(f"   Learning rate: {cfg.learning_rate:.0e}")
    print(f"   EMA: {getattr(cfg, 'use_ema', True)}")
    print(f"   Validation Split: {getattr(cfg, 'validation_split', 0.15)}")
    print(f"   Diagnostics: ON")
    
    # Phase 3統合: データセット & 検証分割
    print("\n📊 Loading dataset with validation split...")
    try:
        train_dataloader, val_dataloader = setup_dataloaders(cfg)
        
        print(f"   Total train batches: {len(train_dataloader)}")
        if val_dataloader:
            print(f"   Total validation batches: {len(val_dataloader)}")
    except Exception as e:
        print(f"❌ データローダー作成エラー: {e}")
        print("   標準データセットでリトライ中...")
        
        # フォールバック
        global USE_IMPROVED
        USE_IMPROVED = False
        train_dataloader, val_dataloader = setup_dataloaders(cfg)
    
    # Phase 3統合: モデル&損失関数（アーキテクチャ自動選択）
    print("\n🤖 Creating Phase 3 integrated model...")
    model, criterion, architecture_type = create_model_and_loss(cfg)
    
    # GPUに移動
    print(f"   Moving model to {cfg.device}...")
    model = model.to(cfg.device)
    
    if cfg.device.type == 'cuda':
        model = model.float()
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    model_device = next(model.parameters()).device
    print(f"   Model device confirmed: {model_device}")
    
    # Phase 3統合学習実行
    print("\n🚀 Phase 3統合学習を開始（診断機能付き）")
    print(f"🎯 目標: Val Loss 43.45 → 25-30")
    
    try:
        train_losses, val_losses, lrs, best_loss = phase3_integrated_training_loop(
            model, train_dataloader, val_dataloader, criterion, cfg, architecture_type
        )
        
        # 結果可視化
        try:
            plot_training_progress(train_losses, lrs, val_losses)
        except Exception as e:
            print(f"⚠️ 可視化エラー: {e}")
        
        print(f"\n🎯 Phase 3統合学習完了!")
        print(f"🏆 Best Loss: {best_loss:.4f}")
        print(f"🔧 Architecture: {architecture_type}")
        
        # 目標達成判定
        print(f"\n📊 目標達成判定:")
        if best_loss < 30.0:
            print("🎉 Step 1目標達成! (Val Loss < 30.0)")
            if best_loss < 20.0:
                print("🏆 予想を上回る成果! (Val Loss < 20.0)")
        elif best_loss < 35.0:
            print("🟡 部分的改善 (Val Loss < 35.0) - 継続推奨")
        else:
            print("🔴 目標未達成 - 方針転換検討")
            print("   推奨: 学習率をさらに2倍、またはアンカー見直し")
        
        # 具体的な改善提案
        improvement_ratio = 43.45 / best_loss if best_loss > 0 else 1
        print(f"\n💡 改善結果:")
        print(f"   改善率: {improvement_ratio:.1f}x (43.45 → {best_loss:.4f})")
        print(f"   次目標: Val Loss < {best_loss * 0.7:.1f}")
        
        if best_loss > 35.0:
            print(f"\n🚨 緊急改善案:")
            print(f"   1. 学習率を2倍に ({cfg.learning_rate:.0e} → {cfg.learning_rate*2:.0e})")
            print(f"   2. バッチサイズを半分に ({cfg.batch_size} → {cfg.batch_size//2})")
            print(f"   3. アンカーサイズの全面見直し")
        
    except Exception as e:
        print(f"❌ 学習エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ Training completed!")
    
    # 最終統計
    if cfg.device.type == 'cuda':
        final_memory = torch.cuda.memory_allocated(0) / 1024**3
        max_memory = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"📊 最終GPU統計:")
        print(f"   現在のメモリ使用量: {final_memory:.2f}GB")
        print(f"   最大メモリ使用量: {max_memory:.2f}GB")
    
    # 学習結果サマリー
    print(f"\n🎯 学習結果サマリー:")
    print(f"   Architecture: {architecture_type}")
    print(f"   最終Loss: {best_loss:.4f}")
    print(f"   改良拡張: {'ON' if USE_IMPROVED else 'OFF'}")
    print(f"   診断機能: ON")
    print(f"   EMA使用: {'Yes' if getattr(cfg, 'use_ema', True) else 'No'}")
    print(f"   検証分割: {'Yes' if getattr(cfg, 'validation_split', 0.15) > 0 else 'No'}")
    print(f"   保存先: {cfg.save_dir}")
    print(f"   診断ログ: {diagnostic_dir}")

if __name__ == "__main__":
    main()# train_phase3_integrated.py - マルチスケールYOLO統合版
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import os

from config import Config

# ★★★ データセット関連修正 ★★★
try:
    # 改良版を優先
    from improved_augmentation import ImprovedFLIRDataset, create_improved_dataloader
    USE_IMPROVED = True
    print("✅ 改良版データセットを使用")
except ImportError:
    # フォールバック
    from dataset import FLIRDataset, collate_fn
    USE_IMPROVED = False
    print("📚 標準データセットを使用")

# ★★★ Phase 3 新アーキテクチャをインポート ★★★
from multiscale_model import MultiScaleYOLO
from advanced_losses import AdvancedMultiScaleLoss
from post_processing import AdvancedPostProcessor, SoftNMS
from diagnostic_training import DiagnosticTrainer

# ★★★ フォールバック用（従来アーキテクチャ） ★★★
from model import SimpleYOLO
from loss import YOLOLoss

# ★★★ 共有VersionTrackerをインポート ★★★
from version_tracker import (
    create_version_tracker, 
    VersionTracker, 
    show_all_project_versions,
    debug_version_status,
    get_version_count
)

# バージョン管理システム初期化
training_version = create_version_tracker("Training System v3.0 - Diagnostic Integrated", "train.py")
training_version.add_modification("診断機能完全統合")
training_version.add_modification("改良版データセット対応")
training_version.add_modification("エラーハンドリング強化")

# ===== Phase 3: EMAクラス実装 =====
class EMAModel:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# ===== アーキテクチャ選択関数 =====
def create_model_and_loss(cfg):
    """Phase 3アーキテクチャ or フォールバック選択"""
    
    if getattr(cfg, 'use_multiscale_architecture', True):
        print("🚀 Phase 3: マルチスケールアーキテクチャを使用")
        try:
            # Step 1のアンカー（実際にはStep 1で生成されたものを使用）
            anchors = {
                'small':  [(7, 11), (14, 28), (22, 65)],      # 52x52 grid
                'medium': [(42, 35), (76, 67), (46, 126)],    # 26x26 grid  
                'large':  [(127, 117), (88, 235), (231, 218)] # 13x13 grid
            }
            
            model = MultiScaleYOLO(num_classes=cfg.num_classes, anchors=anchors)
            criterion = AdvancedMultiScaleLoss(
                anchors=anchors, 
                num_classes=cfg.num_classes,
                use_ciou=getattr(cfg, 'use_ciou', True),
                use_focal=getattr(cfg, 'use_focal', True),
                use_label_smoothing=getattr(cfg, 'use_label_smoothing', False)
            )
            
            print(f"   ✅ MultiScaleYOLO: {sum(p.numel() for p in model.parameters()):,} parameters")
            print(f"   ✅ AdvancedMultiScaleLoss: 3スケール対応")
            
            return model, criterion, "multiscale"
            
        except Exception as e:
            print(f"⚠️ マルチスケール初期化失敗: {e}")
            print("📚 フォールバックモードに切り替えます")
    
    # フォールバック: 従来アーキテクチャ
    print("📚 フォールバック: 従来アーキテクチャを使用")
    model = SimpleYOLO(cfg.num_classes, use_phase2_enhancements=False)
    criterion = YOLOLoss(cfg.num_classes)
    
    print(f"   ✅ SimpleYOLO: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   ✅ YOLOLoss: 単一スケール")
    
    return model, criterion, "fallback"

# ===== Phase 3: データ分割関数 =====
def create_train_val_split(dataset, val_split=0.15):
    """訓練・検証データの分割"""
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 再現性のため
    )
    
    print(f"📊 データ分割完了:")
    print(f"   Train: {train_size} images ({100*(1-val_split):.1f}%)")
    print(f"   Validation: {val_size} images ({100*val_split:.1f}%)")
    return train_dataset, val_dataset

# ===== Phase 3: 検証関数（マルチスケール対応） =====
def validate_model(model, val_dataloader, criterion, device, architecture_type):
    """検証実行（アーキテクチャ自動判定）"""
    model.eval()
    total_val_loss = 0
    val_batches = 0
    
    with torch.no_grad():
        for images, targets in val_dataloader:
            images = images.to(device, non_blocking=True)
            
            if architecture_type == "multiscale":
                # マルチスケール: 辞書形式の出力
                predictions = model(images)
                loss = criterion(predictions, targets)
            else:
                # 従来: タプル形式の出力
                predictions, grid_size = model(images)
                loss = criterion(predictions, targets, grid_size)
            
            total_val_loss += loss.item()
            val_batches += 1
    
    avg_val_loss = total_val_loss / val_batches if val_batches > 0 else float('inf')
    return avg_val_loss

def validate_model_with_postprocessing(model, val_dataloader, criterion, device, architecture_type, use_advanced_postprocessing=True):
    """後処理を含む詳細検証"""
    model.eval()
    total_val_loss = 0
    val_batches = 0
    
    # 後処理システム初期化
    if use_advanced_postprocessing:
        post_processor = AdvancedPostProcessor(
            use_soft_nms=True,
            use_tta=False,      # 検証時は時間効率重視
            use_multiscale=False,
            conf_threshold=0.25,  # 改良設定
            iou_threshold=0.5
        )
        
        detection_stats = {
            'total_detections': 0,
            'high_conf_detections': 0,  # conf > 0.7
            'processed_detections': 0
        }
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_dataloader):
            images = images.to(device, non_blocking=True)
            
            # 通常の損失計算
            if architecture_type == "multiscale":
                predictions = model(images)
                loss = criterion(predictions, targets)
            else:
                predictions, grid_size = model(images)
                loss = criterion(predictions, targets, grid_size)
            
            total_val_loss += loss.item()
            val_batches += 1
            
            # 後処理テスト（サンプリング検証）
            if use_advanced_postprocessing and batch_idx % 10 == 0:  # 10バッチに1回
                try:
                    # 1枚目の画像で後処理テスト
                    single_image = images[0:1]
                    single_pred = {k: v[0:1] for k, v in predictions.items()} if isinstance(predictions, dict) else predictions[0:1]
                    
                    # 後処理実行
                    processed_detections = post_processor.process_predictions(
                        model, single_image, single_pred
                    )
                    
                    # 統計更新
                    detection_stats['total_detections'] += len(processed_detections)
                    high_conf = sum(1 for det in processed_detections if det['score'] > 0.7)
                    detection_stats['high_conf_detections'] += high_conf
                    detection_stats['processed_detections'] += 1
                    
                except Exception as e:
                    pass  # 後処理エラーは無視
    
    avg_val_loss = total_val_loss / val_batches if val_batches > 0 else float('inf')
    
    # 後処理統計を表示
    if use_advanced_postprocessing and detection_stats['processed_detections'] > 0:
        avg_detections = detection_stats['total_detections'] / detection_stats['processed_detections']
        avg_high_conf = detection_stats['high_conf_detections'] / detection_stats['processed_detections']
        
        print(f"📊 後処理統計 (サンプル{detection_stats['processed_detections']}枚):")
        print(f"   平均検出数: {avg_detections:.1f}/画像")
        print(f"   高信頼度検出: {avg_high_conf:.1f}/画像 (conf>0.7)")
    
    return avg_val_loss

# ===== データローダーセットアップ（修正版） =====
def setup_dataloaders(cfg):
    """データローダーセットアップ（改良版対応）"""
    
    if USE_IMPROVED:
        # 改良版データセット使用
        print("🎨 改良版データセット使用中...")
        full_dataset = ImprovedFLIRDataset(
            cfg.train_img_dir, 
            cfg.train_label_dir, 
            cfg.img_size,
            use_improved_augment=True,
            mixup_in_dataset=False  # DataLoaderレベルで処理
        )
        
        # 検証分割
        if cfg.validation_split > 0:
            train_dataset, val_dataset = create_train_val_split(full_dataset, cfg.validation_split)
        else:
            train_dataset = full_dataset
            val_dataset = None
        
        # 改良版DataLoader
        train_dataloader = create_improved_dataloader(
            train_dataset,
            batch_size=cfg.batch_size,
            use_mixup=getattr(cfg, 'use_mixup', True),
            shuffle=True,
            num_workers=2,  # ★★★ 修正: 4 → 2 (警告対策) ★★★
            pin_memory=getattr(cfg, 'pin_memory', True),
            persistent_workers=False,  # ★★★ 修正: True → False (安定性向上) ★★★
            prefetch_factor=2  # ★★★ 修正: デフォルト値明示 ★★★
        )
        
        val_dataloader = None
        if val_dataset:
            val_dataloader = create_improved_dataloader(
                val_dataset,
                batch_size=cfg.batch_size,
                use_mixup=False,  # 検証時はMixUpなし
                shuffle=False,
                num_workers=2,  # ★★★ 修正: 4 → 2 ★★★
                pin_memory=getattr(cfg, 'pin_memory', True),
                persistent_workers=False  # ★★★ 修正: 安定性向上 ★★★
            )
    
    else:
        # 標準データセット使用
        print("📚 標準データセット使用中...")
        full_dataset = FLIRDataset(
            cfg.train_img_dir, 
            cfg.train_label_dir, 
            cfg.img_size, 
            augment=getattr(cfg, 'augment', True)
        )
        
        # 検証分割
        if cfg.validation_split > 0:
            train_dataset, val_dataset = create_train_val_split(full_dataset, cfg.validation_split)
        else:
            train_dataset = full_dataset
            val_dataset = None
        
        # DataLoader作成
        num_workers = 2  # ★★★ 修正: 4 → 2 (警告対策) ★★★
        pin_memory = getattr(cfg, 'pin_memory', True)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False  # ★★★ 修正: 安定性向上 ★★★
        )
        
        val_dataloader = None
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=False  # ★★★ 修正: 安定性向上 ★★★
            )
    
    print(f"   Train batches: {len(train_dataloader)}")
    if val_dataloader:
        print(f"   Validation batches: {len(val_dataloader)}")
    
    return train_dataloader, val_dataloader

# ===== Phase 3: マルチスケール対応学習ループ（診断統合版） =====
def phase3_integrated_training_loop(model, train_dataloader, val_dataloader, criterion, cfg, architecture_type):
    """Phase 3統合学習ループ（診断機能完全統合）"""
    
    # ★★★ 診断機能初期化 ★★★
    diagnostics = DiagnosticTrainer(
        save_dir=os.path.join(cfg.save_dir, "diagnostics")
    )
    
    # オプティマイザ設定
    if getattr(cfg, 'optimizer_type', 'AdamW') == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=getattr(cfg, 'weight_decay', 2e-4),
            betas=getattr(cfg, 'betas', (0.9, 0.999)),
            eps=getattr(cfg, 'eps', 1e-8)
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    # EMA初期化
    ema = None
    if getattr(cfg, 'use_ema', True):
        ema = EMAModel(model, decay=getattr(cfg, 'ema_decay', 0.9995))
        print(f"🔄 EMA initialized with decay {cfg.ema_decay}")
    
    # スケジューラ設定
    scheduler = None
    if getattr(cfg, 'use_scheduler', True):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=cfg.num_epochs,
            eta_min=getattr(cfg, 'min_lr', cfg.learning_rate / 250)
        )
    
    # Early Stopping設定
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 学習統計
    train_losses = []
    val_losses = []
    learning_rates = []
    
    print(f"🚀 Phase 3統合学習開始（診断機能付き）")
    print(f"   Architecture: {architecture_type}")
    print(f"   Optimizer: {getattr(cfg, 'optimizer_type', 'AdamW')}")
    print(f"   EMA: {'ON' if getattr(cfg, 'use_ema', True) else 'OFF'}")
    print(f"   Validation: {'ON' if val_dataloader else 'OFF'}")
    print(f"   Diagnostics: {diagnostics.save_dir}")
    
    for epoch in range(cfg.num_epochs):
        epoch_start = time.time()
        
        # ★★★ エポック診断開始 ★★★
        diagnostics.start_epoch_diagnosis(epoch + 1)
        
        # ===== ウォームアップ処理 =====
        warmup_epochs = getattr(cfg, 'warmup_epochs', 0)
        if epoch < warmup_epochs:
            warmup_lr = cfg.learning_rate * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"🔥 Warmup Epoch {epoch+1}: LR = {warmup_lr:.6f}")
        
        # ===== 訓練フェーズ =====
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        # 進捗トラッカー初期化（利用可能な場合）
        try:
            from progress import MultiScaleProgressTracker
            progress_tracker = MultiScaleProgressTracker(len(train_dataloader), print_interval=getattr(cfg, 'print_interval', 5))
            progress_tracker.start_epoch(epoch + 1, cfg.num_epochs)
            use_progress_tracker = True
        except ImportError:
            use_progress_tracker = False
            print_interval = getattr(cfg, 'print_interval', 5)
        
        for batch_idx, (images, targets) in enumerate(train_dataloader):
            images = images.to(cfg.device, non_blocking=True)
            
            # Forward（アーキテクチャ別）
            if architecture_type == "multiscale":
                predictions = model(images)
                loss = criterion(predictions, targets)
                
                # ★★★ 診断情報取得 ★★★
                loss_components = None
                try:
                    # 詳細情報付きで再計算（診断用）
                    if hasattr(criterion, 'return_components') and batch_idx % 20 == 0:
                        criterion.return_components = True
                        _, loss_components = criterion(predictions, targets)
                        criterion.return_components = False
                except:
                    pass  # エラー時は無視
                
            else:
                predictions, grid_size = model(images)
                loss = criterion(predictions, targets, grid_size)
                loss_components = None
            
            # ★★★ バッチ診断実行 ★★★
            try:
                diagnostics.log_batch_diagnosis(
                    batch_idx, images, targets, predictions, loss_components
                )
            except Exception as e:
                if batch_idx % 50 == 0:  # エラーログを減らす
                    print(f"   ⚠️ 診断エラー (batch {batch_idx}): {e}")
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # 勾配クリッピング
            if hasattr(cfg, 'gradient_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            
            optimizer.step()
            
            # EMA更新
            if ema:
                ema.update()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # 進捗表示
            current_lr = optimizer.param_groups[0]['lr']
            if use_progress_tracker:
                # 詳細進捗表示
                if architecture_type == "multiscale" and loss_components:
                    progress_tracker.update_batch_multiscale(
                        batch_idx, loss.item(), current_lr, loss_components
                    )
                else:
                    progress_tracker.update_batch(batch_idx, loss.item(), current_lr)
            else:
                # 簡易進捗表示
                if batch_idx % print_interval == 0:
                    print(f"   Batch {batch_idx:4d}/{len(train_dataloader)}: Loss={loss.item():.4f}, LR={current_lr:.6f}")
        
        avg_train_loss = epoch_loss / batch_count
        
        # ===== 検証フェーズ =====
        val_loss = float('inf')
        if val_dataloader and epoch % getattr(cfg, 'validate_every', 1) == 0:
            # EMAモデルで検証
            if ema:
                ema.apply_shadow()
            
            # 詳細検証（5エポックに1回）
            if epoch % 5 == 0 and getattr(cfg, 'use_advanced_postprocessing', True):
                print(f"🔧 後処理込み詳細検証実行中...")
                val_loss = validate_model_with_postprocessing(
                    model, val_dataloader, criterion, cfg.device, architecture_type
                )
            else:
                # 通常検証（軽量）
                val_loss = validate_model(
                    model, val_dataloader, criterion, cfg.device, architecture_type
                )
            
            if ema:
                ema.restore()
        
        # スケジューラ更新
        if scheduler and epoch >= warmup_epochs:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # 統計記録
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        learning_rates.append(current_lr)
        
        # ★★★ エポック診断終了 & 改善提案 ★★★
        suggestions = diagnostics.end_epoch_diagnosis(epoch + 1, val_loss, model)
        
        # ログ表示
        val_str = f"Val: {val_loss:6.4f}" if val_loss != float('inf') else "Val: ----"
        print(f"\n📈 Epoch [{epoch+1:2d}/{cfg.num_epochs}] "
              f"Train: {avg_train_loss:6.4f} {val_str} "
              f"Time: {epoch_time:4.1f}s LR: {current_lr:.6f}")
        
        # 改善提案表示
        if suggestions:
            print(f"💡 診断提案:")
            for suggestion in suggestions[:2]:  # 上位2件のみ
                print(f"   {suggestion['type']}: {suggestion['suggestion']}")
        
        # GPU使用量表示（最初の数エポック）
        if cfg.device.type == 'cuda' and epoch < 3:
            memory_used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"   GPU Memory: {memory_used:.2f}GB")
        
        # Early Stopping & Best Model Saving
        current_loss = val_loss if val_loss != float('inf') else avg_train_loss
        min_improvement = getattr(cfg, 'min_improvement', 0.005)
        
        if current_loss < best_val_loss - min_improvement:
            best_val_loss = current_loss
            patience_counter = 0
            
            # EMAモデルを保存
            if ema:
                ema.apply_shadow()
            
            save_best_model_integrated(model, optimizer, ema, epoch, best_val_loss, cfg, architecture_type)
            
            if ema:
                ema.restore()
            
            print(f"🎉 New best loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            patience = getattr(cfg, 'patience', 10)
            print(f"⏳ No improvement for {patience_counter}/{patience} epochs")
            
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered")
                break
        
        # 定期保存
        save_interval = getattr(cfg, 'save_interval', 2)
        if (epoch + 1) % save_interval == 0:
            save_checkpoint_integrated(model, optimizer, epoch, avg_train_loss, cfg, architecture_type)
        
        # メモリクリーンアップ
        if cfg.device.type == 'cuda':
            empty_cache_every = getattr(cfg, 'empty_cache_every_n_batch', 50)
            if (epoch + 1) % (empty_cache_every // 10) == 0:  # エポックベースで調整
                torch.cuda.empty_cache()
    
    # ★★★ 最終診断レポート ★★★
    diagnostics.generate_final_report()
    
    print(f"\n✅ Phase 3統合学習完了!")
    print(f"🏆 Best Loss: {best_val_loss:.4f}")
    
    return train_losses, val_losses, learning_rates, best_val_loss

# ===== 統合版保存関数 =====
def save_best_model_integrated(model, optimizer, ema, epoch, loss, cfg, architecture_type):
    """統合版ベストモデル保存"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'architecture_type': architecture_type,
        'ema_state_dict': ema.shadow if ema else None,
        'config': {
            'batch_size': cfg.batch_size,
            'learning_rate': cfg.learning_rate,
            'ema_decay': getattr(cfg, 'ema_decay', None),
            'validation_split': getattr(cfg, 'validation_split', 0.15),
            'use_multiscale': getattr(cfg, 'use_multiscale_architecture', True)
        },
        'training_stats': {
            'gpu_memory_peak': torch.cuda.max_memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0,
            'parameters': sum(p.numel() for p in model.parameters())
        },
        'version_info': VersionTracker.get_all_trackers()
    }
    
    save_path = os.path.join(cfg.save_dir, f'phase3_integrated_{architecture_type}_loss_{loss:.4f}.pth')
    torch.save(checkpoint, save_path)
    print(f"💾 Phase 3 integrated model saved: {save_path}")

def save_checkpoint_integrated(model, optimizer, epoch, loss, cfg, architecture_type):
    """統合版チェックポイント保存"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'architecture_type': architecture_type,
        'config_info': {
            'batch_size': cfg.batch_size,
            'learning_rate': cfg.learning_rate,
            'optimizer_type': getattr(cfg, 'optimizer_type', 'AdamW')
        },
        'version_info': VersionTracker.get_all_trackers()
    }
    
    save_path = os.path.join(cfg.save_dir, f'checkpoint_integrated_epoch_{epoch+1}.pth')
    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint saved: {save_path}")

def plot_training_progress(losses, lrs, val_losses=None, save_path="training_progress_integrated.png"):
    """学習進捗を可視化（統合版）"""
    try:
        import matplotlib.pyplot as plt
        
        if val_losses:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Training Loss plot
        ax1.plot(losses, 'b-', linewidth=2, label='Train Loss')
        if val_losses:
            valid_val_losses = [loss for loss in val_losses if loss != float('inf')]
            valid_epochs = [i for i, loss in enumerate(val_losses) if loss != float('inf')]
            if valid_val_losses:
                ax1.plot(valid_epochs, valid_val_losses, 'r-', linewidth=2, label='Val Loss')
        
        ax1.set_title('Phase 3 Integrated Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        if val_losses:
            ax1.legend()
        
        # Learning Rate plot
        ax2.plot(lrs, 'r-', linewidth=2)
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Validation Loss separate plot
        if val_losses:
            valid_val_losses = [loss for loss in val_losses if loss != float('inf')]
            valid_epochs = [i for i, loss in enumerate(val_losses) if loss != float('inf')]
            if valid_val_losses:
                ax3.plot(valid_epochs, valid_val_losses, 'g-', linewidth=2)
                ax3.set_title('Validation Loss')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Val Loss')
                ax3.grid(True, alpha=0.3)
                ax3.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()  # メモリ節約
        
        print(f"📊 Training progress saved to {save_path}")
    except ImportError:
        print("⚠️ matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"⚠️ 可視化エラー: {e}")

# ===== GPU環境チェック =====
def comprehensive_gpu_check():
    """包括的なGPU環境チェック"""
    print("\n🔍 GPU環境詳細チェック")
    print("="*60)
    
    print(f"1. CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name(0)}")
        
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   GPU memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
    else:
        print("   ❌ CUDA not available!")
        return False
    
    print(f"2. PyTorch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"3. Selected device: {device}")
    
    return True

# ===== メイン関数 =====
def main():
    print("🚀 Starting Phase 3 Integrated YOLO Training (Diagnostic Enhanced)")
    
    # プロジェクト全体のバージョン情報を表示
    print("\n" + "="*80)
    print("📋 プロジェクト全体バージョン確認")
    print("="*80)
    VersionTracker.print_all_versions()

    # 設定とGPU確認
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # 診断ディレクトリ作成
    diagnostic_dir = os.path.join(cfg.save_dir, "diagnostics")
    os.makedirs(diagnostic_dir, exist_ok=True)
    
    # GPU環境詳細チェック
    if not comprehensive_gpu_check():
        print("❌ GPU使用不可 - CPU学習に切り替えます")
        cfg.device = torch.device('cpu')
        cfg.batch_size = max(cfg.batch_size // 4, 1)
        print(f"   CPU用にバッチサイズを {cfg.batch_size} に調整")
    
    print(f"\n📋 学習設定:")
    print(f"   Device: {cfg.device}")
    print(f"   Batch size: {cfg.batch_size}")
    print(f"   Image size: {cfg.img_size}")
    print(f"   Classes: {cfg.num_classes}")
    print(f"   Learning rate: {cfg.learning_rate:.0e}")
    print(f"   EMA: {getattr(cfg, 'use_ema', True)}")
    print(f"   Validation Split: {getattr(cfg, 'validation_split', 0.15)}")
    print(f"   Diagnostics: ON")
    
    # Phase 3統合: データセット & 検証分割
    print("\n📊 Loading dataset with validation split...")
    try:
        train_dataloader, val_dataloader = setup_dataloaders(cfg)
        
        print(f"   Total train batches: {len(train_dataloader)}")
        if val_dataloader:
            print(f"   Total validation batches: {len(val_dataloader)}")
    except Exception as e:
        print(f"❌ データローダー作成エラー: {e}")
        print("   標準データセットでリトライ中...")
        
        # フォールバック
        global USE_IMPROVED
        USE_IMPROVED = False
        train_dataloader, val_dataloader = setup_dataloaders(cfg)
    
    # Phase 3統合: モデル&損失関数（アーキテクチャ自動選択）
    print("\n🤖 Creating Phase 3 integrated model...")
    model, criterion, architecture_type = create_model_and_loss(cfg)
    
    # GPUに移動
    print(f"   Moving model to {cfg.device}...")
    model = model.to(cfg.device)
    
    if cfg.device.type == 'cuda':
        model = model.float()
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    model_device = next(model.parameters()).device
    print(f"   Model device confirmed: {model_device}")
    
    # Phase 3統合学習実行
    print("\n🚀 Phase 3統合学習を開始（診断機能付き）")
    print(f"🎯 目標: Val Loss 43.45 → 25-30")
    
    try:
        train_losses, val_losses, lrs, best_loss = phase3_integrated_training_loop(
            model, train_dataloader, val_dataloader, criterion, cfg, architecture_type
        )
        
        # 結果可視化
        try:
            plot_training_progress(train_losses, lrs, val_losses)
        except Exception as e:
            print(f"⚠️ 可視化エラー: {e}")
        
        print(f"\n🎯 Phase 3統合学習完了!")
        print(f"🏆 Best Loss: {best_loss:.4f}")
        print(f"🔧 Architecture: {architecture_type}")
        
        # 目標達成判定
        print(f"\n📊 目標達成判定:")
        if best_loss < 30.0:
            print("🎉 Step 1目標達成! (Val Loss < 30.0)")
            if best_loss < 20.0:
                print("🏆 予想を上回る成果! (Val Loss < 20.0)")
        elif best_loss < 35.0:
            print("🟡 部分的改善 (Val Loss < 35.0) - 継続推奨")
        else:
            print("🔴 目標未達成 - 方針転換検討")
            print("   推奨: 学習率をさらに2倍、またはアンカー見直し")
        
        # 具体的な改善提案
        improvement_ratio = 43.45 / best_loss if best_loss > 0 else 1
        print(f"\n💡 改善結果:")
        print(f"   改善率: {improvement_ratio:.1f}x (43.45 → {best_loss:.4f})")
        print(f"   次目標: Val Loss < {best_loss * 0.7:.1f}")
        
        if best_loss > 35.0:
            print(f"\n🚨 緊急改善案:")
            print(f"   1. 学習率を2倍に ({cfg.learning_rate:.0e} → {cfg.learning_rate*2:.0e})")
            print(f"   2. バッチサイズを半分に ({cfg.batch_size} → {cfg.batch_size//2})")
            print(f"   3. アンカーサイズの全面見直し")
        
    except Exception as e:
        print(f"❌ 学習エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ Training completed!")
    
    # 最終統計
    if cfg.device.type == 'cuda':
        final_memory = torch.cuda.memory_allocated(0) / 1024**3
        max_memory = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"📊 最終GPU統計:")
        print(f"   現在のメモリ使用量: {final_memory:.2f}GB")
        print(f"   最大メモリ使用量: {max_memory:.2f}GB")
    
    # 学習結果サマリー
    print(f"\n🎯 学習結果サマリー:")
    print(f"   Architecture: {architecture_type}")
    print(f"   最終Loss: {best_loss:.4f}")
    print(f"   改良拡張: {'ON' if USE_IMPROVED else 'OFF'}")
    print(f"   診断機能: ON")
    print(f"   EMA使用: {'Yes' if getattr(cfg, 'use_ema', True) else 'No'}")
    print(f"   検証分割: {'Yes' if getattr(cfg, 'validation_split', 0.15) > 0 else 'No'}")
    print(f"   保存先: {cfg.save_dir}")
    print(f"   診断ログ: {diagnostic_dir}")

if __name__ == "__main__":
    main()