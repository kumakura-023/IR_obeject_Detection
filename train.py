# train_phase3_integrated.py - マルチスケールYOLO統合版
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import os

from config import Config
from dataset import FLIRDataset, collate_fn

# ★★★ Phase 3 新アーキテクチャをインポート ★★★
from multiscale_model import MultiScaleYOLO
from anchor_loss import MultiScaleAnchorLoss

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
training_version = create_version_tracker("Training System v2.2 - Phase 3 Integrated", "train.py")
training_version.add_modification("Phase 3完全統合: マルチスケール + アンカーベース")
training_version.add_modification("EMA + 検証分割継続")
training_version.add_modification("フォールバック機能付き")
training_version.add_modification("進捗情報追加")

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
            criterion = MultiScaleAnchorLoss(anchors=anchors, num_classes=cfg.num_classes)
            
            print(f"   ✅ MultiScaleYOLO: {sum(p.numel() for p in model.parameters()):,} parameters")
            print(f"   ✅ MultiScaleAnchorLoss: 3スケール対応")
            
            return model, criterion, "multiscale"
            
        except Exception as e:
            print(f"⚠️ マルチスケール初期化失敗: {e}")
            print("📚 フォールバックモードに切り替えます")
    
    # フォールバック: 従来アーキテクチャ
    print("📚 フォールバック: 従来アーキテクチャを使用")
    model = SimpleYOLO(cfg.num_classes)
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

# ===== Phase 3: データローダーセットアップ =====
def setup_dataloaders(cfg):
    """データローダーセットアップ（検証分割対応）"""
    # 全データセット作成
    full_dataset = FLIRDataset(cfg.train_img_dir, cfg.train_label_dir, cfg.img_size, augment=cfg.augment)
    
    # 検証分割
    if cfg.validation_split > 0:
        train_dataset, val_dataset = create_train_val_split(full_dataset, cfg.validation_split)
    else:
        train_dataset = full_dataset
        val_dataset = None
    
    # DataLoader作成
    num_workers = 2 if cfg.device.type == 'cuda' else 0
    pin_memory = cfg.device.type == 'cuda'
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,  # 検証時はシャッフルしない
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    print(f"   Train batches: {len(train_dataloader)}")
    if val_dataloader:
        print(f"   Validation batches: {len(val_dataloader)}")
    
    return train_dataloader, val_dataloader

# ===== Phase 3: マルチスケール対応学習ループ =====
def phase3_integrated_training_loop(model, train_dataloader, val_dataloader, criterion, cfg, architecture_type):
    """Phase 3統合学習ループ（マルチスケール対応）"""
    
    # オプティマイザ設定
    if cfg.optimizer_type == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            eps=cfg.eps
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    # EMA初期化
    ema = None
    if cfg.use_ema:
        ema = EMAModel(model, decay=cfg.ema_decay)
        print(f"🔄 EMA initialized with decay {cfg.ema_decay}")
    
    # スケジューラ設定
    scheduler = None
    if cfg.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=cfg.num_epochs,
            eta_min=cfg.min_lr
        )
    
    # Early Stopping設定
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 学習統計
    train_losses = []
    val_losses = []
    learning_rates = []
    
    print(f"🚀 Phase 3統合学習開始")
    print(f"   Architecture: {architecture_type}")
    print(f"   Optimizer: {cfg.optimizer_type}")
    print(f"   EMA: {'ON' if cfg.use_ema else 'OFF'}")
    print(f"   Validation: {'ON' if val_dataloader else 'OFF'}")
    
    for epoch in range(cfg.num_epochs):
        epoch_start = time.time()
        
        # ===== ウォームアップ処理 =====
        if epoch < cfg.warmup_epochs:
            warmup_lr = cfg.learning_rate * (epoch + 1) / cfg.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"🔥 Warmup Epoch {epoch+1}: LR = {warmup_lr:.6f}")
        
        # ===== 訓練フェーズ =====
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        # 🆕 進捗トラッカー初期化
        from progress import MultiScaleProgressTracker
        progress_tracker = MultiScaleProgressTracker(len(train_dataloader), print_interval=100)
        progress_tracker.start_epoch(epoch + 1, cfg.num_epochs)
        
        for batch_idx, (images, targets) in enumerate(train_dataloader):
            images = images.to(cfg.device, non_blocking=True)
            
            # Forward（アーキテクチャ別）
            if architecture_type == "multiscale":
                predictions = model(images)
                loss = criterion(predictions, targets)
                
                # 🆕 マルチスケール詳細情報取得（100バッチごと）
                scale_losses = None
                if batch_idx % 100 == 0 and hasattr(criterion, 'return_components'):
                    # 詳細情報付きで再計算
                    criterion.return_components = True
                    _, _, scale_losses = criterion(predictions, targets)
                    criterion.return_components = False
                
            else:
                predictions, grid_size = model(images)
                loss = criterion(predictions, targets, grid_size)
                scale_losses = None
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            
            # EMA更新
            if ema:
                ema.update()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # 🆕 詳細進捗表示（100バッチごと）
            current_lr = optimizer.param_groups[0]['lr']
            if architecture_type == "multiscale" and scale_losses:
                progress_tracker.update_batch_multiscale(
                    batch_idx, loss.item(), current_lr, scale_losses
                )
            else:
                progress_tracker.update_batch(batch_idx, loss.item(), current_lr)
        
        avg_train_loss = epoch_loss / batch_count
        
        # ===== 検証フェーズ =====
        val_loss = float('inf')
        if val_dataloader and epoch % cfg.validate_every == 0:
            # EMAモデルで検証
            if ema:
                ema.apply_shadow()
            
            val_loss = validate_model(model, val_dataloader, criterion, cfg.device, architecture_type)
            
            if ema:
                ema.restore()
        
        # スケジューラ更新
        if scheduler and epoch >= cfg.warmup_epochs:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # 統計記録
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        learning_rates.append(current_lr)
        
        # ログ表示
        val_str = f"Val: {val_loss:6.4f}" if val_loss != float('inf') else "Val: ----"
        print(f"\n📈 Epoch [{epoch+1:2d}/{cfg.num_epochs}] "
              f"Train: {avg_train_loss:6.4f} {val_str} "
              f"Time: {epoch_time:4.1f}s LR: {current_lr:.6f}")
        
        # GPU使用量表示（最初の数エポック）
        if cfg.device.type == 'cuda' and epoch < 5:
            memory_used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"   GPU Memory: {memory_used:.2f}GB")
        
        # Early Stopping & Best Model Saving
        current_loss = val_loss if val_loss != float('inf') else avg_train_loss
        
        if current_loss < best_val_loss - cfg.min_improvement:
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
            print(f"⏳ No improvement for {patience_counter}/{cfg.patience} epochs")
            
            if patience_counter >= cfg.patience:
                print(f"🛑 Early stopping triggered")
                break
        
        # 定期保存
        if (epoch + 1) % cfg.save_interval == 0:
            save_checkpoint_integrated(model, optimizer, epoch, avg_train_loss, cfg, architecture_type)
        
        # メモリクリーンアップ
        if cfg.device.type == 'cuda':
            torch.cuda.empty_cache()
    
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
            'ema_decay': cfg.ema_decay if cfg.use_ema else None,
            'validation_split': cfg.validation_split,
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
            'optimizer_type': getattr(cfg, 'optimizer_type', 'Adam')
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
        plt.show()
        
        print(f"📊 Training progress saved to {save_path}")
    except ImportError:
        print("⚠️ matplotlib not available - skipping visualization")

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
    print("🚀 Starting Phase 3 Integrated YOLO Training")
    
    # プロジェクト全体のバージョン情報を表示
    print("\n" + "="*80)
    print("📋 プロジェクト全体バージョン確認")
    print("="*80)
    VersionTracker.print_all_versions()

    # 設定とGPU確認
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    
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
    print(f"   EMA: {cfg.use_ema}")
    print(f"   Validation Split: {cfg.validation_split}")
    
    # Phase 3統合: データセット & 検証分割
    print("\n📊 Loading dataset with validation split...")
    train_dataloader, val_dataloader = setup_dataloaders(cfg)
    
    print(f"   Total train batches: {len(train_dataloader)}")
    if val_dataloader:
        print(f"   Total validation batches: {len(val_dataloader)}")
    
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
    print("\n🚀 Phase 3統合学習を開始")
    
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
    if best_loss < 5.0:
        print("🎉 Phase 3目標達成! (Val Loss < 5.0)")
    if best_loss < 1.0:
        print("🏆 Phase 3完全達成! (Loss < 1.0)")
    if best_loss < 0.5:
        print("🚀 最終目標達成! (Loss < 0.5)")
    
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
    print(f"   EMA使用: {'Yes' if cfg.use_ema else 'No'}")
    print(f"   検証分割: {'Yes' if cfg.validation_split > 0 else 'No'}")
    print(f"   保存先: {cfg.save_dir}")

if __name__ == "__main__":
    main()