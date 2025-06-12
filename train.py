# train.py - Phase 3 EMA&検証分割完全版
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import os

from config import Config
from dataset import FLIRDataset, collate_fn
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
training_version = create_version_tracker("Training System v1.3 - Phase 3 Complete", "train.py")
training_version.add_modification("学習ループ実装")
training_version.add_modification("プロジェクトバージョン表示追加")
training_version.add_modification("gpu未使用原因の追究")
training_version.add_modification("Phase 3: EMA & 検証分割実装")

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

# ===== Phase 3: 検証関数 =====
def validate_model(model, val_dataloader, criterion, device):
    """検証実行"""
    model.eval()
    total_val_loss = 0
    val_batches = 0
    
    with torch.no_grad():
        for images, targets in val_dataloader:
            images = images.to(device, non_blocking=True)
            
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

# ===== Phase 3: EMA&検証分割対応学習ループ =====
def optimized_training_loop_with_ema_val(model, train_dataloader, val_dataloader, criterion, cfg):
    """Phase 3完全版: EMA + 検証分割対応学習ループ"""
    
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
    
    print(f"🚀 Phase 3 EMA&検証分割学習開始")
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
        
        for batch_idx, (images, targets) in enumerate(train_dataloader):
            images = images.to(cfg.device, non_blocking=True)
            
            # Forward
            predictions, grid_size = model(images)
            loss = criterion(predictions, targets, grid_size)
            
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
            
            if batch_idx % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"   Batch [{batch_idx:4d}] Loss: {loss.item():8.4f} "
                      f"AvgLoss: {avg_loss:8.4f} LR: {current_lr:.6f}")
        
        avg_train_loss = epoch_loss / batch_count
        
        # ===== 検証フェーズ =====
        val_loss = float('inf')
        if val_dataloader and epoch % cfg.validate_every == 0:
            # EMAモデルで検証
            if ema:
                ema.apply_shadow()
            
            val_loss = validate_model(model, val_dataloader, criterion, cfg.device)
            
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
        
        # Early Stopping & Best Model Saving
        current_loss = val_loss if val_loss != float('inf') else avg_train_loss
        
        if current_loss < best_val_loss - cfg.min_improvement:
            best_val_loss = current_loss
            patience_counter = 0
            
            # EMAモデルを保存
            if ema:
                ema.apply_shadow()
            
            save_best_model_with_ema(model, optimizer, ema, epoch, best_val_loss, cfg)
            
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
            save_checkpoint(model, optimizer, epoch, avg_train_loss, cfg)
        
        # メモリクリーンアップ
        if cfg.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"\n✅ Phase 3 EMA&検証分割学習完了!")
    print(f"🏆 Best Loss: {best_val_loss:.4f}")
    
    return train_losses, val_losses, learning_rates, best_val_loss

# ===== 旧学習ループ（後方互換性のため残す） =====
def optimized_training_loop(model, dataloader, criterion, cfg):
    """Phase 3: 最適化された学習ループ（旧版・後方互換性）"""
    
    # ===== オプティマイザ設定 =====
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
    
    # ===== スケジューラ設定 =====
    scheduler = None
    if cfg.use_scheduler:
        if cfg.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=cfg.num_epochs,
                eta_min=cfg.min_lr
            )
        elif cfg.scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.1
            )
    
    # ===== Early Stopping設定 =====
    best_loss = float('inf')
    patience_counter = 0
    
    # ===== 学習統計 =====
    train_losses = []
    learning_rates = []
    
    print(f"🚀 Phase 3 最適化学習開始（旧版）")
    print(f"   Optimizer: {cfg.optimizer_type}")
    print(f"   Scheduler: {cfg.scheduler_type if cfg.use_scheduler else 'None'}")
    print(f"   Initial LR: {cfg.learning_rate}")
    
    for epoch in range(cfg.num_epochs):
        epoch_start = time.time()
        
        # ===== ウォームアップ処理 =====
        if epoch < cfg.warmup_epochs:
            warmup_lr = cfg.learning_rate * (epoch + 1) / cfg.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"🔥 Warmup Epoch {epoch+1}: LR = {warmup_lr:.6f}")
        
        # ===== 学習エポック =====
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(cfg.device, non_blocking=True)
            
            # Forward
            predictions, grid_size = model(images)
            loss = criterion(predictions, targets, grid_size)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # バッチログ（改善）
            if batch_idx % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"   Batch [{batch_idx:4d}] Loss: {loss.item():8.4f} "
                      f"AvgLoss: {avg_loss:8.4f} LR: {current_lr:.6f}")
        
        # ===== エポック終了処理 =====
        avg_epoch_loss = epoch_loss / batch_count
        epoch_time = time.time() - epoch_start
        
        # スケジューラ更新（ウォームアップ後）
        if scheduler and epoch >= cfg.warmup_epochs:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # ===== 統計記録 =====
        train_losses.append(avg_epoch_loss)
        learning_rates.append(current_lr)
        
        # ===== ログ表示 =====
        print(f"\n📈 Epoch [{epoch+1:2d}/{cfg.num_epochs}] "
              f"Loss: {avg_epoch_loss:8.4f} Time: {epoch_time:5.1f}s LR: {current_lr:.6f}")
        
        # ===== Early Stopping & Model Saving =====
        if avg_epoch_loss < best_loss - cfg.min_improvement:
            best_loss = avg_epoch_loss
            patience_counter = 0
            
            # ベストモデル保存
            save_optimized_model(model, optimizer, epoch, best_loss, cfg)
            print(f"🎉 New best loss: {best_loss:.4f}")
            
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter}/{cfg.patience} epochs")
            
            if patience_counter >= cfg.patience:
                print(f"🛑 Early stopping triggered")
                break
        
        # ===== 定期保存 =====
        if (epoch + 1) % cfg.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, avg_epoch_loss, cfg)
        
        # ===== メモリクリーンアップ =====
        if cfg.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # ===== 学習完了 =====
    print(f"\n✅ Phase 3 学習完了!")
    print(f"🏆 Best Loss: {best_loss:.4f}")
    
    return train_losses, learning_rates, best_loss

# ===== EMA対応保存関数 =====
def save_best_model_with_ema(model, optimizer, ema, epoch, loss, cfg):
    """EMA対応の改良版モデル保存"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'ema_state_dict': ema.shadow if ema else None,
        'config': {
            'batch_size': cfg.batch_size,
            'learning_rate': cfg.learning_rate,
            'ema_decay': cfg.ema_decay if cfg.use_ema else None,
            'validation_split': cfg.validation_split
        },
        'training_stats': {
            'gpu_memory_peak': torch.cuda.max_memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0,
            'parameters': sum(p.numel() for p in model.parameters())
        },
        'version_info': VersionTracker.get_all_trackers()
    }
    
    save_path = os.path.join(cfg.save_dir, f'phase3_ema_val_loss_{loss:.4f}.pth')
    torch.save(checkpoint, save_path)
    print(f"💾 Phase 3 EMA model saved: {save_path}")

def save_optimized_model(model, optimizer, epoch, loss, cfg):
    """最適化されたモデル保存"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': {
            'optimizer_type': cfg.optimizer_type,
            'learning_rate': cfg.learning_rate,
            'batch_size': cfg.batch_size,
            'scheduler_type': cfg.scheduler_type if cfg.use_scheduler else None
        },
        'training_stats': {
            'gpu_memory_peak': torch.cuda.max_memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0,
            'parameters': sum(p.numel() for p in model.parameters())
        },
        'version_info': VersionTracker.get_all_trackers()
    }
    
    save_path = os.path.join(cfg.save_dir, f'phase3_best_model_loss_{loss:.4f}.pth')
    torch.save(checkpoint, save_path)
    print(f"💾 Phase 3 best model saved: {save_path}")

def save_checkpoint(model, optimizer, epoch, loss, cfg):
    """定期チェックポイント保存"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config_info': {
            'batch_size': cfg.batch_size,
            'learning_rate': cfg.learning_rate,
            'optimizer_type': getattr(cfg, 'optimizer_type', 'Adam')
        },
        'version_info': VersionTracker.get_all_trackers()
    }
    
    save_path = os.path.join(cfg.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint saved: {save_path}")

def plot_training_progress(losses, lrs, val_losses=None, save_path="training_progress.png"):
    """学習進捗を可視化（検証loss対応）"""
    try:
        import matplotlib.pyplot as plt
        
        if val_losses:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Training Loss plot
        ax1.plot(losses, 'b-', linewidth=2, label='Train Loss')
        if val_losses:
            # None値を除外してplot
            valid_val_losses = [loss for loss in val_losses if loss != float('inf')]
            valid_epochs = [i for i, loss in enumerate(val_losses) if loss != float('inf')]
            if valid_val_losses:
                ax1.plot(valid_epochs, valid_val_losses, 'r-', linewidth=2, label='Val Loss')
        
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale for better visualization
        if val_losses:
            ax1.legend()
        
        # Learning Rate plot
        ax2.plot(lrs, 'r-', linewidth=2)
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Validation Loss separate plot (if available)
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

# ===== デバッグ・GPU関連機能 =====
def comprehensive_gpu_check():
    """包括的なGPU環境チェック"""
    print("\n🔍 GPU環境詳細チェック")
    print("="*60)
    
    # 1. CUDA可用性
    print(f"1. CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name(0)}")
        
        # メモリ情報
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   GPU memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
    else:
        print("   ❌ CUDA not available!")
        return False
    
    # 2. PyTorchバージョン
    print(f"2. PyTorch version: {torch.__version__}")
    
    # 3. デバイス設定確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"3. Selected device: {device}")
    
    return True

def test_version_tracking():
    """バージョン管理システムの動作テスト"""
    print("🧪 バージョン管理システムテスト開始")
    
    # デバッグ情報を表示
    debug_version_status()
    
    # 登録されているファイル数をチェック
    count = get_version_count()
    print(f"\n📊 現在の登録状況:")
    print(f"   登録済みファイル数: {count}")
    print(f"   期待値: 4ファイル (dataset, model, loss, train)")
    
    if count >= 4:
        print("✅ バージョン管理システム正常動作")
    else:
        print(f"⚠️ 期待される4ファイルのうち{count}ファイルのみ登録済み")
        print("   他のファイルがまだ読み込まれていない可能性があります")

# ===== メイン関数 =====
def main():
    print("🚀 Starting Modular YOLO Training - Phase 3 Complete")
    
    # プロジェクト全体のバージョン情報を表示
    print("\n" + "="*80)
    print("📋 プロジェクト全体バージョン確認")
    print("="*80)
    VersionTracker.print_all_versions()  # 詳細版

    # ★★★ 設定とGPU確認 ★★★
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # GPU環境詳細チェック
    if not comprehensive_gpu_check():
        print("❌ GPU使用不可 - CPU学習に切り替えます")
        cfg.device = torch.device('cpu')
        cfg.batch_size = max(cfg.batch_size // 4, 1)  # バッチサイズを削減
        print(f"   CPU用にバッチサイズを {cfg.batch_size} に調整")
    
    print(f"\n📋 学習設定:")
    print(f"   Device: {cfg.device}")
    print(f"   Batch size: {cfg.batch_size}")
    print(f"   Image size: {cfg.img_size}")
    print(f"   Classes: {cfg.num_classes}")
    print(f"   EMA: {cfg.use_ema}")
    print(f"   Validation Split: {cfg.validation_split}")
    
    # ★★★ Phase 3: データセット & 検証分割 ★★★
    print("\n📊 Loading dataset with validation split...")
    train_dataloader, val_dataloader = setup_dataloaders(cfg)
    
    print(f"   Total train batches: {len(train_dataloader)}")
    if val_dataloader:
        print(f"   Total validation batches: {len(val_dataloader)}")
    
    # ★★★ モデル（明示的にGPUに移動） ★★★
    print("\n🤖 Creating and setting up model...")
    model = SimpleYOLO(cfg.num_classes).to(cfg.device)
    
    # モデルを明示的にGPUに移動
    print(f"   Moving model to {cfg.device}...")
    model = model.to(cfg.device)
    
    # float32を強制（混合精度を避ける）
    if cfg.device.type == 'cuda':
        model = model.float()
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # モデルがGPUに正しく配置されているか確認
    model_device = next(model.parameters()).device
    print(f"   Model device confirmed: {model_device}")
    
    # ★★★ 損失関数 ★★★
    criterion = YOLOLoss(cfg.num_classes)
    
    # ★★★ Phase 3 EMA&検証分割学習 または 従来学習 ★★★
    if cfg.use_phase3_optimization:
        print("\n🚀 Phase 3 EMA&検証分割学習を開始")
        
        # Phase 3 完全版学習実行
        train_losses, val_losses, lrs, best_loss = optimized_training_loop_with_ema_val(
            model, train_dataloader, val_dataloader, criterion, cfg
        )
        
        # 結果可視化
        try:
            plot_training_progress(train_losses, lrs, val_losses)
        except Exception as e:
            print(f"⚠️ 可視化エラー: {e}")
        
        print(f"\n🎯 Phase 3 EMA&検証分割完了! Best Loss: {best_loss:.4f}")
        if best_loss < 1.0:
            print("🎉 Phase 3 目標達成! (Loss < 1.0)")
        if best_loss < 0.5:
            print("🏆 最終目標達成! (Loss < 0.5)")
            
    else:
        print("\n📚 従来学習を開始")
        
        # 従来学習ループ（単一データローダー）
        train_losses, lrs, best_loss = optimized_training_loop(model, train_dataloader, criterion, cfg)
        
        # 結果可視化
        try:
            plot_training_progress(train_losses, lrs)
        except Exception as e:
            print(f"⚠️ 可視化エラー: {e}")
        
        print(f"\n✅ 従来学習完了! Best Loss: {best_loss:.4f}")
    
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
    print(f"   最終Loss: {best_loss:.4f}")
    print(f"   EMA使用: {'Yes' if cfg.use_ema else 'No'}")
    print(f"   検証分割: {'Yes' if cfg.validation_split > 0 else 'No'}")
    print(f"   保存先: {cfg.save_dir}")

if __name__ == "__main__":
    main()