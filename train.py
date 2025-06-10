# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
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
training_version = create_version_tracker("Training System v1.1", "train.py")
training_version.add_modification("学習ループ実装")
training_version.add_modification("プロジェクトバージョン表示追加")
training_version.add_modification("gpu未使用原因の追究")

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

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        
        # Forward
        predictions, grid_size = model(images)
        loss = criterion(predictions, targets, grid_size)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 進捗表示
        if batch_idx % Config.print_interval == 0:
            print(f"Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


import torch
import time
import psutil
import os

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

def check_model_device(model):
    """モデルがGPUに配置されているか確認"""
    print("\n🔍 モデルデバイス確認")
    print("-"*40)
    
    model_device = next(model.parameters()).device
    print(f"Model device: {model_device}")
    
    # 各レイヤーのデバイスをチェック
    layers_on_gpu = 0
    total_layers = 0
    
    for name, param in model.named_parameters():
        total_layers += 1
        if param.device.type == 'cuda':
            layers_on_gpu += 1
        elif total_layers <= 5:  # 最初の5レイヤーのみ表示
            print(f"   ⚠️ {name}: {param.device}")
    
    print(f"GPU上のレイヤー: {layers_on_gpu}/{total_layers}")
    
    if layers_on_gpu == total_layers:
        print("✅ モデル全体がGPU上にあります")
        return True
    else:
        print(f"❌ {total_layers - layers_on_gpu} レイヤーがCPU上にあります")
        return False

def check_data_device(images, targets):
    """データがGPUに配置されているか確認"""
    print(f"\n🔍 データデバイス確認")
    print("-"*40)
    
    print(f"Images device: {images.device}")
    print(f"Images shape: {images.shape}")
    print(f"Images dtype: {images.dtype}")
    
    if isinstance(targets, list):
        if len(targets) > 0 and torch.is_tensor(targets[0]):
            print(f"Targets[0] device: {targets[0].device}")
            print(f"Targets[0] shape: {targets[0].shape}")
    else:
        print(f"Targets device: {targets.device}")
    
    return images.device.type == 'cuda'

def gpu_utilization_test():
    """GPU使用率テスト"""
    print("\n🔍 GPU使用率テスト")
    print("-"*40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    device = torch.device('cuda')
    
    # 大きなテンソル操作でGPU使用率をテスト
    print("大きなテンソル操作を実行中...")
    
    start_time = time.time()
    
    # GPU上で重い計算
    a = torch.randn(5000, 5000, device=device)
    b = torch.randn(5000, 5000, device=device)
    
    for i in range(10):
        c = torch.mm(a, b)
        if i == 0:
            print(f"First operation result device: {c.device}")
    
    end_time = time.time()
    print(f"GPU計算時間: {end_time - start_time:.2f}秒")
    
    # 同じ計算をCPUで実行して比較
    print("同じ計算をCPUで実行中...")
    start_time = time.time()
    
    a_cpu = torch.randn(5000, 5000)
    b_cpu = torch.randn(5000, 5000)
    
    for i in range(10):
        c_cpu = torch.mm(a_cpu, b_cpu)
    
    end_time = time.time()
    print(f"CPU計算時間: {end_time - start_time:.2f}秒")
    
    # メモリ使用量確認
    memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
    memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"テスト後のGPUメモリ - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")

def check_dataloader_efficiency(dataloader, device, num_batches=3):
    """DataLoaderの効率性をチェック"""
    print(f"\n🔍 DataLoader効率性チェック")
    print("-"*40)
    
    total_load_time = 0
    total_gpu_transfer_time = 0
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        load_start = time.time()
        
        # GPU転送時間測定
        gpu_start = time.time()
        images = images.to(device, non_blocking=True)
        torch.cuda.synchronize()  # GPU転送完了を待つ
        gpu_end = time.time()
        
        load_end = time.time()
        
        batch_load_time = load_end - load_start
        gpu_transfer_time = gpu_end - gpu_start
        
        total_load_time += batch_load_time
        total_gpu_transfer_time += gpu_transfer_time
        
        print(f"Batch {batch_idx}: Load {batch_load_time:.3f}s, GPU transfer {gpu_transfer_time:.3f}s")
        print(f"   Images shape: {images.shape}, device: {images.device}")
    
    avg_load_time = total_load_time / num_batches
    avg_gpu_time = total_gpu_transfer_time / num_batches
    
    print(f"\n平均読み込み時間: {avg_load_time:.3f}s")
    print(f"平均GPU転送時間: {avg_gpu_time:.3f}s")
    
    if avg_gpu_time > avg_load_time * 0.5:
        print("⚠️ GPU転送が遅い可能性があります")
        print("   対策: pin_memory=True, non_blocking=True を試してください")

def monitor_training_step(model, images, targets, criterion, optimizer, device):
    """1ステップの学習をモニタリング"""
    print(f"\n🔍 学習ステップモニタリング")
    print("-"*40)
    
    # 開始時のGPUメモリ
    start_memory = torch.cuda.memory_allocated(0) / 1024**3
    
    # Forward pass
    forward_start = time.time()
    predictions, grid_size = model(images)
    torch.cuda.synchronize()
    forward_end = time.time()
    
    forward_memory = torch.cuda.memory_allocated(0) / 1024**3
    
    # Loss計算
    loss_start = time.time()
    loss = criterion(predictions, targets, grid_size)
    torch.cuda.synchronize()
    loss_end = time.time()
    
    loss_memory = torch.cuda.memory_allocated(0) / 1024**3
    
    # Backward pass
    backward_start = time.time()
    optimizer.zero_grad()
    loss.backward()
    torch.cuda.synchronize()
    backward_end = time.time()
    
    backward_memory = torch.cuda.memory_allocated(0) / 1024**3
    
    # Optimizer step
    step_start = time.time()
    optimizer.step()
    torch.cuda.synchronize()
    step_end = time.time()
    
    final_memory = torch.cuda.memory_allocated(0) / 1024**3
    
    print(f"Forward pass: {forward_end - forward_start:.3f}s (+{forward_memory - start_memory:.2f}GB)")
    print(f"Loss calculation: {loss_end - loss_start:.3f}s (+{loss_memory - forward_memory:.2f}GB)")
    print(f"Backward pass: {backward_end - backward_start:.3f}s (+{backward_memory - loss_memory:.2f}GB)")
    print(f"Optimizer step: {step_end - step_start:.3f}s (+{final_memory - backward_memory:.2f}GB)")
    print(f"Total memory used: {final_memory:.2f}GB")
    
    # 予測とターゲットのデバイス確認
    print(f"Predictions device: {predictions.device}")
    print(f"Loss device: {loss.device}")

# train.py のmain関数に追加する関数
def debug_training_setup(model, dataloader, criterion, optimizer, device):
    """学習セットアップの完全デバッグ"""
    print("\n🚀 学習セットアップ完全デバッグ")
    print("="*60)
    
    # 1. 環境チェック
    if not comprehensive_gpu_check():
        return False
    
    # 2. モデルチェック
    if not check_model_device(model):
        print("🔧 モデルをGPUに移動しています...")
        model = model.to(device)
        check_model_device(model)
    
    # 3. GPU使用率テスト
    gpu_utilization_test()
    
    # 4. DataLoaderチェック
    check_dataloader_efficiency(dataloader, device)
    
    # 5. 実際の学習ステップをテスト
    print("\n🔍 実際の学習ステップをテスト中...")
    model.train()
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        # データをGPUに移動
        images = images.to(device, non_blocking=True)
        
        # データデバイス確認
        if not check_data_device(images, targets):
            print("🔧 データをGPUに移動しています...")
            images = images.to(device)
        
        # 1ステップをモニタリング
        monitor_training_step(model, images, targets, criterion, optimizer, device)
        
        print("✅ デバッグ完了 - 学習を開始します")
        break
    
    return True


def main():
    print("🚀 Starting Modular YOLO Training")
    
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
    
    # ★★★ データセット（GPU最適化） ★★★
    print("\n📊 Loading dataset...")
    dataset = FLIRDataset(cfg.train_img_dir, cfg.train_label_dir, cfg.img_size)
    
    # DataLoader設定をGPU最適化
    num_workers = 2 if cfg.device.type == 'cuda' else 0
    pin_memory = cfg.device.type == 'cuda'
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    print(f"   Dataset: {len(dataset)} images")
    print(f"   Batches: {len(dataloader)}")
    print(f"   DataLoader: workers={num_workers}, pin_memory={pin_memory}")
    
    # ★★★ モデル（明示的にGPUに移動） ★★★
    print("\n🤖 Creating and setting up model...")
    model = SimpleYOLO(cfg.num_classes)
    
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
    
    # ★★★ 損失関数とオプティマイザ ★★★
    criterion = YOLOLoss(cfg.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    # ★★★ 学習開始前の包括的デバッグ ★★★
    if cfg.device.type == 'cuda':
        print("\n🔍 GPU学習セットアップをデバッグ中...")
        debug_success = debug_training_setup(model, dataloader, criterion, optimizer, cfg.device)
        
        if not debug_success:
            print("❌ GPU設定に問題があります。CPU学習に切り替えます。")
            cfg.device = torch.device('cpu')
            model = model.to(cfg.device)
    
    # ★★★ 最適化された学習ループ ★★★
    print(f"\n🎯 Starting training on {cfg.device}...")
    best_loss = float('inf')
    
    # 初回GPU使用率確認
    if cfg.device.type == 'cuda':
        print("\n📊 初回バッチでGPU使用率確認...")
    
    for epoch in range(cfg.num_epochs):
        start_time = time.time()
        
        # 最適化された学習エポック
        avg_loss = train_one_epoch_optimized(
            model, dataloader, criterion, optimizer, cfg.device, epoch
        )
        
        epoch_time = time.time() - start_time
        print(f"\nEpoch [{epoch+1}/{cfg.num_epochs}] "
              f"Loss: {avg_loss:.4f} Time: {epoch_time:.1f}s")
        
        # GPU使用状況表示（最初の3エポックのみ）
        if cfg.device.type == 'cuda' and epoch < 3:
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"   GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        # モデル保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            # バージョン情報付きでチェックポイント保存
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'device': str(cfg.device),
                'version_info': VersionTracker.get_all_trackers()
            }
            
            torch.save(checkpoint, os.path.join(cfg.save_dir, 'best_model.pth'))
            print(f"💾 Best model saved (loss: {best_loss:.4f})")
    
    print("\n✅ Training completed!")
    
    # 最終GPU統計
    if cfg.device.type == 'cuda':
        final_memory = torch.cuda.memory_allocated(0) / 1024**3
        max_memory = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"📊 最終GPU統計:")
        print(f"   現在のメモリ使用量: {final_memory:.2f}GB")
        print(f"   最大メモリ使用量: {max_memory:.2f}GB")

def train_one_epoch_optimized(model, dataloader, criterion, optimizer, device, epoch):
    """GPU最適化された学習エポック"""
    model.train()
    total_loss = 0
    batch_count = 0
    
    # エポック開始時の設定
    if device.type == 'cuda':
        torch.cuda.empty_cache()  # GPU メモリをクリア
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        batch_start_time = time.time()
        
        # ★★★ データを明示的にGPUに移動 ★★★
        images = images.to(device, non_blocking=True)
        
        # targetsがリストの場合の処理
        if isinstance(targets, list):
            # リストの各要素をGPUに移動（必要に応じて）
            pass  # YOLO lossでは通常CPUのまま処理
        else:
            targets = targets.to(device, non_blocking=True)
        
        # GPU同期（最初のバッチのみ）
        if batch_idx == 0 and device.type == 'cuda':
            torch.cuda.synchronize()
            print(f"   First batch data moved to GPU: images {images.device}, shape {images.shape}")
        
        # Forward pass
        forward_start = time.time()
        predictions, grid_size = model(images)
        
        # 最初のバッチで予測形状確認
        if batch_idx == 0:
            print(f"   First batch predictions: shape {predictions.shape}, device {predictions.device}")
        
        # Loss calculation
        loss = criterion(predictions, targets, grid_size)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # 勾配クリッピング（安定性向上）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        # バッチ処理時間（最初の数バッチのみ）
        if batch_idx < 5:
            batch_time = time.time() - batch_start_time
            print(f"   Batch {batch_idx}: {batch_time:.3f}s, Loss: {loss.item():.4f}")
        
        # 進捗表示
        if batch_idx % Config.print_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f} LR: {current_lr:.6f}")
        
        # メモリリークチェック（GPU使用時のみ）
        if device.type == 'cuda' and batch_idx % 100 == 0:
            current_memory = torch.cuda.memory_allocated(0) / 1024**3
            if current_memory > 10.0:  # 10GB以上の場合
                torch.cuda.empty_cache()
                print(f"   GPU memory cleanup at batch {batch_idx}: {current_memory:.2f}GB")
    
    return total_loss / batch_count if batch_count > 0 else 0.0

if __name__ == "__main__":
    main()