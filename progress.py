# enhanced_progress_tracking.py - 詳細進捗表示機能

import time
import torch
from datetime import datetime, timedelta

class DetailedProgressTracker:
    """詳細な進捗追跡クラス"""
    
    def __init__(self, total_batches, print_interval=100):
        self.total_batches = total_batches
        self.print_interval = print_interval
        self.epoch_start_time = None
        self.batch_times = []
        self.recent_losses = []
        self.best_batch_loss = float('inf')
        self.worst_batch_loss = 0.0
        
    def start_epoch(self, epoch, total_epochs):
        """エポック開始"""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        self.epoch_start_time = time.time()
        self.batch_times = []
        self.recent_losses = []
        self.best_batch_loss = float('inf')
        self.worst_batch_loss = 0.0
        
    def update_batch(self, batch_idx, loss_value, current_lr):
        """バッチ更新"""
        current_time = time.time()
        
        # バッチ時間記録
        if len(self.batch_times) > 0:
            batch_time = current_time - self.last_batch_time
            self.batch_times.append(batch_time)
            
            # 最新20バッチの平均時間を保持
            if len(self.batch_times) > 20:
                self.batch_times = self.batch_times[-20:]
        
        self.last_batch_time = current_time
        
        # Loss統計更新
        self.recent_losses.append(loss_value)
        if len(self.recent_losses) > self.print_interval:
            self.recent_losses = self.recent_losses[-self.print_interval:]
            
        self.best_batch_loss = min(self.best_batch_loss, loss_value)
        self.worst_batch_loss = max(self.worst_batch_loss, loss_value)
        
        # 進捗表示
        if batch_idx % self.print_interval == 0:
            self.print_detailed_progress(batch_idx, loss_value, current_lr)
    
    def print_detailed_progress(self, batch_idx, current_loss, current_lr):
        """詳細進捗表示"""
        # 基本情報
        progress_pct = (batch_idx / self.total_batches) * 100
        
        # 時間統計
        elapsed_time = time.time() - self.epoch_start_time
        avg_batch_time = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
        remaining_batches = self.total_batches - batch_idx
        eta_seconds = remaining_batches * avg_batch_time if avg_batch_time > 0 else 0
        eta_time = datetime.now() + timedelta(seconds=eta_seconds)
        
        # Loss統計
        if len(self.recent_losses) > 1:
            avg_recent_loss = sum(self.recent_losses) / len(self.recent_losses)
            loss_trend = current_loss - self.recent_losses[0] if len(self.recent_losses) > 10 else 0
        else:
            avg_recent_loss = current_loss
            loss_trend = 0
            
        # GPU情報
        gpu_memory = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
        gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3 if torch.cuda.is_available() else 0
        
        # 進捗バー作成
        bar_length = 20
        filled_length = int(bar_length * progress_pct / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"\n" + "="*80)
        print(f"📊 Epoch {self.current_epoch}/{self.total_epochs} - Batch Progress")
        print(f"="*80)
        
        # メイン進捗情報
        print(f"🔄 Progress: [{bar}] {progress_pct:5.1f}% ({batch_idx:4d}/{self.total_batches})")
        print(f"⏱️  Time: {elapsed_time/60:5.1f}min elapsed, ETA: {eta_time.strftime('%H:%M:%S')}")
        print(f"💨 Speed: {avg_batch_time:5.2f}s/batch (recent avg)")
        
        # Loss情報
        trend_icon = "📈" if loss_trend > 0 else "📉" if loss_trend < 0 else "➡️"
        print(f"📊 Loss: {current_loss:8.4f} (current) | {avg_recent_loss:8.4f} (avg) {trend_icon}")
        print(f"   Range: {self.best_batch_loss:8.4f} (best) - {self.worst_batch_loss:8.4f} (worst)")
        
        # 学習情報
        print(f"⚙️  Learning Rate: {current_lr:.6f}")
        
        # GPU情報
        if torch.cuda.is_available():
            gpu_util_pct = (gpu_memory / 14.74) * 100  # T4の総容量
            memory_icon = "🟢" if gpu_util_pct < 70 else "🟡" if gpu_util_pct < 90 else "🔴"
            print(f"🖥️  GPU: {gpu_memory:5.2f}GB used ({gpu_util_pct:4.1f}%) {memory_icon}")
            print(f"   Reserved: {gpu_reserved:5.2f}GB")
        
        print(f"="*80)

class MultiScaleProgressTracker(DetailedProgressTracker):
    """マルチスケール学習用進捗トラッカー"""
    
    def __init__(self, total_batches, print_interval=100):
        super().__init__(total_batches, print_interval)
        self.scale_losses = {'small': [], 'medium': [], 'large': []}
        
    def update_batch_multiscale(self, batch_idx, loss_value, current_lr, scale_losses=None):
        """マルチスケール対応バッチ更新"""
        # 基本更新
        self.update_batch(batch_idx, loss_value, current_lr)
        
        # スケール別Loss記録
        if scale_losses:
            for scale, loss_info in scale_losses.items():
                if scale in self.scale_losses:
                    self.scale_losses[scale].append(loss_info.get('total', 0))
                    # 最新20件のみ保持
                    if len(self.scale_losses[scale]) > 20:
                        self.scale_losses[scale] = self.scale_losses[scale][-20:]
    
    def print_detailed_progress(self, batch_idx, current_loss, current_lr):
        """マルチスケール対応詳細進捗表示"""
        # 基本進捗表示
        super().print_detailed_progress(batch_idx, current_loss, current_lr)
        
        # スケール別Loss表示
        if any(losses for losses in self.scale_losses.values()):
            print(f"🎯 Scale-wise Loss (recent avg):")
            for scale, losses in self.scale_losses.items():
                if losses:
                    avg_loss = sum(losses[-10:]) / len(losses[-10:])  # 最新10件の平均
                    scale_icon = "🔍" if scale == 'small' else "🎯" if scale == 'medium' else "🔭"
                    print(f"   {scale_icon} {scale:6s}: {avg_loss:8.4f}")
            print(f"="*80)

def integrate_progress_tracker_to_training():
    """学習ループへの進捗トラッカー統合例"""
    
    # train_phase3_integrated.py の学習ループ内で使用
    code_example = '''
    # 学習ループ内での使用例
    
    # エポック開始時
    progress_tracker = MultiScaleProgressTracker(len(train_dataloader), print_interval=100)
    progress_tracker.start_epoch(epoch + 1, cfg.num_epochs)
    
    for batch_idx, (images, targets) in enumerate(train_dataloader):
        # ... 学習処理 ...
        
        # Forward & Loss計算
        if architecture_type == "multiscale":
            predictions = model(images)
            loss = criterion(predictions, targets)
            
            # マルチスケール進捗更新（詳細情報付き）
            if hasattr(criterion, 'return_components') and batch_idx % 100 == 0:
                # デバッグモードで詳細情報取得
                criterion.return_components = True
                _, _, scale_losses = criterion(predictions, targets)
                criterion.return_components = False
                
                progress_tracker.update_batch_multiscale(
                    batch_idx, loss.item(), current_lr, scale_losses
                )
            else:
                progress_tracker.update_batch(batch_idx, loss.item(), current_lr)
        else:
            # 従来版
            predictions, grid_size = model(images)
            loss = criterion(predictions, targets, grid_size)
            progress_tracker.update_batch(batch_idx, loss.item(), current_lr)
        
        # ... 残りの学習処理 ...
    '''
    
    return code_example

# テスト用
def demo_progress_tracker():
    """進捗トラッカーのデモ"""
    print("🧪 Progress Tracker Demo")
    
    # 模擬的な学習ループ
    total_batches = 50
    tracker = MultiScaleProgressTracker(total_batches, print_interval=10)
    tracker.start_epoch(1, 35)
    
    import random
    base_loss = 25.0
    
    for batch_idx in range(total_batches):
        # 模擬Loss（徐々に下降）
        loss_value = base_loss * (1 - batch_idx * 0.01) + random.uniform(-2, 2)
        current_lr = 0.0008
        
        # 模擬スケール別Loss
        scale_losses = {
            'small': {'total': loss_value * 0.3},
            'medium': {'total': loss_value * 0.4}, 
            'large': {'total': loss_value * 0.3}
        }
        
        tracker.update_batch_multiscale(batch_idx, loss_value, current_lr, scale_losses)
        
        # 少し待つ（リアルっぽく）
        time.sleep(0.1)

if __name__ == "__main__":
    # デモ実行
    demo_progress_tracker()
