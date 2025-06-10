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

import datetime
import hashlib
# ===== ver管理 =====
class VersionTracker:
    """スクリプトのバージョンと修正履歴を追跡"""
    _all_trackers = {}

    def __init__(self, script_name, version="1.0.0"):
        self.script_name = script_name
        self.version = version
        self.load_time = datetime.datetime.now()
        self.modifications = []
        
        VersionTracker._all_trackers[script_name] = self

    def add_modification(self, description, author="AI Assistant"):
        """修正履歴を追加"""
        timestamp = datetime.datetime.now()
        self.modifications.append({
            'timestamp': timestamp,
            'description': description,
            'author': author
        })
        
    def get_file_hash(self, filepath):
        """ファイルのハッシュ値を計算（変更検出用）"""
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()[:8]
        except:
            return "unknown"
    
    def print_version_info(self):
        """バージョン情報を表示"""
        print(f"\n{'='*60}")
        print(f"📋 {self.script_name} - Version {self.version}")
        print(f"⏰ Loaded: {self.load_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if hasattr(self, 'file_hash'):
            print(f"🔗 File Hash: {self.file_hash}")
        
        if self.modifications:
            print(f"📝 Recent Modifications ({len(self.modifications)}):")
            for i, mod in enumerate(self.modifications[-3:], 1):  # 最新3件
                print(f"   {i}. {mod['timestamp'].strftime('%H:%M:%S')} - {mod['description']}")
        
        print(f"{'='*60}\n")

    @staticmethod
    def print_all_versions():
        """プロジェクト全体のバージョン情報を一括表示"""
        if not VersionTracker._all_trackers:
            print("⚠️ バージョン管理対象のファイルが見つかりません")
            return
        
        print(f"\n{'='*80}")
        print(f"🚀 プロジェクト全体バージョン情報")
        print(f"⏰ 表示時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 管理対象ファイル数: {len(VersionTracker._all_trackers)}")
        print(f"{'='*80}")
        
        # 読み込み時刻順にソート
        sorted_trackers = sorted(
            VersionTracker._all_trackers.items(),
            key=lambda x: x[1].load_time
        )
        
        for i, (script_name, tracker) in enumerate(sorted_trackers, 1):
            print(f"\n{i}. 📄 {tracker.script_name}")
            print(f"   📌 Version: {tracker.version}")
            print(f"   ⏰ Loaded: {tracker.load_time.strftime('%H:%M:%S')}")
            
            if hasattr(tracker, 'file_hash') and tracker.file_hash:
                print(f"   🔗 Hash: {tracker.file_hash}")
            
            if tracker.modifications:
                latest_mod = tracker.modifications[-1]
                print(f"   📝 Latest: {latest_mod['timestamp'].strftime('%H:%M:%S')} - {latest_mod['description']}")
                if len(tracker.modifications) > 1:
                    print(f"   📋 Total modifications: {len(tracker.modifications)}")
            else:
                print(f"   📝 Modifications: None")
        
        print(f"\n{'='*80}")
        print(f"🎉 バージョン情報表示完了")
        print(f"{'='*80}\n")

    @staticmethod
    def print_version_summary():
        """コンパクトなバージョンサマリーを表示"""
        if not VersionTracker._all_trackers:
            print("⚠️ バージョン管理対象のファイルが見つかりません")
            return
        
        print(f"\n📊 プロジェクトバージョンサマリー ({len(VersionTracker._all_trackers)} files)")
        print("-" * 60)
        
        for script_name, tracker in VersionTracker._all_trackers.items():
            mod_count = len(tracker.modifications)
            latest_time = tracker.load_time.strftime('%H:%M:%S')
            print(f"📄 {tracker.script_name:<30} v{tracker.version:<8} ({mod_count} mods) {latest_time}")
        
        print("-" * 60)

# 各ファイル用のバージョントラッカーを作成
def create_version_tracker(script_name, filepath=None):
    """バージョントラッカーを作成"""
    tracker = VersionTracker(script_name)
    
    if filepath:
        tracker.file_hash = tracker.get_file_hash(filepath)
    
    return tracker

# バージョン管理システム初期化
training_version = create_version_tracker("Unified Training System v0.0", "dataset.py")
training_version.add_modification("プロトタイプ")

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

def main():
    print("🚀 Starting Modular YOLO Training")
    
    VersionTracker.print_all_versions()
    
    # 設定
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # データセット
    dataset = FLIRDataset(cfg.train_img_dir, cfg.train_label_dir, cfg.img_size)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, 
                          shuffle=True, collate_fn=collate_fn, 
                          num_workers=2, pin_memory=True)
    
    # モデル
    model = SimpleYOLO(cfg.num_classes).to(cfg.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 損失関数とオプティマイザ
    criterion = YOLOLoss(cfg.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    # 学習ループ
    best_loss = float('inf')
    for epoch in range(cfg.num_epochs):
        start_time = time.time()
        
        # 学習
        avg_loss = train_one_epoch(model, dataloader, criterion, optimizer, cfg.device)
        
        epoch_time = time.time() - start_time
        print(f"\nEpoch [{epoch+1}/{cfg.num_epochs}] "
              f"Loss: {avg_loss:.4f} Time: {epoch_time:.1f}s")
        
        # モデル保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_loss
            }, os.path.join(cfg.save_dir, 'best_model.pth'))
            print(f"💾 Best model saved (loss: {best_loss:.4f})")
    
    print("\n✅ Training completed!")

if __name__ == "__main__":
    main()