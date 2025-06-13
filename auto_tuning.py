# auto_hyperparameter_tuning.py - Phase 3用自動ハイパーパラメータチューニング

import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import time
from datetime import datetime
import json

# 既存のコンポーネントをインポート
from dataset import FLIRDataset, collate_fn
from multiscale_model import MultiScaleYOLO
from anchor_loss import MultiScaleAnchorLoss
from model import SimpleYOLO
from loss import YOLOLoss

class AutoTuner:
    """自動ハイパーパラメータチューニングクラス"""
    
    def __init__(self, base_config):
        self.base_config = base_config
        self.study_name = f"phase3_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.best_params = None
        self.best_val_loss = float('inf')
        
        # データセット準備（一度だけ）
        self.setup_dataset()
        
        print(f"🤖 AutoTuner initialized")
        print(f"   Study: {self.study_name}")
        print(f"   Dataset: {len(self.full_dataset)} images")
    
    def setup_dataset(self):
        """データセット準備"""
        self.full_dataset = FLIRDataset(
            self.base_config.train_img_dir,
            self.base_config.train_label_dir,
            self.base_config.img_size,
            augment=True
        )
        
        # 訓練・検証分割
        total_size = len(self.full_dataset)
        val_size = int(total_size * 0.15)
        train_size = total_size - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            self.full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    def create_dataloaders(self, batch_size):
        """データローダー作成"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def objective(self, trial):
        """Optuna目的関数"""
        print(f"\n🔍 Trial {trial.number} starting...")
        
        # ハイパーパラメータ提案
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 48, 64, 96]),
            'ema_decay': trial.suggest_float('ema_decay', 0.995, 0.9999),
            'gradient_clip': trial.suggest_float('gradient_clip', 0.5, 3.0),
            'architecture': trial.suggest_categorical('architecture', ['multiscale', 'simple']),
            'optimizer': trial.suggest_categorical('optimizer', ['AdamW', 'Adam']),
            'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'step', 'none'])
        }
        
        print(f"   Params: LR={params['learning_rate']:.6f}, BS={params['batch_size']}, "
              f"Arch={params['architecture']}")
        
        try:
            # 短時間学習で評価（5エポック）
            val_loss = self.train_and_evaluate(params, max_epochs=5)
            
            print(f"   Result: Val Loss = {val_loss:.4f}")
            
            # ベスト更新チェック
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_params = params.copy()
                print(f"   🎉 New best! Val Loss = {val_loss:.4f}")
            
            return val_loss
            
        except Exception as e:
            print(f"   ❌ Trial failed: {e}")
            return float('inf')
    
    def train_and_evaluate(self, params, max_epochs=5):
        """短時間学習&評価"""
        device = self.base_config.device
        
        # データローダー作成
        train_loader, val_loader = self.create_dataloaders(params['batch_size'])
        
        # モデル&損失関数作成
        if params['architecture'] == 'multiscale':
            anchors = {
                'small':  [(7, 11), (14, 28), (22, 65)],
                'medium': [(42, 35), (76, 67), (46, 126)],
                'large':  [(127, 117), (88, 235), (231, 218)]
            }
            model = MultiScaleYOLO(num_classes=15, anchors=anchors).to(device)
            criterion = MultiScaleAnchorLoss(anchors=anchors, num_classes=15)
        else:
            model = SimpleYOLO(num_classes=15).to(device)
            criterion = YOLOLoss(num_classes=15)
        
        # オプティマイザ
        if params['optimizer'] == 'AdamW':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
        
        # スケジューラ
        scheduler = None
        if params['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        elif params['scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        
        # EMA
        ema = EMAModel(model, decay=params['ema_decay'])
        
        # 短時間学習
        best_val_loss = float('inf')
        
        for epoch in range(max_epochs):
            # 訓練
            model.train()
            train_loss = 0
            train_batches = 0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = images.to(device)
                
                # Forward
                if params['architecture'] == 'multiscale':
                    predictions = model(images)
                    loss = criterion(predictions, targets)
                else:
                    predictions, grid_size = model(images)
                    loss = criterion(predictions, targets, grid_size)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient_clip'])
                optimizer.step()
                
                # EMA更新
                ema.update()
                
                train_loss += loss.item()
                train_batches += 1
                
                # 速度重視: 各エポック最大20バッチまで
                if batch_idx >= 19:
                    break
            
            avg_train_loss = train_loss / train_batches
            
            # 検証
            ema.apply_shadow()
            val_loss = self.validate_model(model, val_loader, criterion, device, params['architecture'])
            ema.restore()
            
            # スケジューラ更新
            if scheduler:
                scheduler.step()
            
            # ベスト更新
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            print(f"     Epoch {epoch+1}: Train {avg_train_loss:.3f}, Val {val_loss:.3f}")
            
            # Early termination for bad trials
            if val_loss > 100:  # 明らかに悪い場合は早期終了
                break
        
        # メモリクリーンアップ
        del model, optimizer, criterion
        torch.cuda.empty_cache()
        
        return best_val_loss
    
    def validate_model(self, model, val_loader, criterion, device, architecture):
        """モデル検証"""
        model.eval()
        total_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images = images.to(device)
                
                if architecture == 'multiscale':
                    predictions = model(images)
                    loss = criterion(predictions, targets)
                else:
                    predictions, grid_size = model(images)
                    loss = criterion(predictions, targets, grid_size)
                
                total_val_loss += loss.item()
                val_batches += 1
                
                # 速度重視: 最大10バッチまで
                if batch_idx >= 9:
                    break
        
        return total_val_loss / val_batches if val_batches > 0 else float('inf')
    
    def optimize(self, n_trials=20, timeout_hours=3):
        """最適化実行"""
        print(f"🚀 Starting hyperparameter optimization")
        print(f"   Trials: {n_trials}")
        print(f"   Timeout: {timeout_hours} hours")
        print(f"   Each trial: ~5 epochs (3-5 minutes)")
        
        # Optuna study作成
        study = optuna.create_study(
            direction='minimize',
            study_name=self.study_name,
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # 最適化実行
        start_time = time.time()
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout_hours * 3600
        )
        
        elapsed_time = time.time() - start_time
        
        # 結果表示
        self.print_results(study, elapsed_time)
        
        return study.best_params, study.best_value
    
    def print_results(self, study, elapsed_time):
        """結果表示"""
        print(f"\n🎉 Hyperparameter optimization completed!")
        print(f"⏱️  Total time: {elapsed_time/3600:.2f} hours")
        print(f"🔢 Trials completed: {len(study.trials)}")
        print(f"🏆 Best val loss: {study.best_value:.4f}")
        
        print(f"\n📊 Best parameters:")
        for key, value in study.best_params.items():
            print(f"   {key}: {value}")
        
        # 結果保存
        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'total_trials': len(study.trials),
            'elapsed_time_hours': elapsed_time / 3600,
            'timestamp': datetime.now().isoformat()
        }
        
        save_path = f"tuning_results_{self.study_name}.json"
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {save_path}")

# EMAクラス（簡易版）
class EMAModel:
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
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# 使用例
def run_auto_tuning():
    """自動チューニング実行"""
    from config import Config
    
    # ベース設定
    base_config = Config()
    
    # AutoTuner初期化
    tuner = AutoTuner(base_config)
    
    # 最適化実行（20試行、3時間制限）
    best_params, best_loss = tuner.optimize(n_trials=20, timeout_hours=3)
    
    print(f"\n🎯 Recommended configuration:")
    print(f"   Best Val Loss: {best_loss:.4f}")
    print(f"   Apply these to config.py:")
    
    for key, value in best_params.items():
        print(f"   {key} = {value}")
    
    return best_params, best_loss

if __name__ == "__main__":
    # 自動チューニング実行
    run_auto_tuning()
