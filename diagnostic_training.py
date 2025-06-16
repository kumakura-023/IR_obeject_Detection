# diagnostic_training.py - 診断的学習システム

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from datetime import datetime
import json

class DiagnosticTrainer:
    """学習過程を詳細に分析・診断するクラス"""
    
    def __init__(self, save_dir="diagnostic_logs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 統計情報収集
        self.detection_stats = {
            'epoch_stats': [],
            'loss_components': defaultdict(list),
            'confidence_distribution': [],
            'class_performance': defaultdict(list),
            'anchor_usage': defaultdict(list),
            'size_distribution': defaultdict(list)
        }
        
        # リアルタイム診断用
        self.current_epoch_detections = []
        self.problematic_samples = []
        
        print(f"🔬 DiagnosticTrainer初期化完了")
        print(f"   保存先: {save_dir}")
        print(f"   診断項目: 検出統計、信頼度分布、クラス性能、アンカー使用率")
    
    def start_epoch_diagnosis(self, epoch):
        """エポック開始時の診断準備"""
        self.current_epoch = epoch
        self.current_epoch_detections = []
        self.current_epoch_losses = []
        
        print(f"\n🔍 Epoch {epoch} 診断開始")
    
    def log_batch_diagnosis(self, batch_idx, images, targets, predictions, loss_components=None):
        """バッチごとの詳細診断"""
        
        # 1. 検出数統計
        batch_detections = self._analyze_detections(predictions, targets)
        self.current_epoch_detections.extend(batch_detections)
        
        # 2. 損失成分記録
        if loss_components:
            for key, value in loss_components.items():
                self.detection_stats['loss_components'][key].append(value)
        
        # 3. 問題サンプル特定（信頼度が異常に低い）
        problematic = self._identify_problematic_samples(
            images, targets, predictions, batch_idx
        )
        self.problematic_samples.extend(problematic)
        
        # 4. リアルタイム警告
        if batch_idx % 20 == 0:
            self._print_realtime_diagnosis(batch_idx, batch_detections)
    
    def _analyze_detections(self, predictions, targets):
        """検出結果の詳細分析"""
        detections = []
        
        # マルチスケール予測の場合
        if isinstance(predictions, dict):
            all_preds = []
            for scale_name, scale_preds in predictions.items():
                # [B, N, 20] -> [B*N, 20]
                B, N, C = scale_preds.shape
                flat_preds = scale_preds.view(-1, C)
                all_preds.append(flat_preds)
            
            # 全予測を結合
            combined_preds = torch.cat(all_preds, dim=0)
        else:
            combined_preds = predictions.view(-1, predictions.shape[-1])
        
        # 信頼度抽出（勾配デタッチ）
        confidences = torch.sigmoid(combined_preds[:, 4]).detach().cpu().numpy()
        
        # クラス予測抽出（勾配デタッチ）
        class_probs = torch.softmax(combined_preds[:, 5:], dim=-1).detach().cpu().numpy()
        class_ids = np.argmax(class_probs, axis=-1)
        
        # 検出統計
        detection_info = {
            'total_predictions': len(confidences),
            'conf_mean': np.mean(confidences),
            'conf_std': np.std(confidences),
            'conf_max': np.max(confidences),
            'conf_above_05': np.sum(confidences > 0.5),
            'conf_above_07': np.sum(confidences > 0.7),
            'conf_above_09': np.sum(confidences > 0.9),
            'conf_distribution': np.histogram(confidences, bins=10, range=(0, 1))[0],
            'class_distribution': np.bincount(class_ids, minlength=15)
        }
        
        detections.append(detection_info)
        return detections
    
    def _identify_problematic_samples(self, images, targets, predictions, batch_idx):
        """問題のあるサンプルを特定"""
        problematic = []
        
        # バッチサイズを取得
        B = images.shape[0]
        
        for i in range(B):
            # この画像のターゲット数
            target_count = len(targets[i]) if len(targets) > i else 0
            
            # この画像の最大信頼度（勾配デタッチ）
            if isinstance(predictions, dict):
                max_conf = 0
                for scale_preds in predictions.values():
                    scale_conf = torch.sigmoid(scale_preds[i, :, 4]).detach().max().item()
                    max_conf = max(max_conf, scale_conf)
            else:
                max_conf = torch.sigmoid(predictions[i, :, 4]).detach().max().item()
            
            # 問題判定
            is_problematic = False
            problem_type = []
            
            if target_count > 0 and max_conf < 0.1:
                is_problematic = True
                problem_type.append("low_confidence_with_targets")
            
            if target_count == 0 and max_conf > 0.8:
                is_problematic = True
                problem_type.append("high_confidence_without_targets")
            
            if target_count > 3 and max_conf < 0.3:
                is_problematic = True
                problem_type.append("many_targets_low_confidence")
            
            if is_problematic:
                problematic.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'target_count': target_count,
                    'max_confidence': max_conf,
                    'problem_types': problem_type
                })
        
        return problematic
    
    def _print_realtime_diagnosis(self, batch_idx, batch_detections):
        """リアルタイム診断情報を表示"""
        if not batch_detections:
            return
        
        latest = batch_detections[-1]
        
        print(f"📊 Batch {batch_idx} 診断:")
        print(f"   予測数: {latest['total_predictions']:,}")
        print(f"   平均信頼度: {latest['conf_mean']:.3f}")
        print(f"   最大信頼度: {latest['conf_max']:.3f}")
        print(f"   高信頼度(>0.5): {latest['conf_above_05']}")
        print(f"   超高信頼度(>0.7): {latest['conf_above_07']}")
        
        # 警告
        if latest['conf_max'] < 0.1:
            print(f"   ⚠️ 警告: 最大信頼度が異常に低い ({latest['conf_max']:.3f})")
        if latest['conf_above_05'] == 0:
            print(f"   ⚠️ 警告: 信頼度0.5以上の検出がゼロ")
    
    def end_epoch_diagnosis(self, epoch, val_loss, model=None):
        """エポック終了時の総合診断"""
        print(f"\n📋 Epoch {epoch} 総合診断")
        print("=" * 60)
        
        # 1. 検出統計サマリー
        if self.current_epoch_detections:
            self._summarize_detection_stats(epoch)
        
        # 2. 問題サンプル分析
        if self.problematic_samples:
            self._analyze_problematic_samples()
        
        # 3. 改善提案
        suggestions = self._generate_improvement_suggestions(val_loss)
        
        # 4. 統計保存
        self._save_epoch_statistics(epoch, val_loss)
        
        # 5. 可視化
        if epoch % 5 == 0:  # 5エポックごと
            self._create_diagnostic_plots(epoch)
        
        return suggestions
    
    def _summarize_detection_stats(self, epoch):
        """検出統計のサマリー"""
        detections = self.current_epoch_detections
        
        # 全バッチの統計を集約
        total_preds = sum(d['total_predictions'] for d in detections)
        avg_conf = np.mean([d['conf_mean'] for d in detections])
        max_conf = max(d['conf_max'] for d in detections)
        total_high_conf = sum(d['conf_above_05'] for d in detections)
        total_super_conf = sum(d['conf_above_07'] for d in detections)
        
        print(f"🎯 検出統計サマリー:")
        print(f"   総予測数: {total_preds:,}")
        print(f"   平均信頼度: {avg_conf:.3f}")
        print(f"   最大信頼度: {max_conf:.3f}")
        print(f"   高信頼度検出: {total_high_conf} ({100*total_high_conf/total_preds:.2f}%)")
        print(f"   超高信頼度検出: {total_super_conf} ({100*total_super_conf/total_preds:.2f}%)")
        
        # エポック統計として記録
        epoch_stats = {
            'epoch': epoch,
            'total_predictions': total_preds,
            'avg_confidence': avg_conf,
            'max_confidence': max_conf,
            'high_conf_detections': total_high_conf,
            'super_conf_detections': total_super_conf,
            'high_conf_rate': total_high_conf / total_preds if total_preds > 0 else 0
        }
        
        self.detection_stats['epoch_stats'].append(epoch_stats)
    
    def _analyze_problematic_samples(self):
        """問題サンプルの分析"""
        if not self.problematic_samples:
            print(f"✅ 問題サンプル: なし")
            return
        
        print(f"⚠️ 問題サンプル分析: {len(self.problematic_samples)}件")
        
        # 問題タイプ別集計
        problem_types = defaultdict(int)
        for sample in self.problematic_samples:
            for prob_type in sample['problem_types']:
                problem_types[prob_type] += 1
        
        print(f"   問題タイプ別:")
        for prob_type, count in problem_types.items():
            print(f"     {prob_type}: {count}件")
        
        # 最も問題の大きなサンプル
        worst_samples = sorted(
            self.problematic_samples, 
            key=lambda x: x['max_confidence'] if 'low_confidence' in str(x['problem_types']) else -x['max_confidence']
        )[:3]
        
        print(f"   最重要問題サンプル:")
        for i, sample in enumerate(worst_samples, 1):
            print(f"     {i}. Batch {sample['batch_idx']}, Sample {sample['sample_idx']}")
            print(f"        Target数: {sample['target_count']}, 最大信頼度: {sample['max_confidence']:.3f}")
            print(f"        問題: {', '.join(sample['problem_types'])}")
    
    def _generate_improvement_suggestions(self, val_loss):
        """改善提案を生成"""
        suggestions = []
        
        # 最新統計
        if self.detection_stats['epoch_stats']:
            latest = self.detection_stats['epoch_stats'][-1]
            
            # 信頼度が低すぎる
            if latest['max_confidence'] < 0.3:
                suggestions.append({
                    'type': 'critical',
                    'issue': '最大信頼度が異常に低い',
                    'suggestion': '学習率を2倍に増加、またはアンカーサイズ見直し'
                })
            
            # 高信頼度検出がほぼない
            if latest['high_conf_rate'] < 0.01:
                suggestions.append({
                    'type': 'important',
                    'issue': '高信頼度検出が1%未満',
                    'suggestion': 'loss重みのobj項を増加、データ拡張を軽減'
                })
            
            # Val Lossが高止まり
            if val_loss > 40:
                suggestions.append({
                    'type': 'important',
                    'issue': 'Val Loss高止まり',
                    'suggestion': '学習率スケジューリング見直し、正則化軽減'
                })
        
        # 問題サンプルが多い
        if len(self.problematic_samples) > 20:
            suggestions.append({
                'type': 'warning',
                'issue': '問題サンプルが多すぎる',
                'suggestion': 'データ品質確認、アノテーション見直し'
            })
        
        return suggestions
    
    def _save_epoch_statistics(self, epoch, val_loss):
        """統計情報をJSONで保存"""
        stats_file = os.path.join(self.save_dir, f"epoch_{epoch}_stats.json")
        
        save_data = {
            'epoch': epoch,
            'val_loss': val_loss,
            'timestamp': datetime.now().isoformat(),
            'detection_stats': self.detection_stats['epoch_stats'][-1] if self.detection_stats['epoch_stats'] else {},
            'problematic_samples_count': len(self.problematic_samples),
            'loss_components': {k: v[-10:] for k, v in self.detection_stats['loss_components'].items()}  # 最新10件
        }
        
        with open(stats_file, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def _create_diagnostic_plots(self, epoch):
        """診断用グラフを作成"""
        if not self.detection_stats['epoch_stats']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Diagnostic Plots - Epoch {epoch}', fontsize=16)
        
        # データ準備
        epochs = [s['epoch'] for s in self.detection_stats['epoch_stats']]
        avg_confs = [s['avg_confidence'] for s in self.detection_stats['epoch_stats']]
        max_confs = [s['max_confidence'] for s in self.detection_stats['epoch_stats']]
        high_conf_rates = [s['high_conf_rate'] for s in self.detection_stats['epoch_stats']]
        
        # 1. 信頼度推移
        axes[0, 0].plot(epochs, avg_confs, 'b-', label='Average', linewidth=2)
        axes[0, 0].plot(epochs, max_confs, 'r-', label='Maximum', linewidth=2)
        axes[0, 0].axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Target (0.5)')
        axes[0, 0].set_title('Confidence Trends')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Confidence')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 高信頼度検出率
        axes[0, 1].plot(epochs, [r*100 for r in high_conf_rates], 'g-', linewidth=2)
        axes[0, 1].set_title('High Confidence Detection Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Rate (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 最新の信頼度分布
        if self.current_epoch_detections:
            latest_dist = self.current_epoch_detections[-1]['conf_distribution']
            bin_edges = np.linspace(0, 1, 11)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            axes[1, 0].bar(bin_centers, latest_dist, width=0.08, alpha=0.7)
            axes[1, 0].set_title('Current Confidence Distribution')
            axes[1, 0].set_xlabel('Confidence')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 損失成分推移
        if self.detection_stats['loss_components']:
            for loss_type, values in self.detection_stats['loss_components'].items():
                if values:
                    axes[1, 1].plot(values[-20:], label=loss_type, linewidth=2)  # 最新20件
            axes[1, 1].set_title('Loss Components')
            axes[1, 1].set_xlabel('Recent Batches')
            axes[1, 1].set_ylabel('Loss Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        plot_file = os.path.join(self.save_dir, f"diagnostic_plots_epoch_{epoch}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 診断グラフ保存: {plot_file}")
    
    def generate_final_report(self):
        """最終診断レポートを生成"""
        print(f"\n{'='*80}")
        print(f"📋 最終診断レポート")
        print(f"{'='*80}")
        
        if not self.detection_stats['epoch_stats']:
            print("⚠️ 統計データが不足しています")
            return
        
        # 全体的な傾向
        first_epoch = self.detection_stats['epoch_stats'][0]
        last_epoch = self.detection_stats['epoch_stats'][-1]
        
        print(f"🔍 学習進捗分析:")
        print(f"   エポック数: {first_epoch['epoch']} → {last_epoch['epoch']}")
        print(f"   平均信頼度: {first_epoch['avg_confidence']:.3f} → {last_epoch['avg_confidence']:.3f}")
        print(f"   最大信頼度: {first_epoch['max_confidence']:.3f} → {last_epoch['max_confidence']:.3f}")
        print(f"   高信頼度率: {first_epoch['high_conf_rate']:.1%} → {last_epoch['high_conf_rate']:.1%}")
        
        # 改善度評価
        conf_improvement = last_epoch['max_confidence'] - first_epoch['max_confidence']
        rate_improvement = last_epoch['high_conf_rate'] - first_epoch['high_conf_rate']
        
        print(f"\n📈 改善度評価:")
        if conf_improvement > 0.1:
            print(f"   ✅ 信頼度改善: +{conf_improvement:.3f} (良好)")
        elif conf_improvement > 0.05:
            print(f"   🟡 信頼度改善: +{conf_improvement:.3f} (普通)")
        else:
            print(f"   ❌ 信頼度改善: +{conf_improvement:.3f} (要改善)")
        
        if rate_improvement > 0.01:
            print(f"   ✅ 検出率改善: +{rate_improvement:.1%} (良好)")
        elif rate_improvement > 0.005:
            print(f"   🟡 検出率改善: +{rate_improvement:.1%} (普通)")
        else:
            print(f"   ❌ 検出率改善: +{rate_improvement:.1%} (要改善)")
        
        # 最終推奨事項
        print(f"\n🎯 最終推奨事項:")
        if last_epoch['max_confidence'] < 0.3:
            print(f"   🚨 緊急: 学習率を2-3倍に増加")
            print(f"   🚨 緊急: アンカーサイズの全面見直し")
        elif last_epoch['max_confidence'] < 0.5:
            print(f"   ⚠️ 重要: 損失関数の重み調整")
            print(f"   ⚠️ 重要: データ拡張の軽減")
        else:
            print(f"   ✅ 順調: 現在の設定を継続")
            print(f"   💡 提案: より長期的な学習を検討")
        
        print(f"{'='*80}\n")


# 統合用の関数
def integrate_diagnostic_training(original_training_function):
    """既存の学習関数に診断機能を統合"""
    
    def enhanced_training_function(model, train_dataloader, val_dataloader, criterion, cfg, architecture_type):
        """診断機能付き学習関数"""
        
        # 診断器初期化
        diagnostics = DiagnosticTrainer(
            save_dir=os.path.join(cfg.save_dir, "diagnostics")
        )
        
        print(f"🔬 診断機能統合版学習開始")
        print(f"   診断ログ: {diagnostics.save_dir}")
        
        # 元の学習ループをラップ
        return original_training_function(
            model, train_dataloader, val_dataloader, criterion, cfg, architecture_type,
            diagnostics=diagnostics  # 診断器を渡す
        )
    
    return enhanced_training_function


# テスト用診断関数
def quick_diagnosis_test():
    """診断システムの動作確認"""
    print("🧪 診断システムテスト開始")
    
    # ダミーデータで診断機能をテスト
    diagnostics = DiagnosticTrainer("test_diagnostic_logs")
    
    # ダミー予測とターゲット
    dummy_predictions = {
        'small': torch.randn(2, 100, 20),
        'medium': torch.randn(2, 50, 20), 
        'large': torch.randn(2, 25, 20)
    }
    dummy_targets = [
        torch.tensor([[0, 0.5, 0.5, 0.1, 0.1]]),
        torch.tensor([[1, 0.3, 0.7, 0.2, 0.3]])
    ]
    dummy_images = torch.randn(2, 1, 416, 416)
    
    # エポック1のテスト
    diagnostics.start_epoch_diagnosis(1)
    
    for batch_idx in range(5):
        diagnostics.log_batch_diagnosis(
            batch_idx, dummy_images, dummy_targets, dummy_predictions
        )
    
    suggestions = diagnostics.end_epoch_diagnosis(1, 45.0)
    
    print(f"✅ 診断システムテスト完了")
    print(f"   提案数: {len(suggestions)}")
    for suggestion in suggestions:
        print(f"   {suggestion['type']}: {suggestion['issue']}")
    
    return True


# 簡易メトリクス計算
def calculate_simple_metrics(model, val_loader, max_batches=10):
    """簡易的な性能メトリクスを計算"""
    model.eval()
    
    metrics = {
        'val_loss': [],
        'detection_rate': [],
        'confidence_stats': {
            'mean': [], 'max': [], 'above_0.5': [], 'above_0.7': []
        },
        'per_image_detections': []
    }
    
    print(f"📊 簡易メトリクス計算開始 (最大{max_batches}バッチ)")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            
            # 予測
            predictions = model(images)
            
            # 信頼度統計
            if isinstance(predictions, dict):
                all_confs = []
                for scale_preds in predictions.values():
                    confs = torch.sigmoid(scale_preds[..., 4]).cpu().numpy().flatten()
                    all_confs.extend(confs)
            else:
                all_confs = torch.sigmoid(predictions[..., 4]).cpu().numpy().flatten()
            
            # メトリクス計算
            metrics['confidence_stats']['mean'].append(np.mean(all_confs))
            metrics['confidence_stats']['max'].append(np.max(all_confs))
            metrics['confidence_stats']['above_0.5'].append(np.sum(np.array(all_confs) > 0.5))
            metrics['confidence_stats']['above_0.7'].append(np.sum(np.array(all_confs) > 0.7))
            
            # 画像あたりの高信頼度検出数
            high_conf_per_image = np.sum(np.array(all_confs) > 0.5) / images.shape[0]
            metrics['per_image_detections'].append(high_conf_per_image)
            
            if batch_idx % 3 == 0:
                print(f"   Batch {batch_idx}: 平均信頼度={np.mean(all_confs):.3f}, "
                      f"高信頼度検出={np.sum(np.array(all_confs) > 0.5)}")
    
    # サマリー統計
    summary = {
        'avg_confidence': np.mean(metrics['confidence_stats']['mean']),
        'max_confidence': np.max(metrics['confidence_stats']['max']),
        'total_high_conf': np.sum(metrics['confidence_stats']['above_0.5']),
        'total_super_conf': np.sum(metrics['confidence_stats']['above_0.7']),
        'avg_detections_per_image': np.mean(metrics['per_image_detections'])
    }
    
    print(f"\n📋 簡易メトリクス結果:")
    print(f"   平均信頼度: {summary['avg_confidence']:.3f}")
    print(f"   最大信頼度: {summary['max_confidence']:.3f}")
    print(f"   高信頼度検出: {summary['total_high_conf']}")
    print(f"   超高信頼度検出: {summary['total_super_conf']}")
    print(f"   平均検出数/画像: {summary['avg_detections_per_image']:.1f}")
    
    return summary, metrics


if __name__ == "__main__":
    # 診断システムのテスト実行
    success = quick_diagnosis_test()
    
    if success:
        print("🎉 診断システム準備完了!")
        print("   次: train.pyに統合して詳細診断を開始")
    else:
        print("❌ 診断システムエラー - 修正が必要")