# diagnostic_training.py - Fixed JSON serialization + enhanced diagnosis

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from datetime import datetime
import json

class DiagnosticTrainer:
    """学習過程を詳細に分析・診断するクラス（修正版）"""
    
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
        
        print(f"🔬 DiagnosticTrainer初期化完了（修正版）")
        print(f"   保存先: {save_dir}")
        print(f"   診断項目: 検出統計、信頼度分布、クラス性能、アンカー使用率")
        print(f"   修正点: JSON serialization, 偽検出対策")
    
    def start_epoch_diagnosis(self, epoch):
        """エポック開始時の診断準備"""
        self.current_epoch = epoch
        self.current_epoch_detections = []
        self.current_epoch_losses = []
        
        print(f"\n🔍 Epoch {epoch} 診断開始")
    
    def log_batch_diagnosis(self, batch_idx, images, targets, predictions, loss_components=None):
        """バッチごとの詳細診断（修正版）"""
        
        try:
            # 1. 検出数統計
            batch_detections = self._analyze_detections(predictions, targets)
            self.current_epoch_detections.extend(batch_detections)
            
            # 2. 損失成分記録（JSON対応）
            if loss_components:
                json_safe_components = self._make_json_safe(loss_components)
                for key, value in json_safe_components.items():
                    self.detection_stats['loss_components'][key].append(value)
            
            # 3. 問題サンプル特定（改良版）
            problematic = self._identify_problematic_samples(
                images, targets, predictions, batch_idx
            )
            self.problematic_samples.extend(problematic)
            
            # 4. リアルタイム警告（頻度調整）
            if batch_idx % 50 == 0:  # 20 → 50（ログ頻度削減）
                self._print_realtime_diagnosis(batch_idx, batch_detections)
                
        except Exception as e:
            # エラーハンドリング強化
            if batch_idx % 100 == 0:
                print(f"   ⚠️ 診断エラー (batch {batch_idx}): {str(e)[:100]}...")
    
    def _make_json_safe(self, obj):
        """オブジェクトをJSON安全な形式に変換"""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif torch.is_tensor(obj):
            return float(obj.detach().cpu().item()) if obj.numel() == 1 else obj.detach().cpu().tolist()
        else:
            return obj
    
    def _analyze_detections(self, predictions, targets):
        """検出結果の詳細分析（改良版）"""

        print(f"DEBUG: _analyze_detections に渡された predictions のタイプ: {type(predictions)}")
        if torch.is_tensor(predictions):
            print(f"DEBUG: _analyze_detections に渡された predictions の形状: {predictions.shape}")
        elif isinstance(predictions, dict):
            print(f"DEBUG: _analyze_detections に渡された predictions (dict) のキーと形状:")
            for k, v in predictions.items():
                print(f"  - {k}: {v.shape}")
        detections = []
        
        try:
            # アーキテクチャ判定を追加
            if isinstance(predictions, dict):
                # マルチスケールの場合：1つのスケールだけサンプリング
                # 全スケールを合計すると予測数が異常になるから
                sample_scale = 'medium'  # 代表として中サイズを使用
                if sample_scale in predictions:
                    scale_preds = predictions[sample_scale]
                    B, N, C = scale_preds.shape
                    # バッチの1枚目のみ
                    combined_preds = scale_preds[0].view(-1, C)  # [N, C]
                else:
                    # フォールバック
                    first_scale = list(predictions.keys())[0]
                    scale_preds = predictions[first_scale]
                    combined_preds = scale_preds[0].view(-1, scale_preds.shape[-1])
            else:
                # シングルスケールの場合
                combined_preds = predictions[0].view(-1, predictions.shape[-1])
            
            # 勾配デタッチして安全に処理
            with torch.no_grad():
                # 信頼度抽出
                confidences = torch.sigmoid(combined_preds[:, 4]).cpu().numpy()
                
                # クラス予測抽出
                class_probs = torch.softmax(combined_preds[:, 5:], dim=-1).cpu().numpy()
                class_ids = np.argmax(class_probs, axis=-1)
                
                # 異常値検出（信頼度1.0の検出）
                perfect_conf_count = np.sum(confidences >= 0.999)
                
                 # 予測数の妥当性チェック
                total_preds = len(confidences)
                expected_max = 2000  # バッチサイズ32 × 13×13グリッド程度が妥当
                
                if total_preds > expected_max:
                    print(f"⚠️ 異常な予測数検出: {total_preds} (期待値: < {expected_max})")
                    print(f"   → 診断システムの設定ミス、またはアーキテクチャ設定ミスの可能性")

                # 検出統計
                detection_info = {
                    'total_predictions': min(total_preds, expected_max),  # 上限設定
                    'conf_mean': float(np.mean(confidences)),
                    'conf_std': float(np.std(confidences)),
                    'conf_max': float(np.max(confidences)),
                    'conf_above_05': int(np.sum(confidences > 0.5)),
                    'conf_above_07': int(np.sum(confidences > 0.7)),
                    'conf_above_09': int(np.sum(confidences > 0.9)),
                    'perfect_conf_count': int(perfect_conf_count),  # 追加
                    'conf_distribution': np.histogram(confidences, bins=10, range=(0, 1))[0].tolist(),
                    'class_distribution': np.bincount(class_ids, minlength=15).tolist()
                }
            
            detections.append(detection_info)
            
        except Exception as e:
            print(f"   ⚠️ 検出分析エラー: {e}")
            # フォールバック統計
            detections.append({
                'total_predictions': 0,
                'conf_mean': 0.0,
                'conf_std': 0.0,
                'conf_max': 0.0,
                'conf_above_05': 0,
                'conf_above_07': 0,
                'conf_above_09': 0,
                'perfect_conf_count': 0,
                'conf_distribution': [0] * 10,
                'class_distribution': [0] * 15
            })
        
        return detections
    
    def _identify_problematic_samples(self, images, targets, predictions, batch_idx):
        """問題のあるサンプルを特定（改良版）"""
        problematic = []
        
        try:
            # バッチサイズを取得
            B = images.shape[0]
            
            for i in range(B):
                # この画像のターゲット数
                target_count = len(targets[i]) if len(targets) > i else 0
                
                # この画像の最大信頼度（勾配デタッチ）
                with torch.no_grad():
                    if isinstance(predictions, dict):
                        max_conf = 0
                        perfect_conf_count = 0
                        for scale_preds in predictions.values():
                            if i < scale_preds.shape[0]:
                                scale_conf = torch.sigmoid(scale_preds[i, :, 4]).cpu()
                                max_conf = max(max_conf, scale_conf.max().item())
                                perfect_conf_count += (scale_conf >= 0.999).sum().item()
                    else:
                        if i < predictions.shape[0]:
                            scale_conf = torch.sigmoid(predictions[i, :, 4]).cpu()
                            max_conf = scale_conf.max().item()
                            perfect_conf_count = (scale_conf >= 0.999).sum().item()
                        else:
                            max_conf = 0
                            perfect_conf_count = 0
                
                # 問題判定（改良版）
                is_problematic = False
                problem_type = []
                
                # 深刻な問題: ターゲットなしで完璧な信頼度
                if target_count == 0 and max_conf >= 0.999:
                    is_problematic = True
                    problem_type.append("perfect_confidence_without_targets")
                # 中程度の問題: ターゲットなしで高信頼度
                elif target_count == 0 and max_conf > 0.8:
                    is_problematic = True
                    problem_type.append("high_confidence_without_targets")
                # ターゲットありで低信頼度
                elif target_count > 0 and max_conf < 0.1:
                    is_problematic = True
                    problem_type.append("low_confidence_with_targets")
                # 多ターゲットで低信頼度
                elif target_count > 3 and max_conf < 0.3:
                    is_problematic = True
                    problem_type.append("many_targets_low_confidence")
                
                if is_problematic:
                    problematic.append({
                        'batch_idx': int(batch_idx),
                        'sample_idx': int(i),
                        'target_count': int(target_count),
                        'max_confidence': float(max_conf),
                        'perfect_conf_count': int(perfect_conf_count),  # 追加
                        'problem_types': problem_type
                    })
                    
        except Exception as e:
            print(f"   ⚠️ 問題サンプル分析エラー: {e}")
        
        return problematic
    
    def _print_realtime_diagnosis(self, batch_idx, batch_detections):
        """リアルタイム診断情報を表示（改良版）"""
        if not batch_detections:
            return
        
        latest = batch_detections[-1]
        
        print(f"📊 Batch {batch_idx} 診断:")
        print(f"   予測数: {latest['total_predictions']:,}")
        print(f"   平均信頼度: {latest['conf_mean']:.3f}")
        print(f"   最大信頼度: {latest['conf_max']:.3f}")
        print(f"   高信頼度(>0.5): {latest['conf_above_05']}")
        print(f"   超高信頼度(>0.7): {latest['conf_above_07']}")
        
        # 新しい警告: 完璧な信頼度検出
        if latest['perfect_conf_count'] > 0:
            print(f"   🚨 完璧信頼度検出: {latest['perfect_conf_count']} (要注意)")
        
        # 警告
        if latest['conf_max'] < 0.1:
            print(f"   ⚠️ 警告: 最大信頼度が異常に低い ({latest['conf_max']:.3f})")
        if latest['conf_above_05'] == 0:
            print(f"   ⚠️ 警告: 信頼度0.5以上の検出がゼロ")
        if latest['perfect_conf_count'] > 100:
            print(f"   🚨 重要警告: 完璧信頼度検出が多すぎる (偽検出の可能性)")
    
    def end_epoch_diagnosis(self, epoch, val_loss, model=None):
        """エポック終了時の総合診断（修正版）"""
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
        
        # 4. 統計保存（JSON安全版）
        try:
            self._save_epoch_statistics(epoch, val_loss)
        except Exception as e:
            print(f"   ⚠️ 統計保存エラー: {e}")
        
        # 5. 可視化（エラー耐性）
        if epoch % 5 == 0:  # 5エポックごと
            try:
                self._create_diagnostic_plots(epoch)
            except Exception as e:
                print(f"   ⚠️ 可視化エラー: {e}")
        
        return suggestions
    
    def _summarize_detection_stats(self, epoch):
        """検出統計のサマリー（改良版）"""
        detections = self.current_epoch_detections
        
        # 全バッチの統計を集約
        total_preds = sum(d['total_predictions'] for d in detections)
        avg_conf = np.mean([d['conf_mean'] for d in detections])
        max_conf = max(d['conf_max'] for d in detections)
        total_high_conf = sum(d['conf_above_05'] for d in detections)
        total_super_conf = sum(d['conf_above_07'] for d in detections)
        total_perfect_conf = sum(d['perfect_conf_count'] for d in detections)  # 追加
        
        print(f"🎯 検出統計サマリー:")
        print(f"   総予測数: {total_preds:,}")
        print(f"   平均信頼度: {avg_conf:.3f}")
        print(f"   最大信頼度: {max_conf:.3f}")
        print(f"   高信頼度検出: {total_high_conf} ({100*total_high_conf/total_preds:.2f}%)")
        print(f"   超高信頼度検出: {total_super_conf} ({100*total_super_conf/total_preds:.2f}%)")
        
        # 新しい統計: 完璧信頼度
        if total_perfect_conf > 0:
            print(f"   🚨 完璧信頼度検出: {total_perfect_conf} ({100*total_perfect_conf/total_preds:.2f}%)")
        
        # エポック統計として記録（JSON安全）
        epoch_stats = {
            'epoch': int(epoch),
            'total_predictions': int(total_preds),
            'avg_confidence': float(avg_conf),
            'max_confidence': float(max_conf),
            'high_conf_detections': int(total_high_conf),
            'super_conf_detections': int(total_super_conf),
            'perfect_conf_detections': int(total_perfect_conf),  # 追加
            'high_conf_rate': float(total_high_conf / total_preds) if total_preds > 0 else 0.0
        }
        
        self.detection_stats['epoch_stats'].append(epoch_stats)
    
    def _analyze_problematic_samples(self):
        """問題サンプルの分析（改良版）"""
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
        
        # 最も深刻な問題（完璧信頼度）
        perfect_conf_problems = [
            s for s in self.problematic_samples 
            if 'perfect_confidence_without_targets' in s['problem_types']
        ]
        
        if perfect_conf_problems:
            print(f"   🚨 重大問題（完璧信頼度）: {len(perfect_conf_problems)}件")
            print(f"      → 損失関数の重み調整が必要")
        
        # 最も問題の大きなサンプル
        worst_samples = sorted(
            self.problematic_samples, 
            key=lambda x: x['max_confidence'] if 'low_confidence' in str(x['problem_types']) else -x['max_confidence']
        )[:3]
        
        print(f"   最重要問題サンプル:")
        for i, sample in enumerate(worst_samples, 1):
            print(f"     {i}. Batch {sample['batch_idx']}, Sample {sample['sample_idx']}")
            print(f"        Target数: {sample['target_count']}, 最大信頼度: {sample['max_confidence']:.3f}")
            if 'perfect_conf_count' in sample:
                print(f"        完璧信頼度検出: {sample['perfect_conf_count']}")
            print(f"        問題: {', '.join(sample['problem_types'])}")
    
    def _generate_improvement_suggestions(self, val_loss):
        """改善提案を生成（改良版）"""
        suggestions = []
        
        # 最新統計
        if self.detection_stats['epoch_stats']:
            latest = self.detection_stats['epoch_stats'][-1]
            
            # 完璧信頼度問題（最優先）
            if latest.get('perfect_conf_detections', 0) > 1000:
                suggestions.append({
                    'type': 'critical',
                    'issue': '完璧信頼度検出が異常に多い（偽検出）',
                    'suggestion': 'lambda_noobj を2倍に増加、信頼度損失の重み強化'
                })
            
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
        """統計情報をJSONで保存（修正版）"""
        stats_file = os.path.join(self.save_dir, f"epoch_{epoch}_stats.json")
        
        # JSON安全なデータ作成
        save_data = {
            'epoch': int(epoch),
            'val_loss': float(val_loss),
            'timestamp': datetime.now().isoformat(),
            'detection_stats': self.detection_stats['epoch_stats'][-1] if self.detection_stats['epoch_stats'] else {},
            'problematic_samples_count': len(self.problematic_samples),
            'loss_components': {}
        }
        
        # 損失成分（最新10件、JSON安全化）
        for k, v in self.detection_stats['loss_components'].items():
            if v:
                safe_values = [self._make_json_safe(val) for val in v[-10:]]
                save_data['loss_components'][k] = safe_values
        
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"   💾 統計保存成功: {stats_file}")
        except Exception as e:
            print(f"   ❌ 統計保存失敗: {e}")
    
    def _create_diagnostic_plots(self, epoch):
        """診断用グラフを作成（エラー耐性版）"""
        if not self.detection_stats['epoch_stats']:
            return
        
        try:
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
            
            # 3. 完璧信頼度検出（新規追加）
            perfect_confs = [s.get('perfect_conf_detections', 0) for s in self.detection_stats['epoch_stats']]
            if any(perfect_confs):
                axes[1, 0].plot(epochs, perfect_confs, 'r-', linewidth=2)
                axes[1, 0].set_title('Perfect Confidence Detections (WARNING)')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 損失成分推移
            if self.detection_stats['loss_components']:
                for loss_type, values in self.detection_stats['loss_components'].items():
                    if values and len(values) > 1:
                        safe_values = [float(v) for v in values[-20:] if v is not None]
                        if safe_values:
                            axes[1, 1].plot(safe_values, label=loss_type, linewidth=2)
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
            
        except Exception as e:
            print(f"   ⚠️ 可視化エラー: {e}")
    
    def generate_final_report(self):
        """最終診断レポートを生成（修正版）"""
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
        
        # 偽検出の警告
        perfect_conf = last_epoch.get('perfect_conf_detections', 0)
        if perfect_conf > 0:
            print(f"   🚨 完璧信頼度検出: {perfect_conf} (偽検出の可能性)")
        
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
        if perfect_conf > 1000:
            print(f"   🚨 緊急: 偽検出対策（lambda_noobj増加）")
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


# ===== 統合用の関数（修正版） =====

def integrate_diagnostic_training(original_training_function):
    """既存の学習関数に診断機能を統合（修正版）"""
    
    def enhanced_training_function(model, train_dataloader, val_dataloader, criterion, cfg, architecture_type):
        """診断機能付き学習関数（エラー耐性強化）"""
        
        # 診断器初期化
        try:
            diagnostics = DiagnosticTrainer(
                save_dir=os.path.join(cfg.save_dir, "diagnostics")
            )
            
            print(f"🔬 診断機能統合版学習開始（修正版）")
            print(f"   診断ログ: {diagnostics.save_dir}")
            print(f"   修正点: JSON serialization, 偽検出対策")
            
            # 元の学習ループをラップ
            return original_training_function(
                model, train_dataloader, val_dataloader, criterion, cfg, architecture_type,
                diagnostics=diagnostics  # 診断器を渡す
            )
            
        except Exception as e:
            print(f"❌ 診断システム初期化エラー: {e}")
            print("   → 診断機能なしで学習継続")
            return original_training_function(
                model, train_dataloader, val_dataloader, criterion, cfg, architecture_type
            )
    
    return enhanced_training_function


# ===== 偽検出対策用設定調整関数 =====

def suggest_config_fixes(diagnostic_results):
    """診断結果に基づく設定修正提案"""
    
    suggestions = {
        'config_changes': {},
        'priority': 'medium',
        'explanation': ''
    }
    
    if not diagnostic_results:
        return suggestions
    
    latest_stats = diagnostic_results.get('epoch_stats', [])
    if not latest_stats:
        return suggestions
    
    latest = latest_stats[-1]
    
    # 偽検出（完璧信頼度）対策
    perfect_conf = latest.get('perfect_conf_detections', 0)
    total_preds = latest.get('total_predictions', 1)
    perfect_rate = perfect_conf / total_preds if total_preds > 0 else 0
    
    if perfect_rate > 0.01:  # 1%以上が完璧信頼度
        suggestions['config_changes'].update({
            'lambda_noobj': 1.0,  # 0.3 → 1.0 (背景への罰則強化)
            'lambda_obj': 1.5,    # 2.0 → 1.5 (物体信頼度を相対的に下げる)
            'learning_rate': 3e-4,  # 5e-4 → 3e-4 (学習を安定化)
        })
        suggestions['priority'] = 'critical'
        suggestions['explanation'] = f'偽検出率{perfect_rate:.1%}は危険レベル。背景罰則を強化。'
    
    # 低信頼度対策
    max_conf = latest.get('max_confidence', 0)
    if max_conf < 0.3:
        suggestions['config_changes'].update({
            'learning_rate': 8e-4,  # より積極的な学習率
            'lambda_coord': 15.0,   # 座標学習をさらに強化
        })
        suggestions['priority'] = 'high'
        suggestions['explanation'] += f' 最大信頼度{max_conf:.3f}は低すぎる。学習強化が必要。'
    
    return suggestions


# ===== テスト用診断関数（修正版） =====

def quick_diagnosis_test():
    """診断システムの動作確認（修正版）"""
    print("🧪 診断システムテスト開始（修正版）")
    
    try:
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
        
        for batch_idx in range(3):  # 5 → 3（テスト短縮）
            diagnostics.log_batch_diagnosis(
                batch_idx, dummy_images, dummy_targets, dummy_predictions
            )
        
        suggestions = diagnostics.end_epoch_diagnosis(1, 45.0)
        
        print(f"✅ 診断システムテスト完了（修正版）")
        print(f"   提案数: {len(suggestions)}")
        for suggestion in suggestions:
            print(f"   {suggestion['type']}: {suggestion['issue']}")
        
        # JSON保存テスト
        test_config = suggest_config_fixes({'epoch_stats': diagnostics.detection_stats['epoch_stats']})
        print(f"   設定提案: {test_config['config_changes']}")
        
        return True
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


# ===== 簡易メトリクス計算（修正版） =====

def calculate_simple_metrics(model, val_loader, max_batches=10):
    """簡易的な性能メトリクスを計算（エラー耐性強化）"""
    model.eval()
    
    metrics = {
        'val_loss': [],
        'detection_rate': [],
        'confidence_stats': {
            'mean': [], 'max': [], 'above_0.5': [], 'above_0.7': [], 'perfect_count': []
        },
        'per_image_detections': []
    }
    
    print(f"📊 簡易メトリクス計算開始 (最大{max_batches}バッチ)")
    
    with torch.no_grad():
        try:
            for batch_idx, (images, targets) in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                
                # 予測
                predictions = model(images)
                
                # 信頼度統計（エラー耐性）
                try:
                    if isinstance(predictions, dict):
                        all_confs = []
                        for scale_preds in predictions.values():
                            confs = torch.sigmoid(scale_preds[..., 4]).cpu().numpy().flatten()
                            all_confs.extend(confs)
                    else:
                        all_confs = torch.sigmoid(predictions[..., 4]).cpu().numpy().flatten()
                    
                    # メトリクス計算
                    metrics['confidence_stats']['mean'].append(float(np.mean(all_confs)))
                    metrics['confidence_stats']['max'].append(float(np.max(all_confs)))
                    metrics['confidence_stats']['above_0.5'].append(int(np.sum(np.array(all_confs) > 0.5)))
                    metrics['confidence_stats']['above_0.7'].append(int(np.sum(np.array(all_confs) > 0.7)))
                    metrics['confidence_stats']['perfect_count'].append(int(np.sum(np.array(all_confs) >= 0.999)))
                    
                    # 画像あたりの高信頼度検出数
                    high_conf_per_image = np.sum(np.array(all_confs) > 0.5) / images.shape[0]
                    metrics['per_image_detections'].append(float(high_conf_per_image))
                    
                except Exception as e:
                    print(f"   ⚠️ バッチ{batch_idx}でエラー: {e}")
                    continue
                
                if batch_idx % 3 == 0:
                    avg_conf = np.mean(all_confs) if 'all_confs' in locals() else 0
                    high_conf_count = np.sum(np.array(all_confs) > 0.5) if 'all_confs' in locals() else 0
                    print(f"   Batch {batch_idx}: 平均信頼度={avg_conf:.3f}, "
                          f"高信頼度検出={high_conf_count}")
        
        except Exception as e:
            print(f"❌ メトリクス計算エラー: {e}")
            return None, None
    
    # サマリー統計
    try:
        summary = {
            'avg_confidence': float(np.mean(metrics['confidence_stats']['mean'])) if metrics['confidence_stats']['mean'] else 0.0,
            'max_confidence': float(np.max(metrics['confidence_stats']['max'])) if metrics['confidence_stats']['max'] else 0.0,
            'total_high_conf': int(np.sum(metrics['confidence_stats']['above_0.5'])),
            'total_super_conf': int(np.sum(metrics['confidence_stats']['above_0.7'])),
            'total_perfect_conf': int(np.sum(metrics['confidence_stats']['perfect_count'])),
            'avg_detections_per_image': float(np.mean(metrics['per_image_detections'])) if metrics['per_image_detections'] else 0.0
        }
        
        print(f"\n📋 簡易メトリクス結果:")
        print(f"   平均信頼度: {summary['avg_confidence']:.3f}")
        print(f"   最大信頼度: {summary['max_confidence']:.3f}")
        print(f"   高信頼度検出: {summary['total_high_conf']}")
        print(f"   超高信頼度検出: {summary['total_super_conf']}")
        print(f"   完璧信頼度検出: {summary['total_perfect_conf']}")
        print(f"   平均検出数/画像: {summary['avg_detections_per_image']:.1f}")
        
        # 警告
        if summary['total_perfect_conf'] > 100:
            print(f"   🚨 警告: 完璧信頼度検出が多すぎます（偽検出の可能性）")
        
        return summary, metrics
        
    except Exception as e:
        print(f"❌ サマリー計算エラー: {e}")
        return None, None


if __name__ == "__main__":
    # 診断システムのテスト実行
    success = quick_diagnosis_test()
    
    if success:
        print("🎉 診断システム準備完了（修正版）!")
        print("   修正点:")
        print("     ✅ JSON serialization エラー修正")
        print("     ✅ 偽検出（完璧信頼度）の検出・対策")
        print("     ✅ エラー耐性強化")
        print("     ✅ 設定修正提案機能")
        print("   次: train.pyに統合して安定した診断を開始")
    else:
        print("❌ 診断システムエラー - 修正が必要")