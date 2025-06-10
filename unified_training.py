# unified_training_corrected.py - 最新プロジェクト構成に対応

import os
import gc
import math
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

# PyTorch AMPの互換性設定
if torch.__version__ >= '2.0':
    from torch.amp import autocast, GradScaler
    AMP_DEVICE = 'cuda'
else:
    from torch.cuda.amp import autocast, GradScaler
    AMP_DEVICE = None

# ===== 修正: 最新のファイル名でimport =====
from dataset import YoloInfraredDataset
from efficientnet_model import create_efficientnet_model
# ★★★ 修正: `build_targets` のみを import するように変更 ★★★
from unified_targets import (
    build_targets,
    get_default_anchors,
    analyze_dataset_statistics,
    compare_anchor_sets,
    prepare_anchor_grid_info
)
from unified_loss import create_enhanced_loss


import datetime
import hashlib

class VersionTracker:
    """スクリプトのバージョンと修正履歴を追跡"""
    
    def __init__(self, script_name, version="1.0.0"):
        self.script_name = script_name
        self.version = version
        self.load_time = datetime.datetime.now()
        self.modifications = []
        
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

# 各ファイル用のバージョントラッカーを作成
def create_version_tracker(script_name, filepath=None):
    """バージョントラッカーを作成"""
    tracker = VersionTracker(script_name)
    
    if filepath:
        tracker.file_hash = tracker.get_file_hash(filepath)
    
    return tracker

# バージョン管理システム初期化
training_version = create_version_tracker("Unified Training System v2.6", "unified_training.py")
training_version.add_modification("バッチサイズを20→8に削減（メモリ不足対策）")
training_version.add_modification("積極的メモリクリーンアップ機能追加")
training_version.add_modification("float32統一でAMP完全無効化")
training_version.add_modification("OOMエラー自動回復機能追加")


# ===== デバッグシステム =====
class LossDebugTracker:
    """ロス計算の全工程を追跡するデバッグシステム"""
    
    def __init__(self):
        self.step_history = []
        self.focal_loss_history = []
        self.target_history = []
        self.prediction_history = []
        self.weight_history = []
        
    def track_step(self, step, batch_idx, epoch, preds, targets, loss_dict, loss_fn):
        """各ステップの詳細情報を記録"""
        
        debug_info = {
            'step': step,
            'batch_idx': batch_idx,
            'epoch': epoch,
            'timestamp': time.time()
        }
        
        # 1. ターゲット分析
        target_obj = targets['objectness']
        debug_info['target_analysis'] = {
            'total_targets': target_obj.numel(),
            'positive_001': (target_obj > 0.001).sum().item(),
            'positive_01': (target_obj > 0.01).sum().item(),
            'positive_1': (target_obj > 0.1).sum().item(),
            'positive_5': (target_obj > 0.5).sum().item(),
            'max_objectness': target_obj.max().item(),
            'mean_positive': target_obj[target_obj > 0.01].mean().item() if (target_obj > 0.01).any() else 0.0,
            'min_positive': target_obj[target_obj > 0.01].min().item() if (target_obj > 0.01).any() else 0.0
        }
        
        # 2. 予測分析
        pred_obj = preds[..., 4]
        pred_obj_sigmoid = torch.sigmoid(pred_obj)
        debug_info['prediction_analysis'] = {
            'pred_obj_logits_mean': pred_obj.mean().item(),
            'pred_obj_logits_std': pred_obj.std().item(),
            'pred_obj_sigmoid_mean': pred_obj_sigmoid.mean().item(),
            'pred_obj_sigmoid_std': pred_obj_sigmoid.std().item(),
            'pred_obj_sigmoid_max': pred_obj_sigmoid.max().item(),
            'pred_obj_sigmoid_min': pred_obj_sigmoid.min().item()
        }
        
        # 3. ロス関数内部状態の追跡
        if hasattr(loss_fn, 'obj_loss_fn') and hasattr(loss_fn.obj_loss_fn, 'gamma'):
            debug_info['focal_loss_state'] = {
                'gamma': getattr(loss_fn.obj_loss_fn, 'gamma', 'N/A'),
                'alpha': getattr(loss_fn.obj_loss_fn, 'alpha', 'N/A'),
                'step_count': getattr(loss_fn.obj_loss_fn, 'step_count', 'N/A'),
                'loss_history_length': len(getattr(loss_fn.obj_loss_fn, 'loss_history', []))
            }
        
        # 4. 動的重み
        if 'dynamic_weights' in loss_dict:
            debug_info['dynamic_weights'] = loss_dict['dynamic_weights']
        
        # 5. ロス値
        debug_info['loss_values'] = {
            'total': loss_dict['total'].item(),
            'box': loss_dict['box'].item(),
            'obj': loss_dict['obj'].item(),
            'cls': loss_dict['cls'].item(),
            'pos_samples': loss_dict['pos_samples']
        }
        
        # 6. 手動ロス計算（検証用）
        debug_info['manual_loss_check'] = self._manual_loss_calculation(preds, targets, loss_fn)
        
        self.step_history.append(debug_info)
        
        # 異常検出
        if step > 5:  # 最初の数ステップはスキップ
            self._detect_anomalies(debug_info, step)
    
    def _manual_loss_calculation(self, preds, targets, loss_fn):
        """手動でロス計算を実行して検証"""
        try:
            device = preds.device
            pred_obj = preds[..., 4]
            target_obj = targets['objectness']
            
            # AdaptiveFocalLossのパラメータを取得
            if hasattr(loss_fn, 'obj_loss_fn'):
                obj_loss_fn = loss_fn.obj_loss_fn
                alpha = getattr(obj_loss_fn, 'alpha', 0.75)
                gamma = getattr(obj_loss_fn, 'gamma', 1.5)
            else:
                alpha, gamma = 0.75, 1.5
            
            # バイナリターゲット作成（loss_fn の処理を模倣）
            obj_target_for_loss = target_obj.clone()
            obj_target_for_loss = torch.where(
                obj_target_for_loss > 0.01,
                obj_target_for_loss,
                torch.zeros_like(obj_target_for_loss)
            )
            obj_target_binary = (obj_target_for_loss > 0.3).float()
            
            # 手動BCE計算
            bce_loss = F.binary_cross_entropy_with_logits(pred_obj, obj_target_binary, reduction='none')
            
            # 手動Focal Loss計算
            probs = torch.sigmoid(pred_obj)
            pt = torch.where(obj_target_binary == 1, probs, 1 - probs)
            alpha_t = torch.where(obj_target_binary == 1, alpha, 1 - alpha)
            focal_loss = alpha_t * (1 - pt) ** gamma * bce_loss
            
            return {
                'binary_targets_sum': obj_target_binary.sum().item(),
                'manual_bce_mean': bce_loss.mean().item(),
                'manual_focal_mean': focal_loss.mean().item(),
                'pt_mean': pt.mean().item(),
                'alpha_t_mean': alpha_t.mean().item(),
                'gamma_used': gamma,
                'alpha_used': alpha
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _detect_anomalies(self, current_info, step):
        """異常値を検出してアラート"""
        current_obj_loss = current_info['loss_values']['obj']
        
        # 過去5ステップの平均と比較
        if len(self.step_history) >= 5:
            recent_obj_losses = [info['loss_values']['obj'] for info in self.step_history[-5:]]
            avg_recent = sum(recent_obj_losses) / len(recent_obj_losses)
            
            # 急激な減少を検出
            if current_obj_loss < avg_recent * 0.1 and current_obj_loss < 0.01:
                print(f"\n🚨 ANOMALY DETECTED at Step {step}!")
                print(f"   Obj Loss dropped from {avg_recent:.6f} to {current_obj_loss:.6f}")
                print(f"   Investigating cause...")
                
                self._investigate_cause(current_info, step)
    
    def _investigate_cause(self, current_info, step):
        """異常の原因を調査"""
        print(f"\n🔍 INVESTIGATING ANOMALY at Step {step}:")
        
        # 1. Focal Loss パラメータチェック
        if 'focal_loss_state' in current_info:
            focal_state = current_info['focal_loss_state']
            print(f"   Focal Loss State:")
            print(f"     Gamma: {focal_state.get('gamma', 'N/A')}")
            print(f"     Alpha: {focal_state.get('alpha', 'N/A')}")
            print(f"     Step Count: {focal_state.get('step_count', 'N/A')}")
        
        # 2. ターゲット変化チェック
        target_analysis = current_info['target_analysis']
        print(f"   Target Analysis:")
        print(f"     Positive targets (>0.01): {target_analysis['positive_01']}")
        print(f"     Mean positive objectness: {target_analysis['mean_positive']:.6f}")
        
        # 3. 手動計算との比較
        if 'manual_loss_check' in current_info:
            manual = current_info['manual_loss_check']
            if 'error' not in manual:
                actual_obj_loss = current_info['loss_values']['obj']
                manual_focal_loss = manual['manual_focal_mean']
                print(f"   Manual vs Actual Loss:")
                print(f"     Manual Focal: {manual_focal_loss:.6f}")
                print(f"     Actual Obj: {actual_obj_loss:.6f}")
                print(f"     Difference: {abs(manual_focal_loss - actual_obj_loss):.6f}")
                print(f"     Binary targets: {manual['binary_targets_sum']}")
                print(f"     Gamma used: {manual['gamma_used']}")
        
        # 4. 動的重みチェック
        if 'dynamic_weights' in current_info:
            weights = current_info['dynamic_weights']
            print(f"   Dynamic Weights:")
            print(f"     Box: {weights['box']:.2f}, Obj: {weights['obj']:.2f}, Cls: {weights['cls']:.2f}")
    
    def save_debug_log(self, filename="loss_debug_log.json"):
        """デバッグログを保存"""
        try:
            # NumPy/Tensorを通常のPython型に変換
            json_safe_log = []
            for entry in self.step_history:
                safe_entry = {}
                for key, value in entry.items():
                    if isinstance(value, dict):
                        safe_entry[key] = {k: float(v) if isinstance(v, (torch.Tensor, float)) else v 
                                         for k, v in value.items()}
                    else:
                        safe_entry[key] = float(value) if isinstance(value, (torch.Tensor, float)) else value
                json_safe_log.append(safe_entry)
            
            with open(filename, 'w') as f:
                json.dump(json_safe_log, f, indent=2)
            print(f"📝 Debug log saved to {filename}")
        except Exception as e:
            print(f"⚠️ Failed to save debug log: {e}")
    
    def print_summary(self):
        """デバッグサマリーを表示"""
        if len(self.step_history) == 0:
            return
        
        print(f"\n📊 LOSS DEBUG SUMMARY ({len(self.step_history)} steps tracked)")
        
        # Obj Lossの推移
        obj_losses = [info['loss_values']['obj'] for info in self.step_history]
        print(f"   Obj Loss: {obj_losses[0]:.6f} → {obj_losses[-1]:.6f}")
        
        # 最大値と最小値
        max_obj = max(obj_losses)
        min_obj = min(obj_losses)
        print(f"   Range: {min_obj:.6f} - {max_obj:.6f}")
        
        # 急激な変化を検出
        for i in range(1, len(obj_losses)):
            if obj_losses[i] < obj_losses[i-1] * 0.1 and obj_losses[i] < 0.01:
                print(f"   🚨 Sharp drop detected at step {self.step_history[i]['step']}")
                break

def check_for_nan_gradients(model, step):
    """モデル全体の勾配にNaNが含まれているかチェックする"""
    nan_found = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"🚨🚨🚨 NaN GRADIENT DETECTED IN ==> {name} at step {step}")
            nan_found = True
    if nan_found:
        # NaNが見つかった場合、デバッガを起動する（Colabで有効）
        import pdb; pdb.set_trace()
    return nan_found

class RootCauseInvestigator:
    """予測値崩壊の根本原因を調査"""
    
    def __init__(self):
        self.weight_history = []
        self.gradient_history = []
        self.activation_history = []
        self.loss_component_history = []
        
    def investigate_prediction_collapse(self, model, preds, targets, loss_dict, 
                                      optimizer, step, batch_idx):
        """予測値崩壊の根本原因を調査"""
        
        print(f"\n🔬 [ROOT CAUSE INVESTIGATION] Step {step}, Batch {batch_idx}")
        
        # 1. モデル重みの異常調査
        self._investigate_model_weights(model, step)
        
        # 2. 勾配の異常調査
        self._investigate_gradients(model, step)
        
        # 3. アクティベーション調査
        self._investigate_activations(model, preds, step)
        
        # 4. ロス成分の寄与度調査
        self._investigate_loss_components(preds, targets, loss_dict, step)
    
    def _investigate_model_weights(self, model, step):
        """モデル重みの異常を調査"""
        print("   🔍 Model Weights Analysis:")
        
        # Detection Headの重み（最も影響が大きい）
        detection_head = model.head.detection_head
        weight_data = detection_head.weight.data
        bias_data = detection_head.bias.data if detection_head.bias is not None else None
        
        # Objectness出力に対応する重み（channel 4）
        # detection_head: [num_anchors * (5+num_classes), in_channels, 1, 1]
        num_anchors = 3
        num_classes = 15
        outputs_per_anchor = 5 + num_classes  # 20
        
        obj_weight_indices = []
        for anchor_idx in range(num_anchors):
            obj_idx = anchor_idx * outputs_per_anchor + 4  # objectness位置
            obj_weight_indices.append(obj_idx)
        
        obj_weights = weight_data[obj_weight_indices]  # [3, 256, 1, 1]
        obj_biases = bias_data[obj_weight_indices] if bias_data is not None else None
        
        weight_stats = {
            'obj_weight_mean': obj_weights.mean().item(),
            'obj_weight_std': obj_weights.std().item(),
            'obj_weight_max': obj_weights.max().item(),
            'obj_weight_min': obj_weights.min().item(),
            'obj_bias_mean': obj_biases.mean().item() if obj_biases is not None else 0.0,
            'obj_bias_std': obj_biases.std().item() if obj_biases is not None else 0.0
        }
        
        self.weight_history.append({'step': step, **weight_stats})
        
        print(f"     Obj weights: mean={weight_stats['obj_weight_mean']:.6f}, "
              f"std={weight_stats['obj_weight_std']:.6f}")
        print(f"     Obj biases: mean={weight_stats['obj_bias_mean']:.6f}, "
              f"std={weight_stats['obj_bias_std']:.6f}")
        
        # 異常検出
        if abs(weight_stats['obj_weight_mean']) > 10.0:
            print(f"     🚨 WEIGHT EXPLOSION: obj_weight_mean = {weight_stats['obj_weight_mean']:.6f}")
        elif abs(weight_stats['obj_weight_mean']) < 1e-6:
            print(f"     🚨 WEIGHT VANISHING: obj_weight_mean = {weight_stats['obj_weight_mean']:.6f}")
        
        if abs(weight_stats['obj_bias_mean']) > 50.0:
            print(f"     🚨 BIAS EXPLOSION: obj_bias_mean = {weight_stats['obj_bias_mean']:.6f}")
        elif abs(weight_stats['obj_bias_mean']) < -50.0:
            print(f"     🚨 LARGE NEGATIVE BIAS: obj_bias_mean = {weight_stats['obj_bias_mean']:.6f}")
            print(f"       → This could cause sigmoid collapse!")
    
    def _investigate_gradients(self, model, step):
        """勾配の異常を調査"""
        print("   🔍 Gradient Analysis:")
        
        detection_head = model.head.detection_head
        if detection_head.weight.grad is not None:
            grad_data = detection_head.weight.grad.data
            
            # Objectness勾配
            num_anchors = 3
            num_classes = 15
            outputs_per_anchor = 5 + num_classes
            
            obj_grad_indices = []
            for anchor_idx in range(num_anchors):
                obj_idx = anchor_idx * outputs_per_anchor + 4
                obj_grad_indices.append(obj_idx)
            
            obj_grads = grad_data[obj_grad_indices]
            
            grad_stats = {
                'obj_grad_mean': obj_grads.mean().item(),
                'obj_grad_std': obj_grads.std().item(),
                'obj_grad_max': obj_grads.max().item(),
                'obj_grad_min': obj_grads.min().item(),
                'obj_grad_norm': obj_grads.norm().item()
            }
            
            self.gradient_history.append({'step': step, **grad_stats})
            
            print(f"     Obj gradients: mean={grad_stats['obj_grad_mean']:.6f}, "
                  f"norm={grad_stats['obj_grad_norm']:.6f}")
            
            # 勾配異常検出
            if grad_stats['obj_grad_norm'] > 100.0:
                print(f"     🚨 GRADIENT EXPLOSION: norm = {grad_stats['obj_grad_norm']:.6f}")
            elif grad_stats['obj_grad_norm'] < 1e-8:
                print(f"     🚨 GRADIENT VANISHING: norm = {grad_stats['obj_grad_norm']:.6f}")
        else:
            print("     ⚠️ No gradients found (may be due to gradient accumulation)")
    
    def _investigate_activations(self, model, preds, step):
        """アクティベーション調査"""
        print("   🔍 Activation Analysis:")
        
        # Objectness logits (sigmoid適用前)
        obj_logits = preds[..., 4]  # [B, N]
        
        activation_stats = {
            'obj_logits_mean': obj_logits.mean().item(),
            'obj_logits_std': obj_logits.std().item(),
            'obj_logits_max': obj_logits.max().item(),
            'obj_logits_min': obj_logits.min().item(),
            'obj_logits_median': obj_logits.median().item()
        }
        
        self.activation_history.append({'step': step, **activation_stats})
        
        print(f"     Obj logits: mean={activation_stats['obj_logits_mean']:.6f}, "
              f"min={activation_stats['obj_logits_min']:.6f}, "
              f"max={activation_stats['obj_logits_max']:.6f}")
        
        # Sigmoid崩壊の判定
        if activation_stats['obj_logits_max'] < -10.0:
            print(f"     🚨 SIGMOID SATURATION (negative): max_logit = {activation_stats['obj_logits_max']:.6f}")
            print(f"       → sigmoid({activation_stats['obj_logits_max']:.2f}) ≈ {torch.sigmoid(torch.tensor(activation_stats['obj_logits_max'])).item():.8f}")
        elif activation_stats['obj_logits_min'] > 10.0:
            print(f"     🚨 SIGMOID SATURATION (positive): min_logit = {activation_stats['obj_logits_min']:.6f}")
    
    def _investigate_loss_components(self, preds, targets, loss_dict, step):
        """ロス成分の寄与度調査"""
        print("   🔍 Loss Component Analysis:")
        
        # 各ロス成分の大きさ
        total_loss = loss_dict['total'].item()
        box_loss = loss_dict['box'].item()
        obj_loss = loss_dict['obj'].item()
        cls_loss = loss_dict['cls'].item()
        
        # 重み付き寄与度
        if 'dynamic_weights' in loss_dict:
            weights = loss_dict['dynamic_weights']
            w_box, w_obj, w_cls = weights['box'], weights['obj'], weights['cls']
            
            weighted_box = w_box * box_loss
            weighted_obj = w_obj * obj_loss
            weighted_cls = w_cls * cls_loss
            
            total_weighted = weighted_box + weighted_obj + weighted_cls
            
            if total_weighted > 0:
                box_contribution = weighted_box / total_weighted
                obj_contribution = weighted_obj / total_weighted
                cls_contribution = weighted_cls / total_weighted
                
                print(f"     Loss contributions:")
                print(f"       Box: {box_contribution:.3f} ({weighted_box:.6f})")
                print(f"       Obj: {obj_contribution:.3f} ({weighted_obj:.6f})")
                print(f"       Cls: {cls_contribution:.3f} ({weighted_cls:.6f})")
                
                # 異常寄与度検出
                if obj_contribution > 0.8:
                    print(f"     🚨 OBJ LOSS DOMINANCE: {obj_contribution:.3f}")
                    print(f"       → Objectness loss is overwhelming other components")
                elif obj_contribution < 0.05:
                    print(f"     🚨 OBJ LOSS TOO WEAK: {obj_contribution:.3f}")


# ===== 修正されたデバッグ関数 =====
def debug_loss_calculation_comprehensive(preds, targets, loss_fn, model, optimizer, 
                                       batch_idx, epoch, tracker: LossDebugTracker, 
                                       step_count, root_investigator: RootCauseInvestigator):
    """包括的ロス計算デバッグ - 修正版"""
    
    # 1. 通常のロス計算を実行
    loss_dict = loss_fn(preds, targets)
    
    # 2. トラッカーに記録
    tracker.track_step(step_count, batch_idx, epoch, preds, targets, loss_dict, loss_fn)
    
    # 3. 予測値異常時に根本原因調査
    pred_obj_sigmoid = torch.sigmoid(preds[..., 4])
    sigmoid_mean = pred_obj_sigmoid.mean().item()
    
    if sigmoid_mean < 0.001 or step_count in [0, 1, 2, 10, 20]:  # 異常時 + 特定ステップ
        root_investigator.investigate_prediction_collapse(
            model, preds, targets, loss_dict, optimizer, step_count, batch_idx
        )
    
    # 4. 最初の数ステップとピンポイントでの詳細出力
    if step_count <= 3 or (batch_idx in [10, 20, 30] and epoch == 0):
        print(f"\n🔬 [COMPREHENSIVE DEBUG] Step {step_count}, Batch {batch_idx}")
        
        # 直前の記録を表示
        if tracker.step_history:
            latest = tracker.step_history[-1]
            
            print(f"   Target Summary:")
            target_info = latest['target_analysis']
            print(f"     Positive (>0.01): {target_info['positive_01']}")
            print(f"     Positive (>0.1): {target_info['positive_1']}")
            print(f"     Mean positive: {target_info['mean_positive']:.6f}")
            
            print(f"   Prediction Summary:")
            pred_info = latest['prediction_analysis']
            print(f"     Sigmoid mean: {pred_info['pred_obj_sigmoid_mean']:.6f}")
            print(f"     Sigmoid std: {pred_info['pred_obj_sigmoid_std']:.6f}")
            
            if 'focal_loss_state' in latest:
                focal_info = latest['focal_loss_state']
                print(f"   Focal Loss State:")
                print(f"     Gamma: {focal_info['gamma']}")
                print(f"     Alpha: {focal_info['alpha']}")
                print(f"     Internal step: {focal_info['step_count']}")
            
            print(f"   Loss Values:")
            loss_info = latest['loss_values']
            print(f"     Total: {loss_info['total']:.6f}")
            print(f"     Box: {loss_info['box']:.6f}")
            print(f"     Obj: {loss_info['obj']:.6f}")
            print(f"     Cls: {loss_info['cls']:.6f}")
            
            if 'manual_loss_check' in latest and 'error' not in latest['manual_loss_check']:
                manual_info = latest['manual_loss_check']
                print(f"   Manual Check:")
                print(f"     Manual focal: {manual_info['manual_focal_mean']:.6f}")
                print(f"     Difference: {abs(manual_info['manual_focal_mean'] - loss_info['obj']):.6f}")
    
    return loss_dict


# ===== 既存のクラスと関数（unified_training_gm.pyから） =====
class TrainingConfig:
    """学習設定を一元管理"""
    def __init__(self):
        # 基本設定
        self.batch_size = 8  # ★★★ 20 → 8 に削減 ★★★
        self.num_classes = 15
        self.epochs = 30
        self.input_size = (640, 512)  # (W, H)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 学習率設定
        self.initial_lr = 1e-3
        self.max_lr = 1e-3
        self.min_lr = 1e-6
        self.warmup_steps = 500

        # Loss重み（初期値）
        self.loss_weights = {
            'box': 5.0,
            'obj': 2.0,
            'cls': 1.0
        }
        
        # Gradient Accumulation（バッチサイズ削減の補償）
        self.accumulation_steps = 8  # ★★★ 4 → 8 に増加 ★★★
        self.effective_batch_size = self.batch_size * self.accumulation_steps  # 8 × 8 = 64
        
        # パス設定
        self.project_root = "/content/drive/MyDrive/EfficientNet_Project"
        self.train_img_dir = "/content/FLIR_YOLO_local/images/train"
        self.train_label_dir = "/content/FLIR_YOLO_local/labels/train"
        self.model_save_path = os.path.join(self.project_root, "saved_models")
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # 進捗表示設定
        self.progress_interval = 10
        self.memory_check_interval = 10  # ★★★ 50 → 10 に短縮 ★★★
        
        # アンカー設定
        self.anchor_threshold = 0.2  # IoU閾値
        self.num_anchors = 9  # 総アンカー数

        print(f"🔧 Memory-optimized config:")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Accumulation steps: {self.accumulation_steps}")
        print(f"   Effective batch size: {self.effective_batch_size}")
        print(f"   Memory check interval: {self.memory_check_interval}")


class WarmRestartScheduler:
    """Cosine Annealing with Warm Restarts"""
    def __init__(self, optimizer, T_0=10, T_mult=2, eta_min=1e-6, verbose=True):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.eta_max = optimizer.param_groups[0]['lr']
        self.verbose = verbose
        
        self.T_cur = 0
        self.T_i = T_0
        self.restart_count = 0
        self.total_steps = 0
        
        self._set_lr(self.eta_max)
    
    def _set_lr(self, lr):
        """オプティマイザの学習率を設定"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        lr = self.eta_min + (self.eta_max - self.eta_min) * \
             (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
        return lr
    
    def step(self):
        self.total_steps += 1
        self.T_cur += 1
        
        new_lr = self.get_lr()
        self._set_lr(new_lr)
        
        if self.T_cur >= self.T_i:
            self.restart_count += 1
            if self.verbose:
                print(f"\n🔄 Learning rate warm restart #{self.restart_count}")
                print(f"   Next cycle: {int(self.T_i * self.T_mult)} steps")
            
            self.T_cur = 0
            self.T_i = int(self.T_i * self.T_mult)
        
        return new_lr


def get_memory_usage() -> Dict[str, float]:
    """メモリ使用量を取得"""
    memory_info = {}
    
    if torch.cuda.is_available():
        memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3
        memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3
    
    memory_info['cpu_percent'] = 0.0  # psutilなしでシンプルに
    
    return memory_info

# ★★★ 積極的メモリクリーンアップ関数 ★★★
def aggressive_memory_cleanup():
    """積極的なメモリクリーンアップ"""
    import gc
    
    # Python ガベージコレクション
    collected = gc.collect()
    
    # CUDA メモリクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 現在のメモリ使用量を取得
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"🧹 Memory cleanup: {collected} objects | GPU: {allocated:.1f}GB alloc, {reserved:.1f}GB reserved")
    
    return allocated if torch.cuda.is_available() else 0

def progressive_unfreezing(model: nn.Module, epoch: int) -> str:
    """段階的レイヤー解凍"""
    total_layers = len(list(model.backbone.parameters()))
    
    if epoch < 3:
        # Stage 1: Backbone完全凍結
        for param in model.backbone.parameters():
            param.requires_grad = False
        stage = "Head+Neck Only"
    elif epoch < 8:
        # Stage 2: 後半50%を解凍
        for i, param in enumerate(model.backbone.parameters()):
            param.requires_grad = i >= total_layers // 2
        stage = "Backbone 50% + Head+Neck"
    elif epoch < 15:
        # Stage 3: 後半75%を解凍
        for i, param in enumerate(model.backbone.parameters()):
            param.requires_grad = i >= total_layers // 4
        stage = "Backbone 75% + Head+Neck"
    else:
        # Stage 4: 全レイヤー解凍
        for param in model.backbone.parameters():
            param.requires_grad = True
        stage = "Full Model"
    
    # Neck/Headは常に学習可能
    for param in model.neck.parameters():
        param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True
    
    return stage


def collate_fn(batch):
    """DataLoader用のcollate関数"""
    return tuple(zip(*batch))


# ===== メイン学習関数（修正版） =====
def main():

    # 最初にバージョン情報を表示
    training_version.print_version_info()

    # 設定初期化
    config = TrainingConfig()
    print("🚀 Starting Unified EfficientNet Training with Comprehensive Debug")
    print(f"📱 Device: {config.device}")
    print(f"🎯 Target: {config.num_classes} classes")
    print(f"📏 Input size: {config.input_size}")
    print("ver 1.5-gemini - Corrected for Latest Project Structure")
    
    # テスト用コード
    from efficientnet_model import test_model_creation

    model = test_model_creation()
    if model:
        print("✅ モデル作成成功！")
    else:
        print("❌ モデル作成失敗")

    # === デバッグトラッカー初期化 ===
    debug_tracker = LossDebugTracker()
    root_investigator = RootCauseInvestigator()
    step_counter = 0
    
    # === AMP設定を完全無効化 ===
    print("🔧 AMP（自動混合精度）を完全無効化します")
    use_amp = False
    scaler = None

    # モデルを明示的にfloat32に設定
    model = create_efficientnet_model(num_classes=config.num_classes, pretrained=True).to(config.device)
    model = model.float()  # 明示的にfloat32に設定
    print(f"🔧 Model dtype: {next(model.parameters()).dtype}")

    
    # データセット準備
    print("\n=== 📚 Dataset Loading ===")
    train_dataset = YoloInfraredDataset(
        image_dir=config.train_img_dir,
        label_dir=config.train_label_dir,
        input_size=config.input_size,
        num_classes=config.num_classes
    )
    
    print(f"✅ Dataset: {len(train_dataset)} samples")
    
    # アンカー最適化
    print("\n=== 🎯 Anchor Optimization ===")
    
    default_anchors = get_default_anchors()
    
    try:
        # データセット分析とアンカー最適化
        optimized_anchors_pixel, class_dist, stats = analyze_dataset_statistics(
            train_dataset,
            num_samples=1000,
            input_size=config.input_size
        )
        
        if optimized_anchors_pixel is None:
            print("⚠️ Using default anchors")
            optimized_anchors_pixel = default_anchors
        else:
            print("✅ Anchor optimization completed")
            
            # デフォルトとの比較
            print("\n=== 📊 Anchor Comparison ===")
            comparison = compare_anchor_sets(
                train_dataset,
                default_anchors,
                optimized_anchors_pixel,
                num_samples=500,
                input_size=config.input_size
            )
            
    except Exception as e:
        print(f"⚠️ Anchor optimization failed: {e}")
        print("   Using default anchors")
        optimized_anchors_pixel = default_anchors
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"📦 Batches per epoch: {len(train_loader)}")
    
    # 進捗表示間隔の調整
    if len(train_loader) < config.progress_interval:
        print(f"⚠️ Only {len(train_loader)} batches, adjusting progress interval to 1")
        config.progress_interval = 1
    else:
        print(f"📊 Progress will be shown every {config.progress_interval} batches")
    
        # モデル初期化
    print("\n=== 🤖 Model Initialization ===")
    model = create_efficientnet_model(num_classes=config.num_classes, pretrained=True).to(config.device)

    # === 📐 Anchor & Grid Setup === の部分
    print("\n=== 📐 Anchor & Grid Setup ===")
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 1, *config.input_size[::-1]).to(config.device)
        # 実際のモデルからグリッドサイズを取得
        feat1, feat2, feat3 = model.backbone(test_input)
        p1, p2, p3 = model.neck(feat1, feat2, feat3)
        grid_sizes = [(p.shape[2], p.shape[3]) for p in [p1, p2, p3]]
        strides = [config.input_size[1] // gs[0] for gs in grid_sizes]
        
        # アンカーポイントとストライドをテンソル形式で統一
        anchor_points_list = []
        strides_list = []
        for i, stride_val in enumerate(strides):
            h, w = grid_sizes[i]
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid = torch.stack((grid_x, grid_y), 2).view(-1, 2)
            anchor_points = (grid.float() + 0.5)
            anchor_points_list.append(anchor_points.to(config.device))
            strides_list.append(torch.full((h * w, 1), stride_val).to(config.device))

        # フラット化したテンソルも作成
        anchor_points_flat = torch.cat(anchor_points_list, dim=0)  # [total_anchors, 2]
        strides_flat = torch.cat(strides_list, dim=0)             # [total_anchors, 1]

    # ★★★ 重要な修正: loss関数の作成部分も修正 ★★★
    anchor_info = {
        'anchor_points': anchor_points_list,      # レベル別リスト（後方互換用）
        'strides_list': strides_list,            # レベル別リスト（後方互換用）
        'anchor_points_flat': anchor_points_flat, # フラット化テンソル
        'strides': strides_flat,                 # テンソル形式
        'grid_sizes': grid_sizes,
        'input_size': config.input_size
    }

    # アンカー・グリッド情報準備（Loss関数用に別途作成）
    anchor_grid_info = prepare_anchor_grid_info(
        anchors_pixel_per_level=optimized_anchors_pixel,
        model_grid_sizes=grid_sizes,
        input_size_wh=config.input_size
    )

    # ★★★ 重要: Loss関数にはanchor_infoを渡す（anchor_grid_infoではない） ★★★
    print("\n=== 🎯 Loss Function Setup ===")
    loss_fn = create_enhanced_loss(
        num_classes=config.num_classes,
        anchor_info=anchor_info,  # ← anchor_grid_info ではなく anchor_info を使う！
        loss_strategy='balanced'
    )

    model.train()
    print(f"✅ Anchor and grid info prepared - Points: {anchor_points_flat.shape}, Strides: {strides_flat.shape}")
    
    # アンカー・グリッド情報準備
    anchor_grid_info = prepare_anchor_grid_info(
        anchors_pixel_per_level=optimized_anchors_pixel,
        model_grid_sizes=grid_sizes,
        input_size_wh=config.input_size
    )
    
    # Loss関数初期化
    print("\n=== 🎯 Loss Function Setup ===")
    loss_fn = create_enhanced_loss(
        num_classes=config.num_classes,
        loss_strategy='balanced',  # unified_loss.pyの戦略を使用
        anchor_info=anchor_grid_info
    )
    
    # オプティマイザとスケジューラ
    print("\n=== ⚙️ Optimizer Setup ===")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.initial_lr,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    batches_per_optimizer_step = len(train_loader) // config.accumulation_steps
    scheduler = WarmRestartScheduler(
        optimizer=optimizer,
        T_0=batches_per_optimizer_step * 2,  # 2エポック相当
        T_mult=1.5,
        eta_min=config.min_lr,
        verbose=True
    )
    
    # unified_training.py の学習ループ部分を以下に置き換え

    # 学習ループ（完全修正版）
    print(f"\n🎬 Training Started with Bug Fixes!")
    best_box_loss = float('inf')
    best_obj_loss = 0.0
    
    # ★★★ 修正1: step_counterを関数レベルで初期化 ★★★
    global_step_counter = 0
    
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        epoch_losses = {"total": [], "box": [], "obj": [], "cls": []}
        
        # 段階的解凍
        unfreezing_stage = progressive_unfreezing(model, epoch)
        
        print(f"\n{'='*80}")
        print(f"🎯 Epoch {epoch+1}/{config.epochs} | Stage: {unfreezing_stage}")
        print(f"📦 Total batches: {len(train_loader)} | Batch size: {config.batch_size}")
        print(f"⚙️ Accumulation steps: {config.accumulation_steps}")
        print(f"{'='*80}")
        
        # 詳細な進捗トラッカー
        successful_batches = 0
        oom_count = 0
        error_count = 0
        batch_times = []
        skipped_batches = 0
        
        # ★★★ 修正2: current_lrを事前に初期化 ★★★
        current_lr = config.initial_lr
        
        # Gradient Accumulation用
        accumulated_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            batch_start_time = time.time()
            
            if len(images) == 0:
                skipped_batches += 1
                if batch_idx % 100 == 0:  # たまに報告
                    print(f"   ⚠️ Batch {batch_idx}: Empty batch (total skipped: {skipped_batches})")
                continue
        
            try:
                # ★★★ 修正3: メモリ情報を安全に取得 ★★★
                current_memory = 0.0
                if batch_idx % 10 == 0:  # 頻度を下げる
                    try:
                        current_memory = aggressive_memory_cleanup()
                        if current_memory > 13.0:
                            print(f"   🚨 Batch {batch_idx}: High memory: {current_memory:.1f}GB")
                    except Exception:
                        current_memory = 0.0
                
                # バッチ準備
                images = torch.stack(images).to(config.device, non_blocking=True).float()
                targets = [t.to(config.device, non_blocking=True).float() for t in targets]
                
                # Forward pass
                preds = model(images)
                preds = preds.float()
                
                # ★★★ 修正4: ターゲット構築のエラーハンドリング強化 ★★★
                try:
                    target_dict = build_targets(
                        predictions=preds.detach(),
                        targets=targets,
                        anchor_info=anchor_info,
                        num_classes=config.num_classes
                    )
                except Exception as target_error:
                    print(f"   ❌ build_targets failed at batch {batch_idx}: {type(target_error).__name__}")
                    error_count += 1
                    continue
                
                # ★★★ 修正5: 損失計算の詳細デバッグ ★★★
                try:
                    loss_dict = loss_fn(preds, target_dict)
                    
                    # 異常な損失値の検出と修正
                    if loss_dict["obj"].item() > 1000.0:
                        print(f"   🚨 ABNORMAL OBJ LOSS DETECTED: {loss_dict['obj'].item():.2f}")
                        print(f"      Box: {loss_dict['box'].item():.4f}")
                        print(f"      Cls: {loss_dict['cls'].item():.4f}")
                        print(f"      Positive samples: {loss_dict.get('pos_samples', 'N/A')}")
                        
                        # 異常値の場合は学習率を1/10に下げて続行
                        if current_lr > 1e-5:
                            current_lr = current_lr / 10.0
                            for g in optimizer.param_groups:
                                g['lr'] = current_lr
                            print(f"      → Emergency LR reduction to {current_lr:.2e}")
                        
                        # それでも異常な場合はスキップ
                        if loss_dict["obj"].item() > 10000.0:
                            print(f"      → Skipping this batch (too abnormal)")
                            error_count += 1
                            continue
                    
                except Exception as loss_error:
                    print(f"   ❌ Loss calculation failed at batch {batch_idx}: {type(loss_error).__name__}")
                    error_count += 1
                    continue
                
                # ★★★ 修正6: 安全な損失値の取得 ★★★
                try:
                    loss = loss_dict["total"] / config.accumulation_steps
                    
                    # NaN/Inf チェック
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"   🚨 NaN/Inf loss detected at batch {batch_idx}, skipping")
                        error_count += 1
                        continue
                    
                    # 異常に大きい損失値のクリッピング
                    if loss.item() > 1000.0:
                        loss = torch.clamp(loss, max=100.0)
                        print(f"   ⚠️ Loss clipped to {loss.item():.2f}")
                    
                    accumulated_loss += loss.item()
                    
                except Exception as e:
                    print(f"   ❌ Loss processing failed: {e}")
                    error_count += 1
                    continue
                
                # ★★★ 修正7: 安全なBackward pass ★★★
                try:
                    loss.backward()
                    
                    # NaN勾配の即座チェック
                    has_nan_grad = False
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f"   🚨 NaN gradient in {name} at batch {batch_idx}")
                            has_nan_grad = True
                            break
                    
                    if has_nan_grad:
                        optimizer.zero_grad()  # 勾配をクリア
                        error_count += 1
                        continue
                        
                except Exception as backward_error:
                    print(f"   ❌ Backward pass failed: {backward_error}")
                    error_count += 1
                    continue
                
                # ★★★ 修正8: Gradient Stepの安全な実行 ★★★
                if (batch_idx + 1) % config.accumulation_steps == 0:
                    try:
                        # 勾配クリッピング
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        # オプティマイザステップ
                        optimizer.step()
                        global_step_counter += 1

                        # ★★★ 修正9: 学習率更新の安全な実行 ★★★
                        if global_step_counter < config.warmup_steps:
                            lr_scale = (global_step_counter + 1) / config.warmup_steps
                            current_lr = config.initial_lr * lr_scale
                            for g in optimizer.param_groups:
                                g['lr'] = current_lr
                        else:
                            try:
                                current_lr = scheduler.step()
                            except Exception:
                                current_lr = optimizer.param_groups[0]['lr']

                        optimizer.zero_grad()
                        accumulated_loss = 0.0
                        
                    except Exception as step_error:
                        print(f"   ❌ Optimizer step failed: {step_error}")
                        optimizer.zero_grad()
                        error_count += 1
                        continue
                
                # ★★★ 修正10: 安全な進捗表示 ★★★
                if batch_idx % config.progress_interval == 0:
                    try:
                        progress_pct = ((batch_idx + 1) / len(train_loader)) * 100
                        elapsed = time.time() - epoch_start_time
                        eta_seconds = elapsed / (batch_idx + 1) * (len(train_loader) - batch_idx - 1)
                        eta_minutes = eta_seconds / 60
                        
                        print(f"""   📊 Batch {batch_idx+1:4d}/{len(train_loader)} ({progress_pct:5.1f}%)
                        📉 Loss: T={loss_dict['total']:.4f} B={loss_dict['box']:.4f} O={loss_dict['obj']:.4f} C={loss_dict['cls']:.4f}
                        ⚙️ LR: {current_lr:.2e} | ETA: {eta_minutes:.1f}min | Mem: {current_memory:.1f}GB
                        ✅ Success: {successful_batches} | ❌ Errors: {error_count} | 🚨 OOM: {oom_count}""")
                        
                    except Exception:
                        print(f"   📊 Batch {batch_idx+1}/{len(train_loader)} - Progress display error")
                
                # 損失履歴に記録
                try:
                    epoch_losses["total"].append(loss_dict["total"].item())
                    epoch_losses["box"].append(loss_dict["box"].item())
                    epoch_losses["obj"].append(loss_dict["obj"].item())
                    epoch_losses["cls"].append(loss_dict["cls"].item())
                except Exception:
                    pass  # 記録失敗は無視
                
                successful_batches += 1
                batch_times.append(time.time() - batch_start_time)
                
                # 中間テンソルを明示的に削除
                try:
                    del preds, target_dict, loss_dict, loss
                except Exception:
                    pass
                
                # 定期的なメモリクリーンアップ
                if batch_idx % config.memory_check_interval == 0:
                    try:
                        aggressive_memory_cleanup()
                    except Exception:
                        pass
                    
            except torch.cuda.OutOfMemoryError as e:
                oom_count += 1
                print(f"   🚨 OOM at batch {batch_idx}/{len(train_loader)} (#{oom_count})")
                try:
                    aggressive_memory_cleanup()
                    optimizer.zero_grad()
                    accumulated_loss = 0.0
                except Exception:
                    pass
                continue
                
            except Exception as e:
                error_count += 1
                print(f"   ⚠️ Unexpected error at batch {batch_idx}: {type(e).__name__}: {str(e)[:100]}")
                try:
                    aggressive_memory_cleanup()
                    optimizer.zero_grad()
                except Exception:
                    pass
                continue
        
        # ★★★ 修正11: エポック終了時の詳細サマリー ★★★
        epoch_time = time.time() - epoch_start_time
        
        print(f"\n{'='*80}")
        print(f"📊 Epoch {epoch+1} Summary | Time: {epoch_time:.1f}s ({epoch_time/60:.1f}min)")
        print(f"✅ Successful: {successful_batches}/{len(train_loader)} ({successful_batches/len(train_loader)*100:.1f}%)")
        print(f"⚠️ Skipped: {skipped_batches} | ❌ Errors: {error_count} | 🚨 OOM: {oom_count}")
        
        if batch_times:
            avg_batch_time = np.mean(batch_times)
            estimated_full_time = avg_batch_time * len(train_loader) / 60
            print(f"⏱️ Avg batch: {avg_batch_time:.2f}s | Est. full epoch: {estimated_full_time:.1f}min")
        
        # ★★★ 修正12: 損失サマリーの安全な計算 ★★★
        if epoch_losses["total"] and len(epoch_losses["total"]) > 0:
            try:
                # 異常値を除外してから平均計算
                total_losses = [l for l in epoch_losses['total'] if l < 1000.0]
                box_losses = [l for l in epoch_losses['box'] if l < 100.0]
                obj_losses = [l for l in epoch_losses['obj'] if l < 1000.0]
                cls_losses = [l for l in epoch_losses['cls'] if l < 100.0]
                
                if total_losses:
                    avg_metrics = {
                        'total': np.mean(total_losses),
                        'box': np.mean(box_losses) if box_losses else 0.0,
                        'obj': np.mean(obj_losses) if obj_losses else 0.0,
                        'cls': np.mean(cls_losses) if cls_losses else 0.0
                    }
                    
                    print(f"📉 Average Losses (outliers removed):")
                    print(f"   Total: {avg_metrics['total']:.4f} | Box: {avg_metrics['box']:.4f}")
                    print(f"   Obj: {avg_metrics['obj']:.4f} | Cls: {avg_metrics['cls']:.4f}")
                    print(f"   Valid samples: {len(total_losses)}/{len(epoch_losses['total'])}")
                    
                    # ★★★ 問題診断 ★★★
                    if successful_batches < len(train_loader) * 0.3:
                        print(f"\n🚨 CRITICAL: Success rate < 30%!")
                        print(f"   → Reduce batch_size to {max(1, config.batch_size//2)}")
                        print(f"   → Increase accumulation_steps to {config.accumulation_steps*2}")
                    
                    if len(total_losses) < len(epoch_losses['total']) * 0.5:
                        print(f"\n🚨 WARNING: 50%+ outlier losses detected!")
                        print(f"   → Loss function may be unstable")
                        print(f"   → Consider reducing learning rate to {current_lr/10:.2e}")
                    
                    if avg_metrics['obj'] < 0.001:
                        print(f"\n🚨 WARNING: Objectness loss collapsed!")
                        print(f"   → Model predictions may have saturated")
                        print(f"   → Consider restarting with lower LR")
                    
                    # モデル保存条件
                    save_model = False
                    
                    if avg_metrics['box'] < best_box_loss and avg_metrics['box'] > 0.01:
                        best_box_loss = avg_metrics['box']
                        save_model = True
                        print(f"🎉 New best Box Loss: {best_box_loss:.4f}")
                    
                    if avg_metrics['obj'] > best_obj_loss and avg_metrics['obj'] > 0.01:
                        best_obj_loss = avg_metrics['obj']
                        save_model = True
                        print(f"🎉 Obj Loss improved: {best_obj_loss:.4f}")
                    
                    # モデル保存
                    if save_model and successful_batches > len(train_loader) * 0.1:
                        try:
                            checkpoint = {
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'best_box_loss': best_box_loss,
                                'best_obj_loss': best_obj_loss,
                                'avg_metrics': avg_metrics,
                                'training_stats': {
                                    'successful_batches': successful_batches,
                                    'total_batches': len(train_loader),
                                    'oom_count': oom_count,
                                    'error_count': error_count,
                                    'epoch_time': epoch_time,
                                    'final_lr': current_lr
                                }
                            }
                            
                            save_path = os.path.join(
                                config.model_save_path,
                                f"fixed_model_epoch_{epoch+1}_box_{avg_metrics['box']:.4f}_success_{successful_batches}.pth"
                            )
                            torch.save(checkpoint, save_path)
                            print(f"💾 Model saved: {os.path.basename(save_path)}")
                        except Exception as save_error:
                            print(f"⚠️ Model save failed: {save_error}")
                    
                else:
                    print(f"❌ No valid losses recorded (all outliers)")
                    
            except Exception as summary_error:
                print(f"❌ Summary calculation failed: {summary_error}")
        else:
            print(f"❌ No losses recorded in this epoch")
            print(f"   → Check data loading and model forward pass")
        
        print(f"{'='*80}")
        
        # 早期終了条件
        if successful_batches == 0:
            print(f"💥 TRAINING HALT: No successful batches in epoch {epoch+1}")
            print(f"   → Check your data, model, or reduce batch size")
            break
        
        if epoch_losses["total"] and len([l for l in epoch_losses['total'] if l < 1000.0]) > 0:
            valid_losses = [l for l in epoch_losses['total'] if l < 1000.0]
            avg_loss = np.mean(valid_losses)
            if avg_loss < 0.5:
                print(f"🎯 TARGET ACHIEVED! Average Loss < 0.5")
                break
    
    # 学習終了処理
    print(f"\n🎊 Training Complete!")
    print(f"🏆 Best Box Loss: {best_box_loss:.4f}")
    print(f"👁️ Best Obj Loss: {best_obj_loss:.4f}")
    
    return model, best_box_loss, best_obj_loss


if __name__ == "__main__":
    try:
        model, best_box_loss, best_obj_loss = main()
        print(f"\n✅ Training completed successfully!")
        print(f"🎯 Final best Box Loss: {best_box_loss:.4f}")
        print(f"👁️ Final best Obj Loss: {best_obj_loss:.4f}")
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()