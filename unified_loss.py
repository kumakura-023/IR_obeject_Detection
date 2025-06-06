import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import defaultdict
from typing import Dict, Optional, List, Tuple


# ★★★ ステップ1: バージョン情報をここに追加 ★★★
# ファイルを修正するたびに、この文字列を更新してください。
__loss_version__ = "v1.4 - lossテストバージョン"


# ===== IoU Loss Functions =====
def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Complete IoU (CIoU) 計算
    
    Args:
        box1, box2: [N, 4] (cx, cy, w, h)
        
    Returns:
        CIoU値 [N]
    """
    b1_x, b1_y, b1_w, b1_h = box1.unbind(-1)
    b2_x, b2_y, b2_w, b2_h = box2.unbind(-1)
    
    # 幅と高さを最小値でクランプ
    b1_w = torch.clamp(b1_w, min=1e-2)
    b1_h = torch.clamp(b1_h, min=1e-2)
    b2_w = torch.clamp(b2_w, min=1e-2)
    b2_h = torch.clamp(b2_h, min=1e-2)
    
    # ボックスの座標計算
    b1_x1, b1_x2 = b1_x - b1_w / 2, b1_x + b1_w / 2
    b1_y1, b1_y2 = b1_y - b1_h / 2, b1_y + b1_h / 2
    b2_x1, b2_x2 = b2_x - b2_w / 2, b2_x + b2_w / 2
    b2_y1, b2_y2 = b2_y - b2_h / 2, b2_y + b2_h / 2
    
    # 交差領域
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = b1_w * b1_h
    area2 = b2_w * b2_h
    union = area1 + area2 - inter_area + eps
    
    # IoU
    iou = inter_area / union
    
    # 中心距離
    center_dist_sq = (b1_x - b2_x)**2 + (b1_y - b2_y)**2
    
    # 外接矩形
    enclose_x1 = torch.min(b1_x1, b2_x1)
    enclose_y1 = torch.min(b1_y1, b2_y1)
    enclose_x2 = torch.max(b1_x2, b2_x2)
    enclose_y2 = torch.max(b1_y2, b2_y2)
    enclose_diag_sq = (enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2 + eps
    
    # アスペクト比項
    v = (4 / (math.pi ** 2)) * (
        torch.atan(b1_w / (b1_h + eps)) - torch.atan(b2_w / (b2_h + eps))
    ) ** 2
    
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    
    ciou = iou - (center_dist_sq / enclose_diag_sq) - alpha * v
    return torch.clamp(ciou, min=-1.0, max=1.0)


def ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """CIoU Loss"""
    ciou = bbox_ciou(pred_boxes, target_boxes)
    return 1 - ciou


# ===== Focal Loss Variants =====
class AdaptiveFocalLoss(nn.Module):
    """適応的Focal Loss"""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 reduction: str = 'mean', adaptive_gamma: bool = False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.adaptive_gamma = adaptive_gamma
        self.step_count = 0
        self.loss_history = []
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # 適応的ガンマ調整
        if self.adaptive_gamma and self.training:
            self.step_count += 1
            current_loss = bce_loss.mean().item()
            self.loss_history.append(current_loss)
            
            if len(self.loss_history) > 100:
                recent_avg = sum(self.loss_history[-50:]) / 50
                older_avg = sum(self.loss_history[-100:-50]) / 50
                
                if recent_avg > older_avg * 1.05:
                    self.gamma = min(self.gamma * 1.01, 3.5)
                elif recent_avg < older_avg * 0.95:
                    self.gamma = max(self.gamma * 0.99, 1.5)
                
                self.loss_history.pop(0)
        
        # Focal Loss計算
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingBCE(nn.Module):
    """ラベルスムージング付きBCE"""
    def __init__(self, smoothing: float = 0.1, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = targets * (1 - self.smoothing) + (self.smoothing / 2)
        
        if self.pos_weight is not None:
            return F.binary_cross_entropy_with_logits(
                inputs, targets_smooth, pos_weight=self.pos_weight, reduction='mean'
            )
        else:
            return F.binary_cross_entropy_with_logits(
                inputs, targets_smooth, reduction='mean'
            )


# ===== Main Loss Class =====
class EnhancedDetectionLoss(nn.Module):
    """統合版物体検出ロス（修正版）"""
    
    def __init__(self, 
                 num_classes: int = 15,
                 lambda_box: float = 5.0,
                 lambda_obj: float = 2.0,
                 lambda_cls: float = 1.0,
                 box_loss_type: str = 'ciou',
                 obj_loss_type: str = 'adaptive_focal',
                 cls_loss_type: str = 'label_smooth',
                 anchor_info: Optional[Dict] = None):
        super().__init__()
        
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        
        # アンカー情報（デコード用）
        self.anchor_info = anchor_info
        
        # Box Loss
        if box_loss_type == 'ciou':
            self.box_loss_fn = ciou_loss
        else:
            self.box_loss_fn = ciou_loss  # デフォルト
        
        # Objectness Loss
        if obj_loss_type == 'adaptive_focal':
            self.obj_loss_fn = AdaptiveFocalLoss(
            alpha=0.75,    # 0.25 → 0.75（正サンプルへの重みを増加）
            gamma=2.0,     # 1.0 → 2.0（easy negativesをより強く抑制）
            adaptive_gamma=True
        )
        else:
            self.obj_loss_fn = AdaptiveFocalLoss(alpha=0.25, gamma=2.0)
        
        # Classification Loss
        if cls_loss_type == 'label_smooth':
            self.cls_loss_fn = LabelSmoothingBCE(smoothing=0.1)
        elif cls_loss_type == 'adaptive_focal':
            self.cls_loss_fn = AdaptiveFocalLoss(alpha=0.25, gamma=1.5)
        else:
            self.cls_loss_fn = LabelSmoothingBCE(smoothing=0.1)
        
        # ステップカウンタとロス履歴
        self.register_buffer('step_count', torch.tensor(0))
        self.loss_history = defaultdict(list)
        
        print(f"🎯 Enhanced Loss initialized:")
        print(f"   Box Loss: {box_loss_type}, Obj Loss: {obj_loss_type}, Cls Loss: {cls_loss_type}")
        print(f"   Loss Weights: Box={lambda_box}, Obj={lambda_obj}, Cls={lambda_cls}")
    
    def forward(self, preds: torch.Tensor, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        ロス計算（修正版）
        
        Args:
            preds: 予測値 [B, N, 5+num_classes]
            targets: ターゲット辞書
            
        Returns:
            ロス辞書
        """
        device = preds.device
        
        # 予測値を分解
        pred_box = preds[..., :4]      # tx, ty, tw, th
        pred_obj = preds[..., 4]       # objectness logits
        pred_cls = preds[..., 5:]      # class logits
        
        # ターゲットを取得
        target_box = targets['boxes']        # tx, ty, tw, th
        target_obj = targets['objectness']   # IoU modulated
        target_cls = targets['classes']      # IoU modulated one-hot
        
        # ===== 修正1: より柔軟なポジティブマスク =====
        # 元々のIoU変調値を保持しつつ、より多くの正サンプルを確保
        pos_mask_strict = target_obj > 0.5    # 厳密な正サンプル
        pos_mask_loose = target_obj > 0.1     # 緩い正サンプル
        pos_mask_any = target_obj > 0.01      # 非常に緩い正サンプル
        
        num_pos_strict = pos_mask_strict.sum()
        num_pos_loose = pos_mask_loose.sum()
        num_pos_any = pos_mask_any.sum()
        
        # Box Loss用のマスクを選択（より多くのサンプルで学習）
        if num_pos_strict > 10:
            pos_mask_for_box = pos_mask_strict
        elif num_pos_loose > 5:
            pos_mask_for_box = pos_mask_loose
        else:
            pos_mask_for_box = pos_mask_any
        
        num_pos = pos_mask_for_box.sum()
        
        # 1. Box Loss（ポジティブサンプルのみ）
        loss_box = torch.tensor(0.0, device=device)
        if num_pos > 0:
            # シンプルにL1 Lossを使用（CIoUの代わりに安定性重視）
            loss_box = F.smooth_l1_loss(
                pred_box[pos_mask_for_box], 
                target_box[pos_mask_for_box],
                reduction='mean'
            )
        
        # ===== 修正2: Objectness Loss の改善 =====
        # ★★★【実験】勾配爆発の原因切り分けのため、Focal Lossを一時的に無効化 ★★★
        # 最も安定した標準のBCE損失で計算し、勾配爆発が収まるかを確認する。
        obj_target_binary = (target_obj > 0.1).float()
        loss_obj = F.binary_cross_entropy_with_logits(pred_obj, obj_target_binary, reduction='mean')

        
        # ===== 修正3: Classification Loss の改善 =====
        loss_cls = torch.tensor(0.0, device=device)
        if num_pos > 0:
            # クラスロスもより多くのサンプルで計算
            cls_pos_mask = pos_mask_loose if num_pos_loose > 0 else pos_mask_any
            
            if cls_pos_mask.any():
                if isinstance(self.cls_loss_fn, AdaptiveFocalLoss):
                    loss_cls = self.cls_loss_fn(pred_cls[cls_pos_mask], target_cls[cls_pos_mask])
                else:
                    loss_cls = self.cls_loss_fn(pred_cls[cls_pos_mask], target_cls[cls_pos_mask])
        
        # ===== 修正4: 動的重み調整の改善 =====
        self.step_count += 1
        current_step = self.step_count.item()
        
        # より穏やかな重み調整
        if current_step < 500:
            # 初期：Objectnessを重視、ただし過度ではない
            w_box = self.lambda_box * 0.8
            w_obj = self.lambda_obj * 2.0  # 3.0 → 2.0に修正
            w_cls = self.lambda_cls * 0.5
        elif current_step < 2000:
            # 中期：バランス重視
            w_box = self.lambda_box * 1.0
            w_obj = self.lambda_obj * 1.5  # 2.5 → 1.5に修正
            w_cls = self.lambda_cls * 0.8
        else:
            # 後期：全てバランスよく
            w_box = self.lambda_box * 1.0
            w_obj = self.lambda_obj * 1.2  # 2.0 → 1.2に修正
            w_cls = self.lambda_cls * 1.0
        
        # ===== 修正5: Loss値の異常検出と補正 =====
        # 異常に小さいロス値を検出してワーニング
        if loss_obj.item() < 1e-6 and current_step > 10:
            print(f"⚠️ WARNING: Objectness loss is too small ({loss_obj.item():.8f}) at step {current_step}")
            # 緊急時の補正：最小値を設定
            loss_obj = torch.maximum(loss_obj, torch.tensor(1e-5, device=device))
        
        if loss_cls.item() < 1e-6 and current_step > 10 and num_pos > 0:
            print(f"⚠️ WARNING: Classification loss is too small ({loss_cls.item():.8f}) at step {current_step}")
            # 緊急時の補正：最小値を設定
            loss_cls = torch.maximum(loss_cls, torch.tensor(1e-5, device=device))
        
        # 総合ロス
        total_loss = w_box * loss_box + w_obj * loss_obj + w_cls * loss_cls
        
        # ロス履歴更新
        self.loss_history['box'].append(loss_box.item())
        self.loss_history['obj'].append(loss_obj.item())
        self.loss_history['cls'].append(loss_cls.item())
        self.loss_history['total'].append(total_loss.item())
        
        # 履歴サイズ制限
        for key in self.loss_history:
            if len(self.loss_history[key]) > 500:
                self.loss_history[key].pop(0)
        
        return {
            'total': total_loss,
            'box': loss_box,
            'obj': loss_obj,
            'cls': loss_cls,
            'pos_samples': num_pos.item(),
            'pos_samples_breakdown': {
                'strict': num_pos_strict.item(),
                'loose': num_pos_loose.item(),
                'any': num_pos_any.item()
            },
            'dynamic_weights': {
                'box': w_box,
                'obj': w_obj,
                'cls': w_cls
            }
        }
    
    def get_loss_statistics(self) -> Dict:
        """ロス統計を取得"""
        stats = {}
        
        for key, values in self.loss_history.items():
            if values:
                tensor_values = torch.tensor(values, dtype=torch.float32)
                stats[key] = {
                    'mean': tensor_values.mean().item(),
                    'std': tensor_values.std().item() if len(values) > 1 else 0.0,
                    'min': tensor_values.min().item(),
                    'max': tensor_values.max().item(),
                    'recent_mean': tensor_values[-100:].mean().item() if len(values) >= 100 else tensor_values.mean().item()
                }
        
        return stats


def create_enhanced_loss(num_classes: int = 15, 
                        loss_strategy: str = 'balanced',
                        anchor_info: Optional[Dict] = None) -> EnhancedDetectionLoss:
    """
    ロス関数を作成（修正版）
    
    Args:
        num_classes: クラス数
        loss_strategy: ロス戦略
        anchor_info: アンカー情報（デコード用）
        
    Returns:
        EnhancedDetectionLoss
    """

    print(f"--- 読み込み中の損失ファイル: unified_loss.py ({__loss_version__}) ---")

    strategies = {
        'balanced': {
            'lambda_box': 5.0,
            'lambda_obj': 2.0,  # 1.0 → 2.0 に調整
            'lambda_cls': 1.0,
            'box_loss_type': 'ciou',
            'obj_loss_type': 'adaptive_focal',
            'cls_loss_type': 'label_smooth'
        },
        'box_focused': {
            'lambda_box': 7.0,
            'lambda_obj': 1.5,  # 2.5 → 1.5 に調整
            'lambda_cls': 0.8,
            'box_loss_type': 'ciou',
            'obj_loss_type': 'adaptive_focal',
            'cls_loss_type': 'label_smooth'
        },
        'obj_focused': {
            'lambda_box': 5.0,
            'lambda_obj': 3.0,
            'lambda_cls': 1.0,
            'box_loss_type': 'ciou',
            'obj_loss_type': 'adaptive_focal',
            'cls_loss_type': 'adaptive_focal'
        }
    }
    
    config = strategies.get(loss_strategy, strategies['balanced'])
    
    if loss_strategy not in strategies:
        print(f"⚠️ Unknown strategy '{loss_strategy}', using 'balanced'")
    
    print(f"🎯 Creating Enhanced Loss with strategy: {loss_strategy}")
    
    return EnhancedDetectionLoss(
        num_classes=num_classes,
        anchor_info=anchor_info,
        **config
    )