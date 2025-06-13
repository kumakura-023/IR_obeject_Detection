# advanced_losses.py - Phase 4: 高度損失関数実装

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class CIoULoss(nn.Module):
    """Complete IoU Loss - 座標回帰の大幅改善"""
    
    def __init__(self):
        super().__init__()
        self.eps = 1e-7
        
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: [N, 4] (x1, y1, x2, y2) or (cx, cy, w, h)
            target_boxes: [N, 4] same format
        Returns:
            ciou_loss: Complete IoU loss
        """
        # Convert to x1y1x2y2 format if needed
        if pred_boxes.shape[-1] == 4:
            pred_x1y1x2y2 = self.xywh_to_x1y1x2y2(pred_boxes)
            target_x1y1x2y2 = self.xywh_to_x1y1x2y2(target_boxes)
        else:
            pred_x1y1x2y2 = pred_boxes
            target_x1y1x2y2 = target_boxes
        
        # IoU計算
        iou = self.calculate_iou(pred_x1y1x2y2, target_x1y1x2y2)
        
        # Distance penalty (中心点距離)
        distance_penalty = self.calculate_distance_penalty(pred_boxes, target_boxes)
        
        # Aspect ratio penalty
        aspect_penalty = self.calculate_aspect_penalty(pred_boxes, target_boxes)
        
        # Complete IoU = IoU - distance_penalty - aspect_penalty
        ciou = iou - distance_penalty - aspect_penalty
        
        # Loss (1 - CIoU)
        ciou_loss = 1 - ciou
        
        return ciou_loss.mean()
    
    def xywh_to_x1y1x2y2(self, boxes):
        """Center format to corner format"""
        cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def calculate_iou(self, pred_boxes, target_boxes):
        """Standard IoU calculation"""
        # Intersection
        x1 = torch.max(pred_boxes[..., 0], target_boxes[..., 0])
        y1 = torch.max(pred_boxes[..., 1], target_boxes[..., 1])
        x2 = torch.min(pred_boxes[..., 2], target_boxes[..., 2])
        y2 = torch.min(pred_boxes[..., 3], target_boxes[..., 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Union
        area_pred = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
        area_target = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
        union = area_pred + area_target - intersection
        
        iou = intersection / (union + self.eps)
        return iou
    
    def calculate_distance_penalty(self, pred_boxes, target_boxes):
        """Center point distance penalty"""
        # Center points
        pred_cx = (pred_boxes[..., 0] + pred_boxes[..., 2]) / 2
        pred_cy = (pred_boxes[..., 1] + pred_boxes[..., 3]) / 2
        target_cx = (target_boxes[..., 0] + target_boxes[..., 2]) / 2
        target_cy = (target_boxes[..., 1] + target_boxes[..., 3]) / 2
        
        # Distance between centers
        center_distance_sq = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        # Diagonal of enclosing box
        enclose_x1 = torch.min(pred_boxes[..., 0], target_boxes[..., 0])
        enclose_y1 = torch.min(pred_boxes[..., 1], target_boxes[..., 1])
        enclose_x2 = torch.max(pred_boxes[..., 2], target_boxes[..., 2])
        enclose_y2 = torch.max(pred_boxes[..., 3], target_boxes[..., 3])
        
        enclose_diagonal_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
        
        distance_penalty = center_distance_sq / (enclose_diagonal_sq + self.eps)
        return distance_penalty
    
    def calculate_aspect_penalty(self, pred_boxes, target_boxes):
        """Aspect ratio penalty"""
        # Width and height
        pred_w = pred_boxes[..., 2] - pred_boxes[..., 0]
        pred_h = pred_boxes[..., 3] - pred_boxes[..., 1]
        target_w = target_boxes[..., 2] - target_boxes[..., 0]
        target_h = target_boxes[..., 3] - target_boxes[..., 1]
        
        # Aspect ratio difference
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(target_w / (target_h + self.eps)) - torch.atan(pred_w / (pred_h + self.eps)), 2
        )
        
        # IoU for alpha calculation
        iou = self.calculate_iou(pred_boxes, target_boxes)
        alpha = v / (1 - iou + v + self.eps)
        
        aspect_penalty = alpha * v
        return aspect_penalty

class FocalLoss(nn.Module):
    """Focal Loss - クラス不均衡対策"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] logits
            targets: [N] class indices or [N, C] one-hot
        """
        # Cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Probability of correct class
        pt = torch.exp(-ce_loss)
        
        # Alpha weighting
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        else:
            alpha_t = self.alpha.gather(0, targets)
        
        # Focal weight
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        
        # Focal loss
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Loss - 汎化性能向上"""
    
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] logits
            targets: [N] class indices
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # True class probability
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

class AdvancedMultiScaleLoss(nn.Module):
    """Phase 4: 高度マルチスケール損失関数"""
    
    def __init__(self, anchors, num_classes=15, 
                 use_ciou=True, use_focal=True, use_label_smoothing=True):
        super().__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.use_ciou = use_ciou
        self.use_focal = use_focal
        self.use_label_smoothing = use_label_smoothing
        
        # Original components
        from anchor_loss import AnchorMatcher
        self.matcher = AnchorMatcher(self.anchors)
        
        # Advanced loss functions
        if use_ciou:
            self.ciou_loss = CIoULoss()
        if use_focal:
            self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        if use_label_smoothing:
            self.label_smoothing = LabelSmoothingLoss(num_classes, smoothing=0.1)
        
        # Standard loss functions (fallback)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
        # Loss weights
        self.lambda_coord = 8.0    # 5.0 → 8.0 (CIoU強化)
        self.lambda_obj = 1.0
        self.lambda_noobj = 0.5
        self.lambda_cls = 1.5      # 1.0 → 1.5 (Focal Loss強化)
        
        print(f"🚀 AdvancedMultiScaleLoss initialized")
        print(f"   CIoU Loss: {'ON' if use_ciou else 'OFF'}")
        print(f"   Focal Loss: {'ON' if use_focal else 'OFF'}")
        print(f"   Label Smoothing: {'ON' if use_label_smoothing else 'OFF'}")
        print(f"   Enhanced weights: coord={self.lambda_coord}, cls={self.lambda_cls}")
    
    def forward(self, predictions, targets):
        """
        Advanced multi-scale loss computation
        """
        device = next(iter(predictions.values())).device
        scale_info = {
            'small': 52,
            'medium': 26,
            'large': 13
        }
        
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_components = {
            'coord_loss': 0,
            'obj_loss': 0,
            'noobj_loss': 0,
            'cls_loss': 0,
            'total_loss': 0,
            'ciou_improvement': 0,  # CIoU improvement tracking
            'focal_improvement': 0  # Focal improvement tracking
        }
        
        for scale_name, grid_size in scale_info.items():
            if scale_name not in predictions:
                continue
                
            preds = predictions[scale_name]
            B, N, C = preds.shape
            
            # Extract predictions
            pred_xy = preds[..., :2]
            pred_wh = preds[..., 2:4]
            pred_conf = preds[..., 4]
            pred_cls = preds[..., 5:]
            
            # Match targets to anchors
            matched_targets = self.matcher.match_targets_to_anchors(
                targets, preds, scale_name, grid_size, device
            )
            
            # Extract target components
            obj_mask = matched_targets[..., 0]
            target_xy = matched_targets[..., 2:4]
            target_wh = matched_targets[..., 4:6]
            target_conf = matched_targets[..., 6]
            
            # Create masks
            obj_mask_bool = obj_mask.bool()
            noobj_mask_bool = ~obj_mask_bool
            
            # Coordinate loss with CIoU
            num_pos = obj_mask_bool.sum()
            if num_pos > 0:
                pos_indices = torch.where(obj_mask_bool)
                
                if self.use_ciou:
                    # CIoU Loss for better coordinate regression
                    pred_boxes = torch.cat([
                        torch.sigmoid(pred_xy[pos_indices]),
                        pred_wh[pos_indices]
                    ], dim=-1)
                    target_boxes = torch.cat([
                        target_xy[pos_indices],
                        target_wh[pos_indices]
                    ], dim=-1)
                    
                    coord_loss = self.ciou_loss(pred_boxes, target_boxes)
                    loss_components['ciou_improvement'] = 1  # Flag for tracking
                else:
                    # Standard MSE loss
                    coord_loss_xy = self.mse_loss(
                        torch.sigmoid(pred_xy[pos_indices]),
                        target_xy[pos_indices]
                    ).sum()
                    coord_loss_wh = self.mse_loss(
                        pred_wh[pos_indices],
                        target_wh[pos_indices]
                    ).sum()
                    coord_loss = (coord_loss_xy + coord_loss_wh) / num_pos
                
                # Class loss with Focal Loss
                cls_targets = matched_targets[pos_indices][:, 1].long()
                
                if self.use_focal:
                    # Focal Loss for class imbalance
                    cls_loss = self.focal_loss(pred_cls[pos_indices], cls_targets)
                    loss_components['focal_improvement'] = 1
                elif self.use_label_smoothing:
                    # Label Smoothing for better generalization
                    cls_loss = self.label_smoothing(pred_cls[pos_indices], cls_targets)
                else:
                    # Standard Cross Entropy
                    cls_loss = self.ce_loss(pred_cls[pos_indices], cls_targets).sum() / num_pos
                
                # Objectness loss for positive samples
                obj_loss = self.bce_loss(
                    pred_conf[pos_indices],
                    target_conf[pos_indices]
                ).sum() / num_pos
                
            else:
                coord_loss = torch.tensor(0.0, device=device)
                cls_loss = torch.tensor(0.0, device=device)
                obj_loss = torch.tensor(0.0, device=device)
            
            # Negative samples objectness loss
            num_neg = noobj_mask_bool.sum()
            if num_neg > 0:
                neg_indices = torch.where(noobj_mask_bool)
                noobj_loss = self.bce_loss(
                    pred_conf[neg_indices],
                    torch.zeros_like(pred_conf[neg_indices])
                ).sum() / num_neg
            else:
                noobj_loss = torch.tensor(0.0, device=device)
            
            # Scale-specific loss
            scale_loss = (
                self.lambda_coord * coord_loss +
                self.lambda_obj * obj_loss +
                self.lambda_noobj * noobj_loss +
                self.lambda_cls * cls_loss
            )
            
            total_loss = total_loss + scale_loss
            
            # Accumulate loss components
            loss_components['coord_loss'] += coord_loss.item()
            loss_components['obj_loss'] += obj_loss.item()
            loss_components['noobj_loss'] += noobj_loss.item()
            loss_components['cls_loss'] += cls_loss.item()
        
        loss_components['total_loss'] = total_loss.item()
        
        return total_loss

def test_advanced_losses():
    """高度損失関数のテスト"""
    print("🧪 Phase 4高度損失関数テスト")
    print("-" * 50)
    
    # Test CIoU Loss
    print("1. CIoU Loss Test:")
    ciou_loss = CIoULoss()
    pred_boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.4, 0.4]])
    target_boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.15, 0.15, 0.35, 0.35]])
    
    ciou_result = ciou_loss(pred_boxes, target_boxes)
    print(f"   CIoU Loss: {ciou_result:.4f}")
    
    # Test Focal Loss
    print("2. Focal Loss Test:")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    inputs = torch.randn(10, 15)  # 10 samples, 15 classes
    targets = torch.randint(0, 15, (10,))
    
    focal_result = focal_loss(inputs, targets)
    print(f"   Focal Loss: {focal_result:.4f}")
    
    # Test Label Smoothing
    print("3. Label Smoothing Test:")
    label_smoothing = LabelSmoothingLoss(15, smoothing=0.1)
    
    ls_result = label_smoothing(inputs, targets)
    print(f"   Label Smoothing Loss: {ls_result:.4f}")
    
    # Test Advanced Multi-Scale Loss
    print("4. Advanced Multi-Scale Loss Test:")
    anchors = {
        'small':  [(7, 11), (14, 28), (22, 65)],
        'medium': [(42, 35), (76, 67), (46, 126)],
        'large':  [(127, 117), (88, 235), (231, 218)]
    }
    
    advanced_loss = AdvancedMultiScaleLoss(
        anchors, num_classes=15,
        use_ciou=True, use_focal=True, use_label_smoothing=False
    )
    
    # Dummy predictions
    predictions = {
        'small': torch.randn(2, 8112, 20, requires_grad=True),
        'medium': torch.randn(2, 2028, 20, requires_grad=True),
        'large': torch.randn(2, 507, 20, requires_grad=True)
    }
    
    targets = [
        torch.tensor([[0, 0.5, 0.5, 0.1, 0.1], [1, 0.3, 0.7, 0.2, 0.3]]),
        torch.tensor([[2, 0.8, 0.2, 0.4, 0.6]])
    ]
    
    try:
        advanced_result = advanced_loss(predictions, targets)
        print(f"   Advanced Loss: {advanced_result:.4f}")
        
        # Backward test
        advanced_result.backward()
        print(f"   ✅ Gradient flow: OK")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_advanced_losses()
    
    if success:
        print("\n🎉 Phase 4 Step 5 準備完了!")
        print("   CIoU Loss, Focal Loss実装済み")
        print("   train.pyに統合可能")
    else:
        print("\n❌ テスト失敗 - 修正が必要")
