# unified_loss.py (最終完成版 - SimOTA対応)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple

# --- ヘルパー関数: CIoU Loss ---
def bbox_ciou(box1, box2, eps=1e-7):
    # (この関数の内容は変更ありません)
    b1_x, b1_y, b1_w, b1_h = box1.unbind(-1)
    b2_x, b2_y, b2_w, b2_h = box2.unbind(-1)
    b1_w = torch.clamp(b1_w, min=1e-2); b1_h = torch.clamp(b1_h, min=1e-2)
    b2_w = torch.clamp(b2_w, min=1e-2); b2_h = torch.clamp(b2_h, min=1e-2)
    b1_x1, b1_x2 = b1_x - b1_w/2, b1_x + b1_w/2
    b1_y1, b1_y2 = b1_y - b1_h/2, b1_y + b1_h/2
    b2_x1, b2_x2 = b2_x - b2_w/2, b2_x + b2_w/2
    b2_y1, b2_y2 = b2_y - b2_h/2, b2_y + b2_h/2
    inter_x1=torch.max(b1_x1,b2_x1); inter_y1=torch.max(b1_y1,b2_y1)
    inter_x2=torch.min(b1_x2,b2_x2); inter_y2=torch.min(b1_y2,b2_y2)
    inter_area=(inter_x2-inter_x1).clamp(0)*(inter_y2-inter_y1).clamp(0)
    area1=(b1_x2-b1_x1)*(b1_y2-b1_y1); area2=(b2_x2-b2_x1)*(b2_y2-b2_y1)
    union=area1+area2-inter_area+eps
    iou=inter_area/union
    enclose_x1=torch.min(b1_x1,b2_x1); enclose_y1=torch.min(b1_y1,b2_y1)
    enclose_x2=torch.max(b1_x2,b2_x2); enclose_y2=torch.max(b1_y2,b2_y2)
    enclose_diag_sq=(enclose_x2-enclose_x1)**2+(enclose_y2-enclose_y1)**2+eps
    center_dist_sq=(b1_x-b2_x)**2+(b1_y-b2_y)**2
    v=(4/(math.pi**2))*torch.pow(torch.atan(b1_w/(b1_h+eps))-torch.atan(b2_w/(b2_h+eps)),2)
    with torch.no_grad():
        alpha=v/(1-iou+v+eps)
    return iou-(center_dist_sq/enclose_diag_sq)-alpha*v

# --- 損失クラス本体 ---
class EnhancedDetectionLoss(nn.Module):
    def __init__(self, num_classes: int, anchor_info: Dict, lambda_box=5.0, lambda_obj=1.0, lambda_cls=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.anchor_info = anchor_info
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls

        self.box_loss_fn = lambda p, t: 1.0 - bbox_ciou(p, t)
        self.obj_loss_fn = F.binary_cross_entropy_with_logits
        self.cls_loss_fn = F.binary_cross_entropy_with_logits
        print("✅ EnhancedDetectionLoss (SimOTA-compatible) initialized.")

    def forward(self, preds: torch.Tensor, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = preds.device
        
        target_boxes_cxcywh = targets['boxes']
        target_obj = targets['objectness']
        target_cls = targets['classes']
        
        pos_mask = target_obj > 0
        num_pos = pos_mask.sum()
        
        pred_logits_all = preds
        pred_logits_box = pred_logits_all[..., :4]
        pred_logits_obj = pred_logits_all[..., 4]
        pred_logits_cls = pred_logits_all[..., 5:]
        
        # --- Box Loss ---
        loss_box = torch.tensor(0.0, device=device)
        if num_pos > 0:
            pred_decoded_box = self.decode_predictions(pred_logits_box)
            pos_pred_boxes = pred_decoded_box[pos_mask]
            pos_target_boxes = target_boxes_cxcywh[pos_mask]
            loss_box = self.box_loss_fn(pos_pred_boxes, pos_target_boxes).sum() / num_pos

        # --- Objectness Loss & Classification Loss ---
        loss_obj = self.obj_loss_fn(pred_logits_obj, target_obj, reduction='sum') / (num_pos + 1e-6)
        
        loss_cls = torch.tensor(0.0, device=device)
        if num_pos > 0:
            loss_cls = self.cls_loss_fn(pred_logits_cls[pos_mask], target_cls[pos_mask], reduction='sum') / num_pos
        
        total_loss = self.lambda_box * loss_box + self.lambda_obj * loss_obj + self.lambda_cls * loss_cls

        return {
            'total': total_loss, 'box': loss_box, 'obj': loss_obj, 'cls': loss_cls, 'pos_samples': num_pos
        }

    def decode_predictions(self, pred_logits_box: torch.Tensor) -> torch.Tensor:
        anchor_points = self.anchor_info['anchor_points'].to(pred_logits_box.device)
        strides = self.anchor_info['strides'].to(pred_logits_box.device)
        
        # ブロードキャストのために形状を合わせる
        if pred_logits_box.ndim > anchor_points.ndim:
            anchor_points = anchor_points.unsqueeze(0)
            strides = strides.unsqueeze(0)

        decoded_xy = (pred_logits_box[..., :2].sigmoid() * 2 - 0.5 + anchor_points) * strides
        decoded_wh = (pred_logits_box[..., 2:].sigmoid() * 2)**2 * strides
        
        # (cx, cy, w, h) 形式に結合
        return torch.cat((decoded_xy, decoded_wh), dim=-1)

def create_enhanced_loss(num_classes: int, anchor_info: Dict, **kwargs) -> EnhancedDetectionLoss:
    print(f"--- 読み込み中の損失ファイル: unified_loss.py (v_final - SimOTA対応版) ---")
    return EnhancedDetectionLoss(num_classes=num_classes, anchor_info=anchor_info, **kwargs)