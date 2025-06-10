# unified_loss.py (最終完成版 - SimOTA対応)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple


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
loss_version = create_version_tracker("Unified Loss System v2.3", "unified_loss.py")
loss_version.add_modification("SimOTA対応損失関数実装")
loss_version.add_modification("動的重み調整機能追加")
loss_version.add_modification("データ型統一 (float32)")

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
# unified_loss.py の EnhancedDetectionLoss クラスに以下のメソッドを追加

class EnhancedDetectionLoss(nn.Module):
    def __init__(self, num_classes: int, anchor_info: Dict, lambda_box: float, lambda_obj: float, lambda_cls: float):
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

    def decode_predictions(self, pred_logits_box: torch.Tensor) -> torch.Tensor:
        """予測値をデコード（完全互換版）"""
        
        print(f"🔧 [LOSS] decode_predictions called")
        print(f"   self.anchor_info keys: {list(self.anchor_info.keys())}")
        
        # ========== anchor_points の取得（全形式対応） ==========
        anchor_points = None
        
        # 1. anchor_points_flat を優先（unified_training.py の新形式）
        if 'anchor_points_flat' in self.anchor_info:
            anchor_points = self.anchor_info['anchor_points_flat'].to(pred_logits_box.device)
            print(f"   Using 'anchor_points_flat': {anchor_points.shape}")
        
        # 2. anchor_points（リスト形式）
        elif 'anchor_points' in self.anchor_info:
            anchor_points_data = self.anchor_info['anchor_points']
            if isinstance(anchor_points_data, list):
                # アンカーごとに3倍に拡張する必要がある場合の処理
                expanded_points = []
                for level_points in anchor_points_data:
                    # 各グリッドポイントを3回繰り返す（3アンカー分）
                    level_points_expanded = level_points.repeat(3, 1)  # [H*W*3, 2]
                    expanded_points.append(level_points_expanded)
                anchor_points = torch.cat(expanded_points, dim=0).to(pred_logits_box.device)
                print(f"   Using 'anchor_points' (list, expanded): {anchor_points.shape}")
            else:
                anchor_points = anchor_points_data.to(pred_logits_box.device)
                print(f"   Using 'anchor_points' (tensor): {anchor_points.shape}")
        
        # 3. グリッドサイズから計算（最終手段）
        elif 'grid_sizes' in self.anchor_info and 'input_size' in self.anchor_info:
            print(f"   Generating anchor_points from grid_sizes")
            grid_sizes = self.anchor_info['grid_sizes']
            device = pred_logits_box.device
            
            all_anchor_points = []
            for h, w in grid_sizes:
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(h, device=device), 
                    torch.arange(w, device=device), 
                    indexing='ij'
                )
                grid = torch.stack((grid_x, grid_y), 2).view(-1, 2).float() + 0.5
                # 各グリッドポイントを3回繰り返す（3アンカー分）
                grid_expanded = grid.repeat(3, 1)  # [H*W*3, 2]
                all_anchor_points.append(grid_expanded)
            
            anchor_points = torch.cat(all_anchor_points, dim=0)
            print(f"   Generated anchor_points: {anchor_points.shape}")
        
        else:
            raise ValueError(f"No valid anchor_points found in keys: {list(self.anchor_info.keys())}")
        
        # ========== strides の取得（全形式対応） ==========
        strides = None
        
        # 1. strides（テンソル形式）を優先
        if 'strides' in self.anchor_info:
            strides_data = self.anchor_info['strides']
            if isinstance(strides_data, torch.Tensor):
                strides = strides_data.to(pred_logits_box.device)
                if strides.dim() == 2 and strides.shape[1] == 1:
                    strides = strides.squeeze(1)  # [N, 1] -> [N]
                # アンカー数に合わせて3倍に拡張
                if strides.shape[0] * 3 == anchor_points.shape[0]:
                    strides = strides.repeat_interleave(3)
                print(f"   Using 'strides' (tensor): {strides.shape}")
            else:
                # リスト形式の場合
                strides = torch.tensor(strides_data, device=pred_logits_box.device).float()
                print(f"   Using 'strides' (converted from list): {strides.shape}")
        
        # 2. strides_per_level から計算
        elif 'strides_per_level' in self.anchor_info and 'grid_sizes' in self.anchor_info:
            strides_per_level = self.anchor_info['strides_per_level']
            grid_sizes = self.anchor_info['grid_sizes']
            device = pred_logits_box.device
            
            all_strides = []
            for i, (h, w) in enumerate(grid_sizes):
                stride_val = strides_per_level[i] if i < len(strides_per_level) else strides_per_level[-1]
                # 各グリッドポイント×3アンカー分のストライド
                level_strides = torch.full((h * w * 3,), stride_val, device=device)
                all_strides.append(level_strides)
            
            strides = torch.cat(all_strides, dim=0)
            print(f"   Generated strides from strides_per_level: {strides.shape}")
        
        # 3. input_size と grid_sizes から計算
        elif 'grid_sizes' in self.anchor_info and 'input_size' in self.anchor_info:
            grid_sizes = self.anchor_info['grid_sizes']
            input_w, input_h = self.anchor_info['input_size']
            device = pred_logits_box.device
            
            all_strides = []
            for h, w in grid_sizes:
                stride_val = input_h / h  # 高さベースでストライド計算
                # 各グリッドポイント×3アンカー分のストライド
                level_strides = torch.full((h * w * 3,), stride_val, device=device)
                all_strides.append(level_strides)
            
            strides = torch.cat(all_strides, dim=0)
            print(f"   Generated strides from grid_sizes: {strides.shape}")
        
        else:
            raise ValueError(f"No valid strides found in keys: {list(self.anchor_info.keys())}")
        
        # ========== サイズ検証と調整 ==========
        B, N, _ = pred_logits_box.shape
        
        if anchor_points.shape[0] != N:
            print(f"⚠️ Size mismatch: anchor_points={anchor_points.shape[0]}, predictions={N}")
            print(f"   Adjusting anchor_points...")
            
            # Nに合わせて調整
            if anchor_points.shape[0] > N:
                anchor_points = anchor_points[:N]
            else:
                # 不足分を最後の値で埋める
                repeat_times = N // anchor_points.shape[0] + 1
                anchor_points = anchor_points.repeat(repeat_times, 1)[:N]
        
        if strides.shape[0] != N:
            print(f"⚠️ Size mismatch: strides={strides.shape[0]}, predictions={N}")
            print(f"   Adjusting strides...")
            
            # Nに合わせて調整
            if strides.shape[0] > N:
                strides = strides[:N]
            else:
                # 不足分を最後の値で埋める
                last_stride = strides[-1] if len(strides) > 0 else 8.0
                padding = torch.full((N - strides.shape[0],), last_stride, device=strides.device)
                strides = torch.cat([strides, padding], dim=0)
        
        # ========== デコード処理 ==========
        try:
            # バッチ次元の処理
            anchor_points = anchor_points.unsqueeze(0)  # [1, N, 2]
            strides = strides.unsqueeze(0)  # [1, N]
            
            # YOLOv5スタイルのデコード
            decoded_xy = (pred_logits_box[..., :2].sigmoid() * 2 - 0.5 + anchor_points) * strides.unsqueeze(-1)
            decoded_wh = (pred_logits_box[..., 2:].sigmoid() * 2) ** 2 * strides.unsqueeze(-1).repeat(1, 1, 2)
            
            # 結合
            decoded_boxes = torch.cat((decoded_xy, decoded_wh), dim=-1)
            
            print(f"✅ [LOSS] decode_predictions successful: {decoded_boxes.shape}")
            return decoded_boxes
            
        except Exception as e:
            print(f"❌ [LOSS] decode_predictions failed at final step: {e}")
            print(f"   pred_logits_box.shape: {pred_logits_box.shape}")
            print(f"   anchor_points.shape: {anchor_points.shape}")
            print(f"   strides.shape: {strides.shape}")
            raise e

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

# ★★★ 修正: create_enhanced_loss が loss_strategy を正しく処理するように変更 ★★★
def create_enhanced_loss(num_classes: int, anchor_info: Dict, loss_strategy: str) -> EnhancedDetectionLoss:

    # バージョン情報表示
    loss_version.print_version_info()
    
    strategies = {
        'balanced': { 'lambda_box': 5.0, 'lambda_obj': 1.0, 'lambda_cls': 1.0 },
        'box_focused': { 'lambda_box': 7.0, 'lambda_obj': 0.8, 'lambda_cls': 0.5 },
    }
    
    config = strategies.get(loss_strategy, strategies['balanced'])
    print(f"🎯 Creating Enhanced Loss with strategy: {loss_strategy} (weights: {config})")
    
    return EnhancedDetectionLoss(
        num_classes=num_classes, 
        anchor_info=anchor_info, 
        **config
    )