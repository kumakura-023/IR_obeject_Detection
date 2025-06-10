# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ★★★ 共有VersionTrackerをインポート ★★★
from version_tracker import create_version_tracker, VersionTracker

# バージョン管理システム初期化
loss_version = create_version_tracker("Loss System v1.0", "loss.py")
loss_version.add_modification("YOLOLoss実装")
loss_version.add_modification("座標損失改善")


class YOLOLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5
        
    def forward(self, predictions, targets, grid_size):
        """
        predictions: [B, H*W, 5+num_classes]
        targets: list of tensors (各画像のターゲット)
        """
        B = predictions.size(0)
        H, W = grid_size
        device = predictions.device
        
        # 損失の初期化
        total_loss = 0
        num_objects = 0
        
        # バッチごとに処理
        for b in range(B):
            pred = predictions[b]  # [H*W, 5+num_classes]
            target = targets[b]    # [N, 5]
            
            if len(target) == 0:
                # オブジェクトがない場合
                conf_pred = pred[:, 4].sigmoid()
                noobj_loss = F.binary_cross_entropy(conf_pred, 
                                                   torch.zeros_like(conf_pred))
                total_loss += self.lambda_noobj * noobj_loss
                continue
            
            # 各ターゲットに対して処理
            batch_loss = 0
            for t in target:
                cls_id, cx, cy, w, h = t
                
                # グリッド位置
                gx = int(cx * W)
                gy = int(cy * H)
                if gx >= W or gy >= H:
                    continue
                    
                gi = gy * W + gx
                
                # 座標損失
                tx = cx * W - gx
                ty = cy * H - gy
                xy_loss = F.mse_loss(pred[gi, :2].sigmoid(), 
                                    torch.tensor([tx, ty], device=device))
                
                # サイズ損失
                tw = torch.log(w * W + 1e-16)
                th = torch.log(h * H + 1e-16)
                wh_loss = F.mse_loss(pred[gi, 2:4], 
                                    torch.tensor([tw, th], device=device))
                
                # Confidence損失
                conf_loss = F.binary_cross_entropy(pred[gi, 4].sigmoid(), 
                                                  torch.tensor(1., device=device))
                
                # クラス損失
                cls_loss = F.cross_entropy(pred[gi:gi+1, 5:], 
                                         torch.tensor([int(cls_id)], device=device))
                
                batch_loss += (self.lambda_coord * (xy_loss + wh_loss) + 
                              conf_loss + cls_loss)
                num_objects += 1
            
            # 負例のConfidence損失
            obj_mask = torch.zeros(H * W, dtype=torch.bool, device=device)
            for t in target:
                gx = int(t[1] * W)
                gy = int(t[2] * H)
                if 0 <= gx < W and 0 <= gy < H:
                    obj_mask[gy * W + gx] = True
            
            noobj_mask = ~obj_mask
            if noobj_mask.any():
                noobj_loss = F.binary_cross_entropy(
                    pred[noobj_mask, 4].sigmoid(),
                    torch.zeros(noobj_mask.sum(), device=device)
                )
                batch_loss += self.lambda_noobj * noobj_loss
            
            total_loss += batch_loss
        
        return total_loss / max(num_objects, 1)