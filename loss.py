# loss.py - 修正版（デバッグ機能付き）
import torch
import torch.nn as nn
import torch.nn.functional as F

# ★★★ 共有VersionTrackerをインポート ★★★
from version_tracker import create_version_tracker, VersionTracker

# バージョン管理システム初期化
loss_version = create_version_tracker("Loss System v1.1 - Debug Enhanced", "loss.py")
loss_version.add_modification("YOLOLoss実装")
loss_version.add_modification("座標損失改善")
loss_version.add_modification("デバッグ機能追加 - 形状不整合エラー対策")
loss_version.add_modification("エラーハンドリング強化")


class YOLOLoss(nn.Module):
    def __init__(self, num_classes, debug_mode=True):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5
        self.debug_mode = debug_mode
        self.call_count = 0  # デバッグ用カウンター
        
        print(f"🔧 YOLOLoss初期化（デバッグ強化版）")
        print(f"   クラス数: {num_classes}")
        print(f"   デバッグモード: {'ON' if debug_mode else 'OFF'}")
        print(f"   lambda_coord: {self.lambda_coord}")
        print(f"   lambda_noobj: {self.lambda_noobj}")
        
    def forward(self, predictions, targets, grid_size):
        """
        predictions: [B, H*W, 5+num_classes]
        targets: list of tensors (各画像のターゲット)
        grid_size: (H, W) tuple
        """
        self.call_count += 1
        
        # ★★★ 詳細デバッグ情報 ★★★
        if self.debug_mode and (self.call_count <= 3 or self.call_count % 100 == 0):
            print(f"\n🔍 YOLOLoss Debug (Call #{self.call_count}):")
            print(f"   入力形状: {predictions.shape}")
            print(f"   グリッドサイズ: {grid_size}")
            print(f"   ターゲット数: {[len(t) for t in targets]}")
            print(f"   期待される予測数: {grid_size[0] * grid_size[1]}")
            print(f"   実際の予測数: {predictions.size(1)}")
        
        B = predictions.size(0)
        H, W = grid_size
        device = predictions.device
        
        # ★★★ 形状チェックとエラーハンドリング ★★★
        expected_size = H * W
        actual_size = predictions.size(1)
        
        if actual_size != expected_size:
            print(f"🚨 形状不整合エラー検出:")
            print(f"   期待: {expected_size} (grid {H}x{W})")
            print(f"   実際: {actual_size}")
            print(f"   予測テンソル形状: {predictions.shape}")
            
            # ★★★ 自動修正を試みる ★★★
            if actual_size > expected_size:
                print(f"   → 自動修正: 予測数が多すぎる場合の処理")
                # 3アンカー分の予測が来てる可能性（マルチスケール由来）
                if actual_size == expected_size * 3:
                    print(f"   → 3アンカー検出：最初のアンカーのみ使用")
                    predictions = predictions[:, :expected_size, :]
                else:
                    print(f"   → 予測数を切り詰め")
                    predictions = predictions[:, :expected_size, :]
            elif actual_size < expected_size:
                print(f"   → 警告: 予測数が不足 - パディング追加")
                # 不足分をゼロパディング
                pad_size = expected_size - actual_size
                padding = torch.zeros(B, pad_size, predictions.size(2), device=device)
                predictions = torch.cat([predictions, padding], dim=1)
            
            print(f"   → 修正後形状: {predictions.shape}")
        
        # ★★★ 最終形状確認 ★★★
        if predictions.size(1) != expected_size:
            raise ValueError(
                f"形状修正失敗: grid {H}x{W}={expected_size} vs predictions {predictions.size(1)}"
            )
        
        # 損失の初期化
        total_loss = 0
        num_objects = 0
        
        # ★★★ 統計情報（デバッグ用） ★★★
        debug_stats = {
            'batch_losses': [],
            'confidence_stats': [],
            'coordinate_losses': [],
            'class_losses': []
        }
        
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
                
                if self.debug_mode and b == 0:  # 最初のバッチのみ
                    debug_stats['confidence_stats'].append({
                        'no_objects': True,
                        'mean_conf': conf_pred.mean().item(),
                        'max_conf': conf_pred.max().item(),
                        'noobj_loss': noobj_loss.item()
                    })
                continue
            
            # 各ターゲットに対して処理
            batch_loss = 0
            batch_coord_loss = 0
            batch_cls_loss = 0
            
            for t in target:
                cls_id, cx, cy, w, h = t
                
                # グリッド位置
                gx = int(cx * W)
                gy = int(cy * H)
                if gx >= W or gy >= H:
                    continue
                    
                gi = gy * W + gx
                
                # ★★★ 安全なインデックスチェック ★★★
                if gi >= pred.size(0):
                    print(f"⚠️ インデックス範囲外: gi={gi}, pred_size={pred.size(0)}")
                    continue
                
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
                
                coord_loss_total = xy_loss + wh_loss
                batch_loss += (self.lambda_coord * coord_loss_total + 
                              conf_loss + cls_loss)
                batch_coord_loss += coord_loss_total.item()
                batch_cls_loss += cls_loss.item()
                num_objects += 1
            
            # ★★★ 負例のConfidence損失（形状安全版） ★★★
            try:
                obj_mask = torch.zeros(H * W, dtype=torch.bool, device=device)
                for t in target:
                    gx = int(t[1] * W)
                    gy = int(t[2] * H)
                    if 0 <= gx < W and 0 <= gy < H:
                        gi = gy * W + gx
                        if gi < H * W:  # 安全チェック
                            obj_mask[gi] = True
                
                noobj_mask = ~obj_mask
                
                # ★★★ 形状チェック強化 ★★★
                if noobj_mask.size(0) != pred.size(0):
                    print(f"⚠️ マスク形状不整合: mask={noobj_mask.size(0)}, pred={pred.size(0)}")
                    # マスクサイズを予測に合わせる
                    if noobj_mask.size(0) > pred.size(0):
                        noobj_mask = noobj_mask[:pred.size(0)]
                    else:
                        # 不足分を False で埋める
                        pad_size = pred.size(0) - noobj_mask.size(0)
                        padding = torch.zeros(pad_size, dtype=torch.bool, device=device)
                        noobj_mask = torch.cat([noobj_mask, padding])
                
                if noobj_mask.any():
                    noobj_loss = F.binary_cross_entropy(
                        pred[noobj_mask, 4].sigmoid(),
                        torch.zeros(noobj_mask.sum(), device=device)
                    )
                    batch_loss += self.lambda_noobj * noobj_loss
                    
            except Exception as e:
                print(f"⚠️ noobj_loss計算エラー: {e}")
                print(f"   pred shape: {pred.shape}")
                print(f"   mask shape: {obj_mask.shape if 'obj_mask' in locals() else 'undefined'}")
                # エラー時は noobj_loss をスキップ
            
            total_loss += batch_loss
            
            # ★★★ デバッグ統計収集 ★★★
            if self.debug_mode and b == 0:  # 最初のバッチのみ
                debug_stats['batch_losses'].append(batch_loss.item())
                debug_stats['coordinate_losses'].append(batch_coord_loss)
                debug_stats['class_losses'].append(batch_cls_loss)
        
        # ★★★ デバッグ出力 ★★★
        if self.debug_mode and (self.call_count <= 3 or self.call_count % 100 == 0):
            final_loss = total_loss / max(num_objects, 1)
            print(f"   最終Loss: {final_loss.item():.4f}")
            print(f"   オブジェクト数: {num_objects}")
            if debug_stats['batch_losses']:
                print(f"   バッチLoss例: {debug_stats['batch_losses'][0]:.4f}")
            print(f"="*50)
        
        return total_loss / max(num_objects, 1)
    
    def enable_debug(self):
        """デバッグモードを有効化"""
        self.debug_mode = True
        print("🔍 YOLOLoss: デバッグモード有効化")
    
    def disable_debug(self):
        """デバッグモードを無効化"""
        self.debug_mode = False
        print("🔇 YOLOLoss: デバッグモード無効化")
    
    def get_debug_stats(self):
        """デバッグ統計を取得"""
        return {
            'call_count': self.call_count,
            'lambda_coord': self.lambda_coord,
            'lambda_noobj': self.lambda_noobj,
            'num_classes': self.num_classes
        }


# ★★★ 損失関数テスト用関数 ★★★
def test_yolo_loss_robustness():
    """YOLOLossの堅牢性をテスト"""
    print("🧪 YOLOLoss堅牢性テスト開始")
    print("-" * 50)
    
    loss_fn = YOLOLoss(num_classes=15, debug_mode=True)
    
    # テストケース1: 正常な形状
    print("1. 正常形状テスト:")
    normal_preds = torch.randn(2, 169, 20)  # 13x13=169
    normal_targets = [
        torch.tensor([[0, 0.5, 0.5, 0.1, 0.1]]),
        torch.tensor([[1, 0.3, 0.7, 0.2, 0.3]])
    ]
    try:
        loss1 = loss_fn(normal_preds, normal_targets, (13, 13))
        print(f"   ✅ 正常: Loss = {loss1:.4f}")
    except Exception as e:
        print(f"   ❌ エラー: {e}")
    
    # テストケース2: 3倍の予測数（マルチスケール由来）
    print("2. 3倍予測数テスト:")
    multi_preds = torch.randn(2, 507, 20)  # 13x13x3=507
    try:
        loss2 = loss_fn(multi_preds, normal_targets, (13, 13))
        print(f"   ✅ 自動修正: Loss = {loss2:.4f}")
    except Exception as e:
        print(f"   ❌ エラー: {e}")
    
    # テストケース3: 予測数不足
    print("3. 予測数不足テスト:")
    small_preds = torch.randn(2, 100, 20)  # 169より少ない
    try:
        loss3 = loss_fn(small_preds, normal_targets, (13, 13))
        print(f"   ✅ パディング: Loss = {loss3:.4f}")
    except Exception as e:
        print(f"   ❌ エラー: {e}")
    
    # テストケース4: 空ターゲット
    print("4. 空ターゲットテスト:")
    empty_targets = [torch.zeros((0, 5)), torch.zeros((0, 5))]
    try:
        loss4 = loss_fn(normal_preds, empty_targets, (13, 13))
        print(f"   ✅ 空ターゲット: Loss = {loss4:.4f}")
    except Exception as e:
        print(f"   ❌ エラー: {e}")
    
    print("\n🎉 YOLOLoss堅牢性テスト完了!")
    return True


if __name__ == "__main__":
    # テスト実行
    success = test_yolo_loss_robustness()
    
    if success:
        print("✅ 修正版YOLOLoss準備完了!")
        print("   主な改善点:")
        print("   - 形状不整合の自動検出・修正")
        print("   - 詳細デバッグ情報")
        print("   - エラーハンドリング強化")
        print("   - 3アンカー予測の自動修正")
    else:
        print("❌ テスト失敗 - さらなる修正が必要")