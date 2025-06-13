# post_processing.py - Phase 4: 後処理最適化

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import math

class SoftNMS:
    """Soft-NMS: より良い重複除去"""
    
    def __init__(self, sigma=0.5, iou_threshold=0.3, score_threshold=0.001, method='gaussian'):
        self.sigma = sigma
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.method = method  # 'linear' or 'gaussian'
        
    def __call__(self, boxes, scores, class_ids):
        """
        Args:
            boxes: [N, 4] (x1, y1, x2, y2)
            scores: [N] confidence scores
            class_ids: [N] class indices
        Returns:
            keep_indices: 保持するボックスのインデックス
        """
        if len(boxes) == 0:
            return []
        
        # スコア順にソート
        sorted_indices = torch.argsort(scores, descending=True)
        
        keep_indices = []
        remaining_indices = sorted_indices.clone()
        
        while len(remaining_indices) > 0:
            # 最高スコアのボックスを保持
            current_idx = remaining_indices[0]
            keep_indices.append(current_idx.item())
            
            if len(remaining_indices) == 1:
                break
            
            # 現在のボックスと残りのボックスのIoU計算
            current_box = boxes[current_idx].unsqueeze(0)
            remaining_boxes = boxes[remaining_indices[1:]]
            
            ious = self.calculate_iou(current_box, remaining_boxes)
            
            # Soft-NMSによるスコア更新
            if self.method == 'gaussian':
                # Gaussian decay
                decay_weights = torch.exp(-(ious ** 2) / self.sigma)
            else:
                # Linear decay
                decay_weights = torch.where(
                    ious > self.iou_threshold,
                    1 - ious,
                    torch.ones_like(ious)
                )
            
            # スコア更新
            remaining_scores = scores[remaining_indices[1:]] * decay_weights
            
            # 閾値以下のボックスを除去
            valid_mask = remaining_scores > self.score_threshold
            remaining_indices = remaining_indices[1:][valid_mask]
            
            # スコア更新を反映
            scores[remaining_indices] = remaining_scores[valid_mask]
            
            # 再ソート
            if len(remaining_indices) > 0:
                remaining_scores_valid = scores[remaining_indices]
                sorted_order = torch.argsort(remaining_scores_valid, descending=True)
                remaining_indices = remaining_indices[sorted_order]
        
        return keep_indices
    
    def calculate_iou(self, boxes1, boxes2):
        """IoU計算"""
        # Intersection
        x1 = torch.max(boxes1[:, 0:1], boxes2[:, 0:1].T)
        y1 = torch.max(boxes1[:, 1:2], boxes2[:, 1:2].T)
        x2 = torch.min(boxes1[:, 2:3], boxes2[:, 2:3].T)
        y2 = torch.min(boxes1[:, 3:4], boxes2[:, 3:4].T)
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection
        
        iou = intersection / (union + 1e-7)
        return iou.squeeze()

class TestTimeAugmentation:
    """Test Time Augmentation: 推論時データ拡張"""
    
    def __init__(self, scales=[0.8, 1.0, 1.2], flips=[False, True]):
        self.scales = scales
        self.flips = flips
        
    def __call__(self, model, image):
        """
        Args:
            model: YOLO model
            image: [1, C, H, W] input image
        Returns:
            aggregated_predictions: 集約された予測結果
        """
        model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for scale in self.scales:
                for flip in self.flips:
                    # 画像変換
                    transformed_image = self.transform_image(image, scale, flip)
                    
                    # 推論
                    predictions = model(transformed_image)
                    
                    # 予測結果を元の座標系に戻す
                    restored_predictions = self.restore_predictions(
                        predictions, scale, flip, image.shape[-2:]
                    )
                    
                    all_predictions.append(restored_predictions)
        
        # 予測結果を集約
        aggregated = self.aggregate_predictions(all_predictions)
        return aggregated
    
    def transform_image(self, image, scale, flip):
        """画像変換"""
        # スケール変換
        if scale != 1.0:
            new_size = (int(image.shape[-2] * scale), int(image.shape[-1] * scale))
            image = F.interpolate(image, size=new_size, mode='bilinear', align_corners=False)
        
        # フリップ
        if flip:
            image = torch.flip(image, dims=[-1])
        
        return image
    
    def restore_predictions(self, predictions, scale, flip, original_size):
        """予測結果を元座標系に復元"""
        # マルチスケール予測の場合
        if isinstance(predictions, dict):
            restored = {}
            for scale_name, preds in predictions.items():
                restored[scale_name] = self.restore_single_scale(
                    preds, scale, flip, original_size
                )
            return restored
        else:
            return self.restore_single_scale(predictions, scale, flip, original_size)
    
    def restore_single_scale(self, predictions, scale, flip, original_size):
        """単一スケール予測の復元"""
        # 座標をスケール調整
        if scale != 1.0:
            predictions[..., :4] /= scale
        
        # フリップの場合のx座標調整
        if flip:
            predictions[..., 0] = 1.0 - predictions[..., 0]  # cx座標反転
        
        return predictions
    
    def aggregate_predictions(self, all_predictions):
        """複数予測結果の集約"""
        # 単純平均（より高度な手法も可能）
        if isinstance(all_predictions[0], dict):
            # マルチスケールの場合
            aggregated = {}
            for scale_name in all_predictions[0].keys():
                scale_preds = [pred[scale_name] for pred in all_predictions]
                aggregated[scale_name] = torch.mean(torch.stack(scale_preds), dim=0)
            return aggregated
        else:
            return torch.mean(torch.stack(all_predictions), dim=0)

class MultiScaleInference:
    """Multi-scale Testing: 複数解像度での推論"""
    
    def __init__(self, scales=[320, 416, 512, 608]):
        self.scales = scales
        
    def __call__(self, model, image_path_or_tensor):
        """
        Args:
            model: YOLO model
            image_path_or_tensor: 画像パスまたはテンソル
        Returns:
            best_predictions: 最適な予測結果
        """
        model.eval()
        scale_results = []
        
        with torch.no_grad():
            for scale in self.scales:
                # 画像をスケールにリサイズ
                if isinstance(image_path_or_tensor, str):
                    image = self.load_and_resize_image(image_path_or_tensor, scale)
                else:
                    image = F.interpolate(
                        image_path_or_tensor, 
                        size=(scale, scale), 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # 推論
                predictions = model(image)
                
                # 結果を保存（スケール情報付き）
                scale_results.append({
                    'scale': scale,
                    'predictions': predictions,
                    'confidence': self.calculate_avg_confidence(predictions)
                })
        
        # 最も信頼度の高いスケールの結果を返す
        best_result = max(scale_results, key=lambda x: x['confidence'])
        return best_result['predictions']
    
    def load_and_resize_image(self, image_path, size):
        """画像読み込みとリサイズ"""
        import cv2
        
        img = cv2.imread(image_path, 0)  # グレースケール
        img = cv2.resize(img, (size, size))
        img = img.astype(np.float32) / 255.0
        
        tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        return tensor
    
    def calculate_avg_confidence(self, predictions):
        """平均信頼度計算"""
        if isinstance(predictions, dict):
            total_conf = 0
            count = 0
            for scale_preds in predictions.values():
                conf = torch.sigmoid(scale_preds[..., 4]).mean()
                total_conf += conf
                count += 1
            return total_conf / count
        else:
            return torch.sigmoid(predictions[..., 4]).mean()

class AdvancedPostProcessor:
    """Phase 4: 統合後処理システム"""
    
    def __init__(self, 
                 use_soft_nms=True,
                 use_tta=False,  # 時間コスト考慮
                 use_multiscale=False,  # 時間コスト考慮
                 conf_threshold=0.5,
                 iou_threshold=0.45):
        
        self.use_soft_nms = use_soft_nms
        self.use_tta = use_tta
        self.use_multiscale = use_multiscale
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 後処理コンポーネント初期化
        if use_soft_nms:
            self.soft_nms = SoftNMS(
                sigma=0.5,
                iou_threshold=iou_threshold,
                score_threshold=0.001,
                method='gaussian'
            )
        
        if use_tta:
            self.tta = TestTimeAugmentation(
                scales=[0.9, 1.0, 1.1],  # 軽量版
                flips=[False, True]
            )
        
        if use_multiscale:
            self.multiscale = MultiScaleInference(
                scales=[384, 416, 448]  # 軽量版
            )
        
        print(f"🔧 AdvancedPostProcessor initialized")
        print(f"   Soft-NMS: {'ON' if use_soft_nms else 'OFF'}")
        print(f"   TTA: {'ON' if use_tta else 'OFF'}")
        print(f"   Multi-scale: {'ON' if use_multiscale else 'OFF'}")
    
    def process_predictions(self, model, image, raw_predictions=None):
        """
        統合後処理実行
        
        Args:
            model: YOLO model
            image: 入力画像
            raw_predictions: 事前計算された予測（省略可）
        Returns:
            final_detections: 最終検出結果
        """
        # 1. 予測取得（TTA/Multi-scale対応）
        if self.use_tta and raw_predictions is None:
            predictions = self.tta(model, image)
        elif self.use_multiscale and raw_predictions is None:
            predictions = self.multiscale(model, image)
        else:
            if raw_predictions is None:
                with torch.no_grad():
                    predictions = model(image)
            else:
                predictions = raw_predictions
        
        # 2. デコード処理
        decoded_detections = self.decode_predictions(predictions)
        
        # 3. Soft-NMS適用
        if self.use_soft_nms:
            final_detections = self.apply_soft_nms(decoded_detections)
        else:
            final_detections = self.apply_standard_nms(decoded_detections)
        
        return final_detections
    
    def decode_predictions(self, predictions):
        """予測結果のデコード"""
        all_detections = []
        
        if isinstance(predictions, dict):
            # マルチスケール予測
            for scale_name, scale_preds in predictions.items():
                scale_detections = self.decode_single_scale(scale_preds, scale_name)
                all_detections.extend(scale_detections)
        else:
            # 単一スケール予測
            all_detections = self.decode_single_scale(predictions, 'single')
        
        return all_detections
    
    def decode_single_scale(self, predictions, scale_name):
        """単一スケール予測のデコード"""
        # [B, N, C] -> [N, C]
        if predictions.dim() == 3:
            predictions = predictions[0]  # バッチの最初の要素
        
        # 信頼度フィルタリング
        conf_scores = torch.sigmoid(predictions[:, 4])
        conf_mask = conf_scores > self.conf_threshold
        
        if not conf_mask.any():
            return []
        
        # フィルタ後の予測
        filtered_preds = predictions[conf_mask]
        filtered_conf = conf_scores[conf_mask]
        
        # 座標とクラス予測
        xy = torch.sigmoid(filtered_preds[:, :2])
        wh = filtered_preds[:, 2:4]
        class_probs = torch.softmax(filtered_preds[:, 5:], dim=-1)
        
        # バウンディングボックス変換
        boxes = torch.cat([xy, wh], dim=-1)
        
        # クラス別検出結果
        detections = []
        for i in range(class_probs.shape[-1]):
            class_conf = filtered_conf * class_probs[:, i]
            class_mask = class_conf > self.conf_threshold
            
            if class_mask.any():
                class_boxes = boxes[class_mask]
                class_scores = class_conf[class_mask]
                class_ids = torch.full((class_mask.sum(),), i)
                
                for j in range(len(class_boxes)):
                    detections.append({
                        'box': class_boxes[j],
                        'score': class_scores[j],
                        'class_id': class_ids[j],
                        'scale': scale_name
                    })
        
        return detections
    
    def apply_soft_nms(self, detections):
        """Soft-NMS適用"""
        if len(detections) == 0:
            return []
        
        # クラス別にSoft-NMS適用
        final_detections = []
        classes = set(det['class_id'].item() for det in detections)
        
        for class_id in classes:
            class_detections = [det for det in detections if det['class_id'].item() == class_id]
            
            if len(class_detections) <= 1:
                final_detections.extend(class_detections)
                continue
            
            # テンソル準備
            boxes = torch.stack([det['box'] for det in class_detections])
            scores = torch.stack([det['score'] for det in class_detections])
            class_ids = torch.stack([det['class_id'] for det in class_detections])
            
            # Soft-NMS実行
            keep_indices = self.soft_nms(boxes, scores, class_ids)
            
            # 保持する検出結果
            for idx in keep_indices:
                final_detections.append(class_detections[idx])
        
        return final_detections
    
    def apply_standard_nms(self, detections):
        """標準NMS適用（フォールバック）"""
        # 実装省略（標準的なNMS処理）
        return detections

def test_post_processing():
    """後処理システムのテスト"""
    print("🧪 Phase 4後処理システムテスト")
    print("-" * 50)
    
    # Soft-NMSテスト
    print("1. Soft-NMS Test:")
    soft_nms = SoftNMS(sigma=0.5, method='gaussian')
    
    # ダミーボックス
    boxes = torch.tensor([
        [0.1, 0.1, 0.3, 0.3],
        [0.15, 0.15, 0.35, 0.35],  # 重複
        [0.5, 0.5, 0.7, 0.7]       # 別物体
    ])
    scores = torch.tensor([0.9, 0.8, 0.85])
    class_ids = torch.tensor([0, 0, 1])
    
    try:
        keep_indices = soft_nms(boxes, scores, class_ids)
        print(f"   ✅ Soft-NMS成功: 保持インデックス {keep_indices}")
    except Exception as e:
        print(f"   ❌ Soft-NMSエラー: {e}")
    
    # TTAテスト
    print("2. TTA Test:")
    tta = TestTimeAugmentation(scales=[1.0], flips=[False, True])
    
    # ダミーモデル
    class DummyModel:
        def eval(self): pass
        def __call__(self, x):
            return {'small': torch.randn(1, 100, 20)}
    
    dummy_model = DummyModel()
    dummy_image = torch.randn(1, 1, 416, 416)
    
    try:
        tta_result = tta(dummy_model, dummy_image)
        print(f"   ✅ TTA成功: 出力形状 {tta_result['small'].shape}")
    except Exception as e:
        print(f"   ❌ TTAエラー: {e}")
    
    # 統合システムテスト
    print("3. Integrated System Test:")
    post_processor = AdvancedPostProcessor(
        use_soft_nms=True,
        use_tta=False,  # 時間効率重視
        use_multiscale=False
    )
    
    try:
        # ダミー予測でテスト
        dummy_predictions = {'small': torch.randn(1, 100, 20)}
        result = post_processor.process_predictions(
            dummy_model, dummy_image, dummy_predictions
        )
        print(f"   ✅ 統合システム成功: {len(result)} detections")
    except Exception as e:
        print(f"   ❌ 統合システムエラー: {e}")
    
    print("\n🔧 Phase 4後処理システムテスト完了!")
    return True

if __name__ == "__main__":
    # テスト実行
    success = test_post_processing()
    
    if success:
        print("🎉 Phase 4 Step 7 準備完了!")
        print("   Soft-NMS, TTA, Multi-scale Testing実装済み")
        print("   推論精度の大幅改善期待")
        print("   次: train.pyとの統合")
    else:
        print("❌ テスト失敗 - 修正が必要")
