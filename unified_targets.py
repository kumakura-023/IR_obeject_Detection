# 修正版 unified_targets.py
# Obj Loss異常値問題を解決

import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict, Optional

# ===== ユーティリティ関数 =====
def get_default_anchors() -> List[List[Tuple[int, int]]]:
    """デフォルトアンカー（YOLOv3ベース）"""
    return [
        [(10,13), (16,30), (33,23)],
        [(30,61), (62,45), (59,119)],
        [(116,90), (156,198), (373,326)]
    ]

def analyze_dataset_statistics(dataset, num_samples=1000, input_size=(640, 512)):
    """
    データセット統計分析とアンカー最適化
    """
    print(f"📊 Analyzing dataset statistics from {min(len(dataset), num_samples)} samples...")
    
    box_sizes_pixel = []
    aspect_ratios = []
    scale_distribution = {'small': 0, 'medium': 0, 'large': 0}
    class_distribution = defaultdict(int)
    
    start_time = time.time()
    processed = 0
    img_w, img_h = input_size
    
    for i in range(min(len(dataset), num_samples)):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            print(f"   Progress: {i}/{min(len(dataset), num_samples)} ({i/min(len(dataset), num_samples)*100:.1f}%) - {elapsed:.1f}s")
        
        try:
            _, targets = dataset[i]
            
            if torch.is_tensor(targets):
                if targets.is_cuda:
                    targets = targets.cpu()
                if targets.dim() == 1:
                    targets = targets.unsqueeze(0)
                elif targets.size(0) == 0:
                    continue
                    
                targets_np = targets.numpy()
                
                for j in range(targets_np.shape[0]):
                    target_data = targets_np[j]
                    if len(target_data) >= 4:
                        # 相対座標からピクセル座標へ変換
                        w_rel, h_rel = target_data[2], target_data[3]
                        if w_rel > 0 and h_rel > 0:
                            w_pixel = w_rel * img_w
                            h_pixel = h_rel * img_h
                            box_sizes_pixel.append((w_pixel, h_pixel))
                            
                            # アスペクト比
                            aspect_ratios.append(w_pixel / h_pixel)
                            
                            # サイズ分布
                            area = w_pixel * h_pixel
                            if area < 32*32:
                                scale_distribution['small'] += 1
                            elif area < 96*96:
                                scale_distribution['medium'] += 1
                            else:
                                scale_distribution['large'] += 1
                            
                            # クラス分布
                            if len(target_data) > 5:
                                class_scores = target_data[5:]
                                class_id = np.argmax(class_scores)
                                class_distribution[class_id] += 1
                            
                            processed += 1
            
        except Exception as e:
            if i < 5:
                print(f"⚠️ Sample {i} analysis failed: {e}")
            continue
    
    total_time = time.time() - start_time
    print(f"✅ Analysis completed: {processed} valid boxes found in {total_time:.1f}s")
    
    if len(box_sizes_pixel) == 0:
        print("❌ No valid boxes found for analysis")
        return None, defaultdict(int), {}
    
    box_sizes_np = np.array(box_sizes_pixel)
    
    # 統計情報計算
    statistics = {
        'total_boxes': len(box_sizes_np),
        'avg_width': np.mean(box_sizes_np[:, 0]),
        'avg_height': np.mean(box_sizes_np[:, 1]),
        'std_width': np.std(box_sizes_np[:, 0]),
        'std_height': np.std(box_sizes_np[:, 1]),
        'avg_aspect_ratio': np.mean(aspect_ratios),
        'scale_distribution': scale_distribution,
        'processing_time': total_time
    }
    
    # 統計表示
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total boxes: {statistics['total_boxes']}")
    print(f"   Avg size: {statistics['avg_width']:.1f} x {statistics['avg_height']:.1f} pixels")
    print(f"   Size distribution: Small={scale_distribution['small']}, Medium={scale_distribution['medium']}, Large={scale_distribution['large']}")
    
    # アンカー最適化
    print("\n🔧 Generating optimized anchors...")
    anchor_start = time.time()
    optimized_anchors_pixel = generate_anchors_kmeans(box_sizes_np, num_anchors=9)
    anchor_time = time.time() - anchor_start
    print(f"✅ Anchor optimization completed in {anchor_time:.1f}s")
    
    return optimized_anchors_pixel, dict(class_distribution), statistics


def generate_anchors_kmeans(box_sizes_pixel: np.ndarray, num_anchors: int = 9, max_iter: int = 300) -> List[List[Tuple[int, int]]]:
    """
    K-means++でアンカーを最適化
    """
    from sklearn.cluster import KMeans
    
    print(f"🔧 Generating {num_anchors} anchors using K-means++ (max {max_iter} iterations)...")
    
    if len(box_sizes_pixel) < num_anchors:
        print("⚠️ Not enough boxes, using default anchors")
        return get_default_anchors()
    
    # K-means++実行
    kmeans = KMeans(
        n_clusters=num_anchors, 
        init='k-means++',
        random_state=42, 
        max_iter=max_iter,
        n_init=10
    )
    clusters = kmeans.fit(box_sizes_pixel)
    
    # クラスタ中心を取得
    anchors_centers = clusters.cluster_centers_
    
    # 面積でソート
    areas = anchors_centers[:, 0] * anchors_centers[:, 1]
    sorted_indices = np.argsort(areas)
    sorted_anchors = anchors_centers[sorted_indices]
    
    # 3つのFPNレベルに分割
    anchors_per_level = num_anchors // 3
    optimized_anchors = []
    
    for level in range(3):
        start_idx = level * anchors_per_level
        if level == 2:  # 最後のレベルは残り全部
            end_idx = num_anchors
        else:
            end_idx = (level + 1) * anchors_per_level
            
        level_anchors = []
        for idx in range(start_idx, end_idx):
            if idx < len(sorted_anchors):
                w, h = sorted_anchors[idx]
                level_anchors.append((int(w), int(h)))
        
        # 不足分を補完
        while len(level_anchors) < anchors_per_level and level_anchors:
            level_anchors.append(level_anchors[-1])
        
        optimized_anchors.append(level_anchors)
    
    print("🎉 Generated optimized anchors:")
    for i, level in enumerate(optimized_anchors):
        print(f"   Level {i}: {level}")
    
    return optimized_anchors


def prepare_anchor_grid_info(anchors_pixel_per_level: List[List[Tuple[int, int]]], 
                           model_grid_sizes: List[Tuple[int, int]], 
                           input_size_wh: Tuple[int, int]) -> Dict:
    """
    アンカーとグリッド情報を準備
    """
    img_w, img_h = input_size_wh
    strides_per_level = []
    
    # ストライド計算
    for level_idx, (grid_h, grid_w) in enumerate(model_grid_sizes):
        stride = img_h // grid_h  # 通常は正方形ストライド
        strides_per_level.append(stride)
    
    return {
        'anchors_pixel_per_level': anchors_pixel_per_level,
        'strides_per_level': strides_per_level,
        'grid_sizes': model_grid_sizes,
        'input_size': input_size_wh
    }


def build_targets(targets: List[torch.Tensor], 
                  anchors_pixel_per_level: List[List[Tuple[int, int]]],
                  strides_per_level: List[int],
                  grid_sizes: List[Tuple[int, int]], 
                  input_size: Tuple[int, int],
                  num_classes: int, 
                  device: torch.device,
                  anchor_threshold: float = 0.2) -> Dict[str, torch.Tensor]:
    """
    ターゲット構築（修正版：IoU変調を緩和）
    """
    B = len(targets)
    img_w, img_h = input_size
    
    # 全予測数を計算
    total_predictions = 0
    for level_idx, (grid_h, grid_w) in enumerate(grid_sizes):
        num_anchors = len(anchors_pixel_per_level[level_idx])
        total_predictions += grid_h * grid_w * num_anchors
    
    # 統一ターゲットテンソル初期化
    unified_targets = torch.zeros((B, total_predictions, 5 + num_classes), 
                                 device=device, dtype=torch.float32)
    
    # 各バッチアイテムを処理
    for b_idx, batch_targets in enumerate(targets):
        if len(batch_targets) == 0:
            continue
            
        gt_data = batch_targets.to(device) if torch.is_tensor(batch_targets) else \
                  torch.tensor(batch_targets, device=device, dtype=torch.float32)
        
        if gt_data.size(0) == 0:
            continue
        
        # GT情報を抽出
        gt_boxes_rel = gt_data[:, :4]  # (cx, cy, w, h) in 0-1
        gt_objectness = gt_data[:, 4]
        gt_classes = gt_data[:, 5:] if gt_data.size(1) > 5 else torch.zeros((gt_data.size(0), num_classes), device=device)
        
        # 有効なGTのみフィルタ
        valid_mask = (gt_boxes_rel[:, 2] > 0) & (gt_boxes_rel[:, 3] > 0) & (gt_objectness > 0)
        if not valid_mask.any():
            continue
            
        gt_boxes_rel = gt_boxes_rel[valid_mask]
        gt_objectness = gt_objectness[valid_mask]
        gt_classes = gt_classes[valid_mask]
        
        # GTボックスをピクセル単位に変換
        gt_boxes_pixel = gt_boxes_rel * torch.tensor([img_w, img_h, img_w, img_h], device=device)
        
        # 各FPNレベルで処理
        current_offset = 0
        for level_idx, (grid_h, grid_w) in enumerate(grid_sizes):
            stride = strides_per_level[level_idx]
            anchors = torch.tensor(anchors_pixel_per_level[level_idx], device=device, dtype=torch.float32)
            num_anchors = len(anchors)
            
            # GT中心のグリッド座標を計算
            gt_cx = gt_boxes_pixel[:, 0]
            gt_cy = gt_boxes_pixel[:, 1]
            grid_x = (gt_cx / stride).long().clamp(0, grid_w - 1)
            grid_y = (gt_cy / stride).long().clamp(0, grid_h - 1)
            
            # 各GTに対してアンカーを割り当て
            for gt_idx in range(len(gt_boxes_pixel)):
                gt_w = gt_boxes_pixel[gt_idx, 2]
                gt_h = gt_boxes_pixel[gt_idx, 3]
                
                # IoU計算（簡易版）
                ious = []
                for anchor_w, anchor_h in anchors:
                    inter_w = min(gt_w, anchor_w)
                    inter_h = min(gt_h, anchor_h)
                    inter_area = inter_w * inter_h
                    union_area = gt_w * gt_h + anchor_w * anchor_h - inter_area
                    iou = inter_area / (union_area + 1e-8)
                    ious.append(iou)
                
                ious = torch.tensor(ious, device=device)
                
                # 閾値以上のアンカーまたは最良のアンカーを選択
                matched_anchors = torch.where(ious > anchor_threshold)[0]
                if len(matched_anchors) == 0:
                    matched_anchors = [ious.argmax().item()]
                
                # 各マッチしたアンカーに対してターゲットを設定
                for anchor_idx in matched_anchors:
                    # ターゲットインデックス計算
                    target_idx = current_offset + \
                                anchor_idx * grid_h * grid_w + \
                                grid_y[gt_idx] * grid_w + \
                                grid_x[gt_idx]
                    
                    if target_idx >= total_predictions:
                        continue
                    
                    # オフセット計算
                    tx = (gt_cx[gt_idx] / stride) - grid_x[gt_idx].float()
                    ty = (gt_cy[gt_idx] / stride) - grid_y[gt_idx].float()
                    
                    # スケール計算
                    anchor_w, anchor_h = anchors[anchor_idx]
                    tw = torch.log(gt_w / anchor_w + 1e-8)
                    th = torch.log(gt_h / anchor_h + 1e-8)
                    
                    # ===== 修正：IoU変調を緩和 =====
                    iou_score = ious[anchor_idx].item()
                    
                    # IoU変調を控えめにする
                    if iou_score > 0.7:
                        # 高IoUの場合は元の値をほぼ保持
                        obj_modulation = 1.0
                        cls_modulation = 1.0
                    elif iou_score > 0.5:
                        # 中IoUの場合は軽い変調
                        obj_modulation = 0.8 + 0.2 * iou_score
                        cls_modulation = 0.9
                    elif iou_score > 0.3:
                        # 低IoUの場合は適度な変調
                        obj_modulation = 0.6 + 0.4 * iou_score
                        cls_modulation = 0.7
                    else:
                        # 極低IoUの場合のみ強い変調
                        obj_modulation = iou_score
                        cls_modulation = iou_score * 0.5
                    
                    # ターゲット設定
                    unified_targets[b_idx, target_idx, 0] = tx
                    unified_targets[b_idx, target_idx, 1] = ty
                    unified_targets[b_idx, target_idx, 2] = tw
                    unified_targets[b_idx, target_idx, 3] = th
                    unified_targets[b_idx, target_idx, 4] = gt_objectness[gt_idx] * obj_modulation
                    unified_targets[b_idx, target_idx, 5:] = gt_classes[gt_idx] * cls_modulation
            
            current_offset += num_anchors * grid_h * grid_w
    
    return {
        'boxes': unified_targets[..., :4],      # tx, ty, tw, th
        'objectness': unified_targets[..., 4],  # 緩和されたIoU変調objectness
        'classes': unified_targets[..., 5:]     # 緩和されたIoU変調classes
    }


def evaluate_anchor_performance(box_sizes_pixel: np.ndarray, anchors_pixel_per_level: List[List[Tuple[int, int]]]) -> Dict:
    """
    アンカー性能を詳細評価
    """
    print("📈 Evaluating anchor performance...")
    
    # 全アンカーをフラット化
    all_anchors = []
    for level_anchors in anchors_pixel_per_level:
        all_anchors.extend(level_anchors)
    
    ious = []
    anchor_usage = defaultdict(int)
    
    for box_w, box_h in box_sizes_pixel:
        box_area = box_w * box_h
        best_iou = 0
        best_anchor_idx = -1
        
        for anchor_idx, (anchor_w, anchor_h) in enumerate(all_anchors):
            # IoU計算
            inter_w = min(box_w, anchor_w)
            inter_h = min(box_h, anchor_h)
            intersection = inter_w * inter_h
            
            anchor_area = anchor_w * anchor_h
            union = box_area + anchor_area - intersection
            iou = intersection / (union + 1e-8)
            
            if iou > best_iou:
                best_iou = iou
                best_anchor_idx = anchor_idx
        
        ious.append(best_iou)
        if best_anchor_idx >= 0:
            anchor_usage[best_anchor_idx] += 1
    
    # パフォーマンス指標計算
    ious_array = np.array(ious)
    performance = {
        'mean_iou': np.mean(ious_array),
        'std_iou': np.std(ious_array),
        'coverage_50': np.mean(ious_array > 0.5),
        'coverage_70': np.mean(ious_array > 0.7),
        'coverage_90': np.mean(ious_array > 0.9),
        'anchor_usage': dict(anchor_usage),
        'most_used_anchor': max(anchor_usage.items(), key=lambda x: x[1])[0] if anchor_usage else -1,
        'unused_anchors': len(all_anchors) - len(anchor_usage)
    }
    
    print(f"   平均IoU: {performance['mean_iou']:.4f} (±{performance['std_iou']:.4f})")
    print(f"   50%カバレッジ: {performance['coverage_50']:.2%}")
    print(f"   70%カバレッジ: {performance['coverage_70']:.2%}")
    print(f"   90%カバレッジ: {performance['coverage_90']:.2%}")
    print(f"   未使用アンカー: {performance['unused_anchors']}/{len(all_anchors)}")
    
    return performance


def evaluate_anchor_quality(dataset, anchors_pixel_per_level: List[List[Tuple[int, int]]], 
                          num_samples: int = 500, input_size: Tuple[int, int] = (640, 512)) -> Dict:
    """
    データセットでアンカー品質を評価
    """
    print(f"📏 Evaluating anchor quality with {num_samples} samples...")
    
    # まずデータセットからボックスサイズを収集
    box_sizes_pixel = []
    img_w, img_h = input_size
    
    for i in range(min(len(dataset), num_samples)):
        try:
            _, targets = dataset[i]
            if not torch.is_tensor(targets) or targets.size(0) == 0:
                continue
                
            targets_np = targets.cpu().numpy() if targets.is_cuda else targets.numpy()
            
            for target in targets_np:
                if len(target) >= 4:
                    w_rel, h_rel = target[2], target[3]
                    if w_rel > 0 and h_rel > 0:
                        w_pixel = w_rel * img_w
                        h_pixel = h_rel * img_h
                        box_sizes_pixel.append((w_pixel, h_pixel))
                        
        except Exception as e:
            continue
    
    if len(box_sizes_pixel) == 0:
        print("⚠️ No valid boxes for evaluation")
        return {"mean_iou": 0, "coverage_50": 0, "coverage_70": 0}
    
    # アンカー性能を評価
    box_sizes_np = np.array(box_sizes_pixel)
    performance = evaluate_anchor_performance(box_sizes_np, anchors_pixel_per_level)
    
    return performance


def compare_anchor_sets(dataset, anchors_set1: List[List[Tuple[int, int]]], 
                       anchors_set2: List[List[Tuple[int, int]]],
                       num_samples: int = 500, input_size: Tuple[int, int] = (640, 512)) -> Dict:
    """
    2つのアンカーセットを比較
    """
    print("\n🆚 Comparing anchor sets...")
    
    # それぞれ評価
    print("\n--- Anchor Set 1 (Default) ---")
    perf1 = evaluate_anchor_quality(dataset, anchors_set1, num_samples, input_size)
    
    print("\n--- Anchor Set 2 (Optimized) ---")
    perf2 = evaluate_anchor_quality(dataset, anchors_set2, num_samples, input_size)
    
    # 改善を計算
    comparison = {
        'set1_performance': perf1,
        'set2_performance': perf2,
        'improvement': {
            'mean_iou': perf2['mean_iou'] - perf1['mean_iou'],
            'coverage_50': perf2['coverage_50'] - perf1['coverage_50'],
            'coverage_70': perf2['coverage_70'] - perf1['coverage_70']
        }
    }
    
    print("\n📊 Improvement Summary:")
    print(f"   Mean IoU: {comparison['improvement']['mean_iou']:+.4f}")
    print(f"   50% Coverage: {comparison['improvement']['coverage_50']:+.2%}")
    print(f"   70% Coverage: {comparison['improvement']['coverage_70']:+.2%}")
    
    if comparison['improvement']['mean_iou'] > 0.05:
        print("   🎉 Significant improvement! Box Loss reduction expected.")
    elif comparison['improvement']['mean_iou'] > 0.02:
        print("   ✅ Good improvement achieved.")
    else:
        print("   ⚠️ Limited improvement. Consider different strategies.")
    
    return comparison