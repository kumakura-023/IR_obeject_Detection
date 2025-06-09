# 修正版 unified_targets.py
# Obj Loss異常値問題を解決

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import time
from scipy.optimize import linear_sum_assignment
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


def build_targets(
    predictions: torch.Tensor,
    targets: List[torch.Tensor],
    anchor_info: Dict,
    num_classes: int,
    topk_candidates: int = 10,
    iou_threshold: float = 0.5,
):
    
    B, N, C = predictions.shape
    device = predictions.device
    
    # アンカーグリッド情報
    anchor_points = anchor_info['anchor_points']
    strides = anchor_info['strides']
    
    # ターゲットテンソル初期化
    target_boxes = torch.zeros((B, N, 4), device=device)
    target_obj = torch.zeros((B, N), device=device)
    target_cls = torch.zeros((B, N, num_classes), device=device)
    
    for b_idx in range(B):
        batch_targets = targets[b_idx]
        if len(batch_targets) == 0:
            continue

        gt_boxes = batch_targets[:, :4] * torch.tensor([*anchor_info['input_size']], device=device).repeat(2)
        gt_classes = batch_targets[:, 5:]
        num_gt = len(gt_boxes)

        # 1. 候補領域の選定
        is_in_box_list = []
        is_in_center_list = []
        
        for grid_idx, (grid_h, grid_w) in enumerate(anchor_info['grid_sizes']):
            stride = strides[grid_idx]
            anchor_grid = anchor_points[grid_idx]
            
            # GTボックスの中心がどのグリッドセル内にあるか
            gt_cx, gt_cy = gt_boxes[:, 0], gt_boxes[:, 1]
            
            # 中心から半径stride/2の正方形を候補領域とする
            x_lim = gt_cx.unsqueeze(1) + torch.stack([-stride/2, stride/2], dim=-1)
            y_lim = gt_cy.unsqueeze(1) + torch.stack([-stride/2, stride/2], dim=-1)
            
            is_in_center = (anchor_grid[:, 0] > x_lim[:, 0]) & (anchor_grid[:, 0] < x_lim[:, 1]) & \
                           (anchor_grid[:, 1] > y_lim[:, 0]) & (anchor_grid[:, 1] < y_lim[:, 1])
            
            is_in_center_list.append(is_in_center.T)
        
        is_in_center = torch.cat(is_in_center_list, dim=0)
        
        # 2. コスト計算
        pred_logits = predictions[b_idx]
        pred_boxes = pred_logits[:, :4]
        pred_obj = pred_logits[:, 4]
        pred_cls = pred_logits[:, 5:]
        
        candidate_mask = is_in_center.any(dim=1)
        candidate_preds_box = pred_boxes[candidate_mask]
        candidate_preds_obj = pred_obj[candidate_mask]
        candidate_preds_cls = pred_cls[candidate_mask]
        
        # IoUコスト
        pair_wise_iou = get_ious(gt_boxes, candidate_preds_box, box_format='xywh')
        iou_cost = -torch.log(pair_wise_iou + 1e-8)
        
        # クラス分類コスト
        gt_cls_matrix = F.one_hot(torch.argmax(gt_classes, dim=1), num_classes).float()
        pred_cls_sigmoid = candidate_preds_cls.sigmoid()
        
        cls_cost = F.binary_cross_entropy(
            pred_cls_sigmoid.unsqueeze(0).repeat(num_gt, 1, 1),
            gt_cls_matrix.unsqueeze(1).repeat(1, len(candidate_preds_cls), 1),
            reduction='none'
        ).sum(-1)
        
        cost_matrix = cls_cost + 3.0 * iou_cost
        
        # 3. Dynamic K マッチング
        matching_matrix = torch.zeros_like(cost_matrix)
        
        # 各GTに対して、最もコストの低い10個の候補を選択
        n_candidate_k = min(topk_candidates, len(candidate_preds_box))
        topk_ious, _ = torch.topk(pair_wise_iou, n_candidate_k, dim=1)
        
        # 各GTのkを動的に決定
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        
        for gt_i in range(num_gt):
            _, pos_idx = torch.topk(cost_matrix[gt_i], k=dynamic_ks[gt_i], largest=False)
            matching_matrix[gt_i, pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        # 4. ターゲット割り当て
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).any():
            _, cost_argmin = torch.min(cost_matrix[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        
        fg_mask_in_cand = (matching_matrix.sum(0) > 0)
        matched_gt_inds = matching_matrix[:, fg_mask_in_cand].argmax(0)
        
        candidate_indices = torch.where(candidate_mask)[0]
        final_pos_indices = candidate_indices[fg_mask_in_cand]
        
        # 最終的なターゲットを作成
        target_obj[b_idx, final_pos_indices] = 1.0
        target_cls[b_idx, final_pos_indices] = gt_classes[matched_gt_inds]
        target_boxes[b_idx, final_pos_indices] = gt_boxes[matched_gt_inds]

    return {
        'boxes': target_boxes,
        'objectness': target_obj,
        'classes': target_cls
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

def get_default_anchors() -> List[List[Tuple[int, int]]]:
    """デフォルトアンカー（YOLOv3ベース）"""
    return [
        [(10,13), (16,30), (33,23)],
        [(30,61), (62,45), (59,119)],
        [(116,90), (156,198), (373,326)]
    ]

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def get_ious(bboxes1, bboxes2, box_format='xyxy', iou_type='iou'):
    if bboxes1.shape[0] == 0 or bboxes2.shape[0] == 0:
        return torch.zeros(bboxes1.shape[0], bboxes2.shape[0], device=bboxes1.device)
        
    if box_format == 'xywh':
        bboxes1 = xywh2xyxy(bboxes1)
        bboxes2 = xywh2xyxy(bboxes2)

    b1_x1, b1_y1, b1_x2, b1_y2 = bboxes1.split(1, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = bboxes2.split(1, -1)
    
    inter_x1 = torch.max(b1_x1, b2_x1.transpose(0, 1))
    inter_y1 = torch.max(b1_y1, b2_y1.transpose(0, 1))
    inter_x2 = torch.min(b1_x2, b2_x2.transpose(0, 1))
    inter_y2 = torch.min(b1_y2, b2_y2.transpose(0, 1))

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = area1 + area2.transpose(0, 1) - inter_area
    
    return inter_area / (union_area + 1e-6)