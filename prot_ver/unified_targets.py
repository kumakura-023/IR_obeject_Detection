# 修正版 unified_targets.py (v3)
# データ構造の不整合に柔軟に対応

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import time
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Optional


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
targets_version = create_version_tracker("Unified Targets System v3.2", "unified_targets.py")
targets_version.add_modification("アンカーマスクサイズ不整合修正 (264960 vs 88320)")
targets_version.add_modification("候補アンカー選定ロジック改善")
targets_version.add_modification("float32型統一でデータ型エラー修正")
targets_version.add_modification("段階的デバッグ出力追加")
targets_version.add_modification("変数スコープエラー修正 (num_anchors_per_grid)")

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
    # ★★★ バージョン追跡（修正版） ★★★
    VERSION = "3.4-clean-fix"
    
    # バージョン情報表示（初回のみ）
    if not hasattr(build_targets, '_version_printed'):
        print(f"\n{'='*60}")
        print(f"📋 Unified Targets System v{VERSION}")
        print(f"⏰ Clean Fix Applied: 2025-06-09 16:00")
        print(f"📝 Fixed: Variable scope, duplicate code, naming issues")
        print(f"{'='*60}\n")
        build_targets._version_printed = True
    
    B, N, C = predictions.shape
    device = predictions.device
    
    print(f"🔧 [TARGETS v{VERSION}] Processing: {predictions.shape}")
    
    # anchor_infoから必要な情報を取得
    grid_sizes = anchor_info['grid_sizes']
    input_w, input_h = anchor_info['input_size']
    
    # ★★★ 修正1: 変数を使用前に定義 ★★★
    # グリッドポイント数を計算
    num_grid_points_per_level = [h * w for h, w in grid_sizes]
    total_grid_points = sum(num_grid_points_per_level)
    num_anchors_per_grid = N // total_grid_points  # 自動検出
    
    # 妥当性チェック
    if num_anchors_per_grid <= 0:
        print(f"❌ Invalid num_anchors_per_grid: {num_anchors_per_grid}")
        print(f"   N: {N}, total_grid_points: {total_grid_points}")
        num_anchors_per_grid = 3  # フォールバック値

    # デバッグ出力（最小限）
    if B <= 2:
        print(f"🔧 Debug: total_grid_points = {total_grid_points}, num_anchors_per_grid = {num_anchors_per_grid}")

    # ★★★ 修正2: ストライド情報の処理 ★★★
    if 'strides' in anchor_info:
        strides_data = anchor_info['strides']
        if isinstance(strides_data, list):
            stride_values = strides_data
        else:
            stride_values = [input_h // gs[0] for gs in grid_sizes]
    else:
        stride_values = [input_h // gs[0] for gs in grid_sizes]
    
    # ★★★ 修正3: アンカーポイントの処理 ★★★
    if isinstance(anchor_info['anchor_points'], list):
        anchor_points_per_level = [ap.to(device) for ap in anchor_info['anchor_points']]
    else:
        # フォールバック：グリッドから計算
        anchor_points_per_level = []
        for h, w in grid_sizes:
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid = torch.stack((grid_x, grid_y), 2).view(-1, 2)
            anchor_points = (grid.float() + 0.5)
            anchor_points_per_level.append(anchor_points.to(device))
    
    # ★★★ 修正4: アンカーポイントの拡張 ★★★
    anchor_points_expanded = []
    strides_expanded = []
    
    for level_idx, (anchor_points, stride_val) in enumerate(zip(anchor_points_per_level, stride_values)):
        num_points = anchor_points.shape[0]
        
        for anchor_idx in range(num_anchors_per_grid):
            anchor_points_scaled = anchor_points * stride_val
            anchor_points_expanded.append(anchor_points_scaled)
            strides_expanded.append(torch.full((num_points,), stride_val, device=device))

    anchor_points_flat = torch.cat(anchor_points_expanded, dim=0)
    strides_flat = torch.cat(strides_expanded, dim=0)

    # ターゲットテンソル初期化
    target_boxes = torch.zeros((B, N, 4), device=device)
    target_obj = torch.zeros((B, N), device=device)
    target_cls = torch.zeros((B, N, num_classes), device=device)
    
    for b_idx in range(B):
        batch_targets = targets[b_idx]
        if len(batch_targets) == 0:
            continue

        gt_boxes_rel = batch_targets[:, :4]
        gt_boxes_abs = gt_boxes_rel * torch.tensor([input_w, input_h, input_w, input_h], device=device)
        gt_classes = batch_targets[:, 5:]
        num_gt = len(gt_boxes_abs)

        # ★★★ 修正5: 候補領域選定 ★★★
        gt_cx, gt_cy = gt_boxes_abs[:, 0], gt_boxes_abs[:, 1]
        
        is_in_center_list = []
        start_idx = 0
        
        for level_idx, stride_val in enumerate(stride_values):
            num_points_level = num_grid_points_per_level[level_idx]
            end_idx = start_idx + num_points_level * num_anchors_per_grid
            
            level_anchor_points = anchor_points_flat[start_idx:end_idx]
            
            offset = stride_val / 2
            x_lim = gt_cx.unsqueeze(1) + torch.tensor([-offset, offset], device=device).unsqueeze(0)
            y_lim = gt_cy.unsqueeze(1) + torch.tensor([-offset, offset], device=device).unsqueeze(0)
            
            anchor_x = level_anchor_points[:, 0].unsqueeze(0)
            anchor_y = level_anchor_points[:, 1].unsqueeze(0)
            
            is_in_x = (anchor_x > x_lim[:, 0].unsqueeze(1)) & (anchor_x < x_lim[:, 1].unsqueeze(1))
            is_in_y = (anchor_y > y_lim[:, 0].unsqueeze(1)) & (anchor_y < y_lim[:, 1].unsqueeze(1))
            
            is_in_center = is_in_x & is_in_y
            is_in_center_list.append(is_in_center.T)
            
            start_idx = end_idx

        is_in_center = torch.cat(is_in_center_list, dim=0)
        candidate_mask = is_in_center.any(dim=1)
        
        # デバッグ出力（最小限）
        if b_idx == 0:
            print(f"🔧 [v{VERSION}] Batch {b_idx}: candidates = {candidate_mask.sum()}")

        if not candidate_mask.any():
            print(f"⚠️ No candidates found for batch {b_idx}")
            continue

        # ★★★ 修正6: テンソル操作（クリーン版） ★★★
        try:
            # 予測値の抽出
            pred_logits = predictions[b_idx][candidate_mask]
            pred_boxes = pred_logits[:, :4].float()
            pred_obj = pred_logits[:, 4]
            pred_cls = pred_logits[:, 5:]
            
            # アンカーポイントの抽出
            candidate_anchor_points = anchor_points_flat[candidate_mask].float()
            candidate_strides = strides_flat[candidate_mask].float()
            
            # ボックスデコード
            decoded_pred_boxes = torch.cat(
                (
                    (pred_boxes[:, :2].sigmoid() * 2 - 0.5 + candidate_anchor_points) * candidate_strides.unsqueeze(1),
                    (pred_boxes[:, 2:].sigmoid() * 2)**2 * candidate_strides.unsqueeze(1).repeat(1, 2)
                ),
                dim=-1
            )
            
            # IoU計算
            gt_boxes_abs_float = gt_boxes_abs.float()
            pair_wise_iou = get_ious(gt_boxes_abs_float, decoded_pred_boxes, box_format='xywh')
            iou_cost = -torch.log(pair_wise_iou + 1e-8)
            
            # クラス分類コスト
            gt_cls_matrix = F.one_hot(torch.argmax(gt_classes, dim=1), num_classes).float()
            pred_cls_sigmoid = pred_cls.sigmoid().float()
            
            pred_cls_input = pred_cls_sigmoid.unsqueeze(0).repeat(num_gt, 1, 1).to(device).float()
            gt_cls_input = gt_cls_matrix.unsqueeze(1).repeat(1, pred_cls_sigmoid.shape[0], 1).to(device).float()
            
            cls_cost = F.binary_cross_entropy(
                pred_cls_input,
                gt_cls_input,
                reduction='none'
            ).sum(-1)
            
            # コストマトリックス計算
            is_in_center_candidates = is_in_center[candidate_mask]
            cost_matrix = cls_cost + 3.0 * iou_cost + 100000.0 * (~is_in_center_candidates.T)
            
            if b_idx == 0:
                print(f"🎉 [v{VERSION}] Cost matrix computed: {cost_matrix.shape}")
            
            # ★★★ 修正7: Dynamic K マッチング ★★★
            n_candidate_k = min(topk_candidates, pred_boxes.shape[0])
            if n_candidate_k > 0:
                topk_ious, _ = torch.topk(pair_wise_iou, n_candidate_k, dim=1)
                dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
                
                matching_matrix = torch.zeros_like(cost_matrix)
                for gt_i in range(num_gt):
                    if dynamic_ks[gt_i] > 0:
                        _, pos_idx = torch.topk(cost_matrix[gt_i], k=dynamic_ks[gt_i], largest=False)
                        matching_matrix[gt_i, pos_idx] = 1.0

                # 競合解決
                anchor_matching_gt = matching_matrix.sum(0)
                if (anchor_matching_gt > 1).any():
                    conflicting_indices = torch.where(anchor_matching_gt > 1)[0]
                    cost_matrix_conflict = cost_matrix[:, conflicting_indices]
                    _, cost_argmin = torch.min(cost_matrix_conflict, dim=0)
                    
                    matching_matrix[:, conflicting_indices] = 0.0
                    matching_matrix[cost_argmin, conflicting_indices] = 1.0
                
                # ターゲット割り当て
                fg_mask_in_cand = (matching_matrix.sum(0) > 0)
                if fg_mask_in_cand.any():
                    matched_gt_inds = matching_matrix[:, fg_mask_in_cand].argmax(0)
                    
                    candidate_indices = torch.where(candidate_mask)[0]
                    final_pos_indices = candidate_indices[fg_mask_in_cand]
                    
                    target_obj[b_idx, final_pos_indices] = 1.0
                    target_cls[b_idx, final_pos_indices] = gt_classes[matched_gt_inds]
                    target_boxes[b_idx, final_pos_indices] = gt_boxes_abs[matched_gt_inds]
            
        except Exception as step_error:
            print(f"❌ [v{VERSION}] Error in batch {b_idx}: {step_error}")
            print(f"   Error type: {type(step_error).__name__}")
            continue

    # ターゲットボックスを正規化
    normalizer = torch.tensor([input_w, input_h, input_w, input_h], device=device).unsqueeze(0).unsqueeze(0)
    target_boxes /= (normalizer + 1e-6)

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