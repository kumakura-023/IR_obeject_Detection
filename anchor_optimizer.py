# anchor_optimizer.py - FLIRデータセット用アンカー最適化
# プロジェクトルートに配置

import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import pickle


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

def analyze_flir_dataset_for_anchors(dataset, num_samples=1000, save_results=True):
    """
    FLIRデータセットのボックス分布分析
    """
    print(f"🔍 FLIRデータセット分析開始 ({num_samples}サンプル)...")
    
    box_sizes_pixel = []
    class_distribution = defaultdict(int)
    
    for i in range(min(len(dataset), num_samples)):
        if i % 100 == 0:
            print(f"   進捗: {i}/{num_samples}")
            
        try:
            _, targets = dataset[i]
            if torch.is_tensor(targets) and targets.size(0) > 0:
                targets_np = targets.cpu().numpy()
                
                for target in targets_np:
                    if len(target) >= 4:
                        w_rel, h_rel = target[2], target[3]
                        if w_rel > 0 and h_rel > 0:
                            # FLIR: 640x512
                            w_pixel = w_rel * 640
                            h_pixel = h_rel * 512
                            box_sizes_pixel.append((w_pixel, h_pixel))
                            
                            # クラス分析
                            if len(target) > 5:
                                class_scores = target[5:]
                                class_id = np.argmax(class_scores)
                                class_distribution[class_id] += 1
                                
        except Exception as e:
            if i < 5:
                print(f"⚠️ Sample {i} エラー: {e}")
            continue
    
    box_array = np.array(box_sizes_pixel)
    
    # 統計計算
    stats = {
        'total_boxes': len(box_sizes_pixel),
        'width_stats': {
            'min': box_array[:, 0].min(),
            'max': box_array[:, 0].max(), 
            'mean': box_array[:, 0].mean(),
            'std': box_array[:, 0].std()
        },
        'height_stats': {
            'min': box_array[:, 1].min(),
            'max': box_array[:, 1].max(),
            'mean': box_array[:, 1].mean(), 
            'std': box_array[:, 1].std()
        },
        'class_distribution': dict(class_distribution)
    }
    
    # 面積ベース分類
    areas = box_array[:, 0] * box_array[:, 1]
    small_count = np.sum(areas < 32*32)
    medium_count = np.sum((areas >= 32*32) & (areas < 96*96))
    large_count = np.sum(areas >= 96*96)
    
    stats['size_distribution'] = {
        'small': {'count': small_count, 'ratio': small_count/len(areas)},
        'medium': {'count': medium_count, 'ratio': medium_count/len(areas)},
        'large': {'count': large_count, 'ratio': large_count/len(areas)}
    }
    
    print(f"\n📊 分析結果:")
    print(f"総ボックス数: {stats['total_boxes']}")
    print(f"幅: {stats['width_stats']['min']:.1f} - {stats['width_stats']['max']:.1f} (平均: {stats['width_stats']['mean']:.1f})")
    print(f"高さ: {stats['height_stats']['min']:.1f} - {stats['height_stats']['max']:.1f} (平均: {stats['height_stats']['mean']:.1f})")
    print(f"Small(<32²): {small_count} ({small_count/len(areas)*100:.1f}%)")
    print(f"Medium(32²-96²): {medium_count} ({medium_count/len(areas)*100:.1f}%)")
    print(f"Large(>96²): {large_count} ({large_count/len(areas)*100:.1f}%)")
    
    if save_results:
        # 結果保存
        with open('flir_analysis_results.pkl', 'wb') as f:
            pickle.dump({'box_sizes': box_array, 'stats': stats}, f)
        print("✅ 分析結果を flir_analysis_results.pkl に保存")
    
    return box_array, stats

def optimize_anchors_for_flir(box_sizes, num_anchors=9, max_iter=300):
    """
    K-means++でアンカー最適化
    """
    print(f"🎯 アンカー最適化開始 (K={num_anchors})...")
    
    if len(box_sizes) < num_anchors:
        print("⚠️ ボックス数不足、デフォルト使用")
        return get_default_anchors()
    
    # K-means実行
    kmeans = KMeans(n_clusters=num_anchors, random_state=42, max_iter=max_iter, n_init=10)
    clusters = kmeans.fit(box_sizes)
    
    anchors_centers = clusters.cluster_centers_
    
    # 面積でソートして3レベルに分割
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
    
    print(f"✅ アンカー最適化完了:")
    for i, level in enumerate(optimized_anchors):
        print(f"  Level {i}: {level}")
    
    return optimized_anchors

def evaluate_anchor_performance(box_sizes, anchors_per_level):
    """
    アンカー性能評価
    """
    print("📈 アンカー性能評価中...")
    
    all_anchors = []
    for level in anchors_per_level:
        all_anchors.extend(level)
    
    ious = []
    anchor_usage = defaultdict(int)
    
    for box_w, box_h in box_sizes:
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
    
    performance = {
        'mean_iou': np.mean(ious),
        'coverage_50': np.mean(np.array(ious) > 0.5),
        'coverage_70': np.mean(np.array(ious) > 0.7),
        'anchor_usage': dict(anchor_usage)
    }
    
    print(f"平均IoU: {performance['mean_iou']:.4f}")
    print(f"50%カバレッジ: {performance['coverage_50']:.2%}")
    print(f"70%カバレッジ: {performance['coverage_70']:.2%}")
    
    return performance

def get_default_anchors():
    """デフォルトアンカー"""
    return [
        [(10,13), (16,30), (33,23)],
        [(30,61), (62,45), (59,119)],
        [(116,90), (156,198), (373,326)]
    ]

def save_optimized_anchors(anchors, filename="optimized_anchors.py"):
    """
    最適化アンカーをPythonファイルとして保存
    """
    content = f'''# FLIRデータセット用最適化アンカー
# 自動生成: anchor_optimizer.py

OPTIMIZED_ANCHORS = {anchors}

def get_flir_anchors():
    """FLIRデータセット用最適化アンカーを取得"""
    return OPTIMIZED_ANCHORS

if __name__ == "__main__":
    print("FLIR最適化アンカー:")
    for i, level in enumerate(OPTIMIZED_ANCHORS):
        print(f"Level {{i}}: {{level}}")
'''
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 最適化アンカーを {filename} に保存")

def compare_with_default(box_sizes):
    """
    デフォルトアンカーとの比較
    """
    print("\n🆚 デフォルトアンカーとの比較:")
    
    default_anchors = get_default_anchors()
    optimized_anchors = optimize_anchors_for_flir(box_sizes)
    
    print("\n--- デフォルトアンカー性能 ---")
    default_perf = evaluate_anchor_performance(box_sizes, default_anchors)
    
    print("\n--- 最適化アンカー性能 ---")
    optimized_perf = evaluate_anchor_performance(box_sizes, optimized_anchors)
    
    print("\n📊 改善効果:")
    iou_improvement = optimized_perf['mean_iou'] - default_perf['mean_iou']
    coverage_50_improvement = optimized_perf['coverage_50'] - default_perf['coverage_50']
    coverage_70_improvement = optimized_perf['coverage_70'] - default_perf['coverage_70']
    
    print(f"平均IoU改善: {iou_improvement:+.4f}")
    print(f"50%カバレッジ改善: {coverage_50_improvement:+.2%}")
    print(f"70%カバレッジ改善: {coverage_70_improvement:+.2%}")
    
    if iou_improvement > 0.05:
        print("🎉 大幅改善! Box Loss向上が期待できます")
    elif iou_improvement > 0.02:
        print("✅ 改善効果あり")
    else:
        print("⚠️ 改善効果は限定的")
    
    return optimized_anchors, {'default': default_perf, 'optimized': optimized_perf}

def main():
    """
    アンカー最適化メイン実行
    """
    print("🚀 FLIRデータセット用アンカー最適化")
    print("="*50)
    
    try:
        # データセット読み込み
        from dataset import YoloInfraredDataset
        
        dataset = YoloInfraredDataset(
            image_dir="/content/FLIR_YOLO_local/images/train",
            label_dir="/content/FLIR_YOLO_local/labels/train",
            input_size=(640, 512),
            num_classes=15
        )
        
        print(f"📁 データセット読み込み: {len(dataset)}枚")
        
        # 1. データセット分析
        box_sizes, stats = analyze_flir_dataset_for_anchors(dataset, num_samples=1000)
        
        if len(box_sizes) == 0:
            print("❌ 有効なボックスが見つかりません")
            return
        
        # 2. アンカー最適化と比較
        optimized_anchors, comparison = compare_with_default(box_sizes)
        
        # 3. 結果保存
        save_optimized_anchors(optimized_anchors, "optimized_anchors.py")
        
        print(f"\n🎯 最適化完了!")
        print(f"次のステップ:")
        print(f"1. enhanced_training_warm_scheduler_gemini.py でアンカーを使用")
        print(f"2. from optimized_anchors import get_flir_anchors")
        print(f"3. optimized_anchors_pixel = get_flir_anchors()")
        
        return optimized_anchors, stats, comparison
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    main()