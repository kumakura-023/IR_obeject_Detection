# anchor_generator.py - Step 1: データセット固有アンカー生成

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from dataset import FLIRDataset
from config import Config


# ★★★ 共有VersionTrackerをインポート ★★★
from version_tracker import create_version_tracker, VersionTracker

# バージョン管理システム初期化
loss_version = create_version_tracker("Loss System v1.1", "anchor_generator.py")
loss_version.add_modification("アンカー生成実装")

def generate_anchors_kmeans(dataset, k=9, img_size=416, sample_limit=1000):
    """
    データセット固有のアンカーをK-means++で生成
    
    Args:
        dataset: FLIRDataset
        k: アンカー数（9個: 3スケール × 3アンカー）
        img_size: 画像サイズ
        sample_limit: 処理するサンプル数上限（高速化）
    
    Returns:
        anchors_dict: スケール別アンカー辞書
        stats: 生成統計情報
    """
    print(f"🔍 Step 1: データセット固有アンカー生成開始")
    print(f"   データセットサイズ: {len(dataset)}")
    print(f"   サンプル制限: {min(len(dataset), sample_limit)}")
    print(f"   アンカー数: {k}")
    print("-" * 50)
    
    # 1. データセットからバウンディングボックス収集
    all_boxes = []
    sample_count = 0
    valid_boxes = 0
    
    for i in range(min(len(dataset), sample_limit)):
        try:
            _, targets = dataset[i]
            
            if len(targets) > 0:
                for target in targets:
                    # target format: [class, cx, cy, w, h] (normalized)
                    if len(target) >= 5:
                        w, h = target[3].item(), target[4].item()
                        
                        # 正常値チェック
                        if 0 < w < 1 and 0 < h < 1:
                            # 実際のピクセルサイズに変換
                            w_pixels = w * img_size
                            h_pixels = h * img_size
                            
                            all_boxes.append([w_pixels, h_pixels])
                            valid_boxes += 1
            
            sample_count += 1
            
            # 進捗表示
            if (i + 1) % 100 == 0:
                print(f"   処理済み: {i + 1}/{min(len(dataset), sample_limit)}, "
                      f"有効ボックス: {valid_boxes}")
                
        except Exception as e:
            print(f"   警告: サンプル{i}でエラー: {e}")
            continue
    
    if len(all_boxes) == 0:
        raise ValueError("有効なバウンディングボックスが見つかりません")
    
    all_boxes = np.array(all_boxes)
    print(f"\n📊 収集結果:")
    print(f"   処理サンプル数: {sample_count}")
    print(f"   有効ボックス数: {len(all_boxes)}")
    print(f"   幅の範囲: {all_boxes[:, 0].min():.1f} - {all_boxes[:, 0].max():.1f} pixels")
    print(f"   高さの範囲: {all_boxes[:, 1].min():.1f} - {all_boxes[:, 1].max():.1f} pixels")
    
    # 2. K-means++でクラスタリング
    print(f"\n🔄 K-means++クラスタリング実行中...")
    
    try:
        kmeans = KMeans(
            n_clusters=k, 
            init='k-means++', 
            n_init=10, 
            random_state=42,
            max_iter=300
        )
        kmeans.fit(all_boxes)
        
        # クラスタ中心をアンカーとして使用
        anchors = kmeans.cluster_centers_
        
        print(f"   クラスタリング完了")
        print(f"   収束: {kmeans.n_iter_}回反復")
        print(f"   慣性: {kmeans.inertia_:.2f}")
        
    except Exception as e:
        print(f"❌ K-meansエラー: {e}")
        # フォールバック: 手動でアンカー生成
        print("   フォールバック: 手動アンカー生成")
        anchors = generate_fallback_anchors(all_boxes, k)
    
    # 3. アンカーを面積順にソート
    areas = anchors[:, 0] * anchors[:, 1]
    sorted_indices = np.argsort(areas)
    anchors = anchors[sorted_indices]
    
    # 4. スケール別に分割
    anchors_dict = organize_anchors_by_scale(anchors)
    
    # 5. 統計情報生成
    stats = calculate_anchor_stats(all_boxes, anchors_dict)
    
    print(f"\n✅ アンカー生成完了!")
    print_anchor_results(anchors_dict, stats)
    
    return anchors_dict, stats

def generate_fallback_anchors(boxes, k=9):
    """K-means失敗時のフォールバックアンカー生成"""
    print("   フォールバック: 分位数ベースのアンカー生成")
    
    # 幅と高さの分位数を計算
    widths = boxes[:, 0]
    heights = boxes[:, 1]
    
    anchors = []
    for i in range(k):
        # 分位数ベースでアンカーサイズを決定
        w_percentile = (i + 1) * (100 / (k + 1))
        h_percentile = (i + 1) * (100 / (k + 1))
        
        w = np.percentile(widths, w_percentile)
        h = np.percentile(heights, h_percentile)
        
        anchors.append([w, h])
    
    return np.array(anchors)

def organize_anchors_by_scale(anchors):
    """9個のアンカーを3つのスケール（小・中・大）に分割"""
    
    if len(anchors) != 9:
        raise ValueError(f"アンカー数は9個である必要があります。現在: {len(anchors)}")
    
    # 面積で3グループに分割
    small_anchors = [(int(w), int(h)) for w, h in anchors[:3]]    # 最小の3個
    medium_anchors = [(int(w), int(h)) for w, h in anchors[3:6]]  # 中間の3個
    large_anchors = [(int(w), int(h)) for w, h in anchors[6:9]]   # 最大の3個
    
    organized_anchors = {
        'small': small_anchors,   # 52x52 グリッド用
        'medium': medium_anchors, # 26x26 グリッド用  
        'large': large_anchors    # 13x13 グリッド用
    }
    
    return organized_anchors

def calculate_iou_wh(box1_wh, box2_wh):
    """幅・高さのみでIoU計算（中心は同じと仮定）"""
    w1, h1 = box1_wh
    w2, h2 = box2_wh
    
    # 交差領域（中心が同じなので、小さい方の幅・高さ）
    intersection = min(w1, w2) * min(h1, h2)
    
    # 和領域
    union = w1 * h1 + w2 * h2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_anchor_stats(boxes, anchors_dict):
    """アンカーの品質統計を計算"""
    print(f"\n📊 アンカー品質評価中...")
    
    all_ious = []
    anchor_usage = {scale: [0, 0, 0] for scale in ['small', 'medium', 'large']}
    scale_list = ['small', 'medium', 'large']
    
    # サンプル数制限（計算高速化）
    sample_boxes = boxes[:min(len(boxes), 500)]
    
    for box_wh in sample_boxes:
        best_iou = 0
        best_scale = None
        best_anchor_idx = None
        
        # 全アンカーと比較
        for scale in scale_list:
            for anchor_idx, anchor_wh in enumerate(anchors_dict[scale]):
                iou = calculate_iou_wh(box_wh, anchor_wh)
                
                if iou > best_iou:
                    best_iou = iou
                    best_scale = scale
                    best_anchor_idx = anchor_idx
        
        all_ious.append(best_iou)
        if best_scale and best_anchor_idx is not None:
            anchor_usage[best_scale][best_anchor_idx] += 1
    
    stats = {
        'avg_iou': np.mean(all_ious),
        'iou_50': np.mean(np.array(all_ious) > 0.5),
        'iou_70': np.mean(np.array(all_ious) > 0.7),
        'anchor_usage': anchor_usage,
        'total_boxes': len(boxes),
        'evaluated_boxes': len(sample_boxes)
    }
    
    return stats

def print_anchor_results(anchors_dict, stats):
    """アンカー生成結果を表示"""
    print(f"\n📋 生成されたアンカー:")
    for scale, anchor_list in anchors_dict.items():
        print(f"   {scale:6s} (grid): {anchor_list}")
        areas = [w * h for w, h in anchor_list]
        print(f"   {' ':6s}  (area): {areas}")
    
    print(f"\n📊 品質評価:")
    print(f"   平均IoU: {stats['avg_iou']:.3f}")
    print(f"   IoU > 0.5: {stats['iou_50']:.1%}")
    print(f"   IoU > 0.7: {stats['iou_70']:.1%}")
    
    print(f"\n📈 アンカー使用頻度:")
    for scale, usage in stats['anchor_usage'].items():
        total = sum(usage)
        if total > 0:
            percentages = [f"{u/total*100:.1f}%" for u in usage]
            print(f"   {scale:6s}: {usage} → {percentages}")

def visualize_anchors(anchors_dict, save_path="step1_anchors_visualization.png"):
    """アンカーを可視化"""
    print(f"\n📊 アンカー可視化中...")
    
    plt.figure(figsize=(15, 5))
    
    scales = ['small', 'medium', 'large']
    grid_sizes = [52, 26, 13]
    colors = ['red', 'green', 'blue']
    
    for i, (scale, grid_size) in enumerate(zip(scales, grid_sizes)):
        plt.subplot(1, 3, i+1)
        
        anchor_set = anchors_dict[scale]
        
        # グリッドセルサイズ
        cell_size = 416 // grid_size
        
        for j, (w, h) in enumerate(anchor_set):
            # アンカーボックスを描画
            rect = plt.Rectangle(
                (cell_size//2 - w//2, cell_size//2 - h//2), 
                w, h, 
                linewidth=2, 
                edgecolor=colors[j], 
                facecolor='none',
                label=f'Anchor {j+1}: ({w}, {h})'
            )
            plt.gca().add_patch(rect)
        
        plt.xlim(0, cell_size)
        plt.ylim(0, cell_size)
        plt.gca().invert_yaxis()
        plt.title(f'{scale.capitalize()} Scale\n({grid_size}x{grid_size} grid)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"   可視化保存: {save_path}")

def save_anchors_config(anchors_dict, stats, save_path="step1_anchors_config.py"):
    """config.py用のアンカー設定を保存"""
    
    config_text = f'''# ===== Step 1: 生成されたアンカー設定 =====
# 生成日時: {import_datetime()}

# データセット固有アンカー
use_anchors = True
anchors = {{
    'small':  {anchors_dict['small']},   # 52x52 grid (小物体用)
    'medium': {anchors_dict['medium']},  # 26x26 grid (中物体用)
    'large':  {anchors_dict['large']}    # 13x13 grid (大物体用)
}}

# アンカー品質統計
anchor_quality = {{
    'avg_iou': {stats['avg_iou']:.3f},
    'iou_50_percent': {stats['iou_50']:.3f},
    'iou_70_percent': {stats['iou_70']:.3f},
    'total_boxes_analyzed': {stats['total_boxes']},
    'generation_method': 'K-means++',
    'dataset_specific': True
}}

print(f"📊 Step 1アンカー設定読み込み完了")
print(f"   平均IoU: {{anchor_quality['avg_iou']}}")
print(f"   IoU>0.5: {{anchor_quality['iou_50_percent']:.1%}}")
'''
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(config_text)
    
    print(f"💾 設定ファイル保存: {save_path}")
    
    return config_text

def import_datetime():
    """現在時刻を取得"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ===== Step 1実行関数 =====
def run_step1():
    """Step 1を実行"""
    print("🚀 Step 1: アンカー生成開始")
    print("=" * 60)
    
    try:
        # 1. 設定とデータセット読み込み
        cfg = Config()
        print(f"📋 設定読み込み完了")
        print(f"   画像サイズ: {cfg.img_size}")
        print(f"   クラス数: {cfg.num_classes}")
        
        # 2. データセット作成
        dataset = FLIRDataset(
            cfg.train_img_dir, 
            cfg.train_label_dir, 
            cfg.img_size, 
            augment=False  # アンカー生成時は拡張なし
        )
        print(f"📊 データセット読み込み完了: {len(dataset)}枚")
        
        # 3. アンカー生成
        anchors_dict, stats = generate_anchors_kmeans(dataset, k=9, img_size=cfg.img_size)
        
        # 4. 可視化
        visualize_anchors(anchors_dict)
        
        # 5. 設定保存
        config_text = save_anchors_config(anchors_dict, stats)
        
        # 6. 結果サマリー
        print("\n" + "=" * 60)
        print("✅ Step 1完了!")
        print("=" * 60)
        print(f"📊 生成結果サマリー:")
        print(f"   平均IoU: {stats['avg_iou']:.3f}")
        print(f"   IoU>0.5率: {stats['iou_50']:.1%}")
        print(f"   評価ボックス数: {stats['evaluated_boxes']}")
        
        if stats['avg_iou'] > 0.4:
            print("🎉 高品質なアンカーが生成されました!")
        elif stats['avg_iou'] > 0.3:
            print("✅ 良好なアンカーが生成されました")
        else:
            print("⚠️ アンカー品質が低い可能性があります")
        
        print(f"\n📝 次のステップ:")
        print(f"   1. 可視化結果を確認")
        print(f"   2. アンカー品質が良好か評価")
        print(f"   3. 問題なければStep 2に進行")
        
        return anchors_dict, stats, True
        
    except Exception as e:
        print(f"❌ Step 1でエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False

# ===== 使用例 =====
if __name__ == "__main__":
    # Step 1実行
    anchors, stats, success = run_step1()
    
    if success:
        print("🎉 Step 1成功! Step 2の準備完了")
    else:
        print("❌ Step 1失敗 - エラーを確認してください")
