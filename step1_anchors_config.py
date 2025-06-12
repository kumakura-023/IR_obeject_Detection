# ===== Step 1: 生成されたアンカー設定 =====
# 生成日時: 2025-06-12 05:14:18

# データセット固有アンカー
use_anchors = True
anchors = {
    'small':  [(7, 11), (14, 28), (22, 65)],   # 52x52 grid (小物体用)
    'medium': [(42, 35), (76, 67), (46, 126)],  # 26x26 grid (中物体用)
    'large':  [(127, 117), (88, 235), (231, 218)]    # 13x13 grid (大物体用)
}

# アンカー品質統計
anchor_quality = {
    'avg_iou': 0.561,
    'iou_50_percent': 0.666,
    'iou_70_percent': 0.194,
    'total_boxes_analyzed': 16455,
    'generation_method': 'K-means++',
    'dataset_specific': True
}

print(f"📊 Step 1アンカー設定読み込み完了")
print(f"   平均IoU: {anchor_quality['avg_iou']}")
print(f"   IoU>0.5: {anchor_quality['iou_50_percent']:.1%}")
