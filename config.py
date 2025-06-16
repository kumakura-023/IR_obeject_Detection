# config_emergency_fix.py - 偽検出問題緊急修正版
import torch

class Config:
    # データパス
    train_img_dir = "/content/FLIR_YOLO_local/images/train"
    train_label_dir = "/content/FLIR_YOLO_local/labels/train"
    save_dir = "/content/drive/MyDrive/IR_obj_detection/modular_yolo/checkpoints"
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ★★★ 緊急修正: マルチスケール完全無効化 ★★★
    use_multiscale_architecture = False
    
    # ★★★ 緊急修正: 偽検出対策の損失重み ★★★
    lambda_coord = 5.0      # 座標損失
    lambda_obj = 1.0        # 物体信頼度
    lambda_noobj = 3.0      # 背景信頼度（強化: 0.5 → 3.0）
    lambda_cls = 1.0        # クラス損失
    
    # ★★★ 緊急修正: 保守的学習率 ★★★
    learning_rate = 8e-5    # 非常に保守的（3e-4 → 8e-5）
    base_lr = 8e-5
    min_lr = 1e-6
    
    # ウォームアップ（削除）
    warmup_epochs = 0       # 3 → 0（複雑さ除去）
    
    # バッチサイズ（縮小）
    batch_size = 32         # 96 → 32（安定性重視）
    img_size = 416
    num_classes = 15
    num_epochs = 20         # 50 → 20（短期テスト）
    
    # ★★★ 緊急修正: データ拡張最小化 ★★★
    augment = True
    brightness_range = 0.1  # 0.3 → 0.1（最小限）
    noise_level = 0.01      # 0.02 → 0.01（最小限）
    contrast_range = 0.1    # 0.3 → 0.1（最小限）
    gaussian_blur_prob = 0.0 # 0.3 → 0.0（完全OFF）
    
    # 高度データ拡張（完全OFF）
    use_advanced_augmentation = False
    use_mixup = False
    use_mosaic = False
    use_cutmix = False
    
    # ★★★ 緊急修正: 高度機能すべてOFF ★★★
    use_ciou = False
    use_focal = False
    use_label_smoothing = False
    
    # オプティマイザ（シンプル）
    optimizer_type = "Adam"
    weight_decay = 1e-4
    momentum = 0.9
    
    # スケジューラ（OFF）
    use_scheduler = False    # True → False（複雑さ除去）
    
    # EMA（OFF）
    use_ema = False         # True → False（複雑さ除去）
    
    # 学習安定化
    gradient_clip = 1.0     # 5.0 → 1.0（強めのクリッピング）
    patience = 5            # 10 → 5（短期評価）
    min_improvement = 0.02  # 0.01 → 0.02（より厳しく）
    
    # 検証設定
    validation_split = 0.15
    validate_every = 1
    
    # DataLoader（最小限）
    dataloader_num_workers = 0
    pin_memory = False
    persistent_workers = False
    
    # GPU最適化（最小限）
    mixed_precision = False
    torch_compile = False
    channels_last = False
    cudnn_benchmark = True
    
    # デバッグ設定（強化）
    debug_mode = True
    profile_training = True
    
    # 表示設定（頻繁に）
    print_interval = 5      # 10 → 5（頻繁に確認）
    save_interval = 2
    
    # 診断機能（維持）
    use_diagnostic_training = True
    log_detection_stats = True
    save_detection_samples = True
    
    # 後処理（シンプル）
    use_advanced_postprocessing = False
    postprocessing_config = {
        'use_soft_nms': False,
        'use_tta': False,
        'use_multiscale': False,
        'conf_threshold': 0.7,   # 0.5 → 0.7（より厳しく）
        'iou_threshold': 0.5,
    }
    
    print(f"🚨 緊急修正設定ロード完了")
    print(f"   目標: 偽検出完全撲滅")
    print(f"   戦略: 最小限構成でデバッグ")
    print(f"   期待: 予測数 < 1,000、完璧信頼度 < 10")
    
    # ★★★ 緊急診断基準 ★★★
    print(f"🎯 緊急診断基準:")
    print(f"   1. 予測数 < 1,000 (現在16,224)")
    print(f"   2. 完璧信頼度 < 10 (現在372)")
    print(f"   3. 平均信頼度 < 0.3 (現在0.603)")
    print(f"   4. 高信頼度検出 < 500 (現在10,170)")
    print(f"   5. 学習Loss < 50.0")
    
    print(f"⚠️ この設定でも偽検出が続く場合:")
    print(f"   → モデルアーキテクチャに根本的問題")
    print(f"   → model.pyの完全書き直しが必要")