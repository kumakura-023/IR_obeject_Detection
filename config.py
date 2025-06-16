import torch

class Config:
    # データパス
    train_img_dir = "/content/FLIR_YOLO_local/images/train"
    train_label_dir = "/content/FLIR_YOLO_local/labels/train"
    save_dir = "/content/drive/MyDrive/IR_obj_detection/modular_yolo/checkpoints"
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ★★★ 重要決断: マルチスケールを無効化 ★★★
    # 予測数1,022,112は明らかに異常 → 標準アーキテクチャに戻る
    use_multiscale_architecture = False  # True → False
    
    print(f"🏗️ アーキテクチャ修正:")
    print(f"   マルチスケール: ON → OFF")
    print(f"   → SimpleYOLO（単一スケール）に切り替え")
    print(f"   → 予測数を大幅削減")
    print(f"   → 偽検出爆発を根本解決")
    
    # ★★★ 標準的な損失重み（SimpleYOLO用） ★★★
    lambda_coord = 5.0          # 標準的な値
    lambda_obj = 1.0            # 標準的な値
    lambda_noobj = 0.5          # 標準的な値（SimpleYOLOには十分）
    lambda_cls = 1.0            # 標準的な値
    
    print(f"⚖️ 損失重み正常化:")
    print(f"   coord=5.0, obj=1.0, noobj=0.5, cls=1.0")
    print(f"   → 標準的なYOLOバランス")
    print(f"   → 過度な調整をリセット")
    
    # ★★★ 学習率も正常化 ★★★
    learning_rate = 3e-4        # 2e-4 → 3e-4 (標準的な値に戻す)
    base_lr = 3e-4
    min_lr = 1e-5
    
    print(f"📈 学習率正常化:")
    print(f"   3e-4 (標準的な値)")
    print(f"   → SimpleYOLOに最適化")
    
    # ウォームアップ（標準）
    warmup_epochs = 3
    
    # バッチサイズ（維持）
    batch_size = 96
    img_size = 416
    num_classes = 15
    num_epochs = 50
    
    # ★★★ データ拡張も正常化 ★★★
    augment = True
    brightness_range = 0.3      # 標準的な値
    noise_level = 0.02          # 標準的な値
    contrast_range = 0.3        # 標準的な値
    gaussian_blur_prob = 0.3    # 標準的な値
    
    # 高度データ拡張（標準）
    use_advanced_augmentation = True
    use_mixup = True
    use_mosaic = False          # SimpleYOLOには重すぎる
    use_cutmix = False
    
    mixup_prob = 0.5           # 標準的な値
    mosaic_prob = 0.0
    
    print(f"🎨 データ拡張正常化:")
    print(f"   すべて標準的な値に戻す")
    print(f"   → SimpleYOLOに最適化")
    
    # ★★★ 高度損失関数も無効化 ★★★
    use_ciou = False            # True → False (SimpleYOLOには複雑すぎる)
    use_focal = False           # True → False (SimpleYOLOには複雑すぎる)
    use_label_smoothing = False
    
    print(f"🔧 損失関数簡素化:")
    print(f"   CIoU: OFF, Focal: OFF")
    print(f"   → 標準MSE + CrossEntropy")
    print(f"   → SimpleYOLOに最適化")
    
    # 学習パラメータ（標準化）
    weight_decay = 1e-4         # 3e-4 → 1e-4 (標準値)
    momentum = 0.937
    
    # オプティマイザ（標準）
    optimizer_type = "Adam"     # AdamW → Adam (SimpleYOLOには十分)
    betas = (0.9, 0.999)
    eps = 1e-8
    
    # スケジューラ（標準）
    use_scheduler = True
    scheduler_type = "cosine"
    
    # 学習安定化（標準）
    gradient_clip = 5.0         # 1.5 → 5.0 (標準値)
    patience = 10
    min_improvement = 0.01      # 0.003 → 0.01 (より緩く)
    
    # EMA設定（標準）
    use_ema = True
    ema_decay = 0.999
    
    # 検証設定（維持）
    validation_split = 0.15
    validate_every = 1
    
    # DataLoader最適化（維持）
    dataloader_num_workers = 0
    pin_memory = False
    persistent_workers = False
    prefetch_factor = 2
    
    # メモリ効率化（維持）
    mixed_precision = False
    gradient_accumulation_steps = 1
    
    # GPU最適化（標準）
    torch_compile = False
    channels_last = False       # True → False (SimpleYOLOには不要)
    memory_debug = False
    target_memory_usage = 8.0
    empty_cache_every_n_batch = 50  # 20 → 50 (標準頻度)
    cudnn_benchmark = True
    
    # デバッグ設定（標準）
    debug_mode = True
    profile_training = True
    
    # 表示・保存設定（標準）
    print_interval = 10
    save_interval = 5
    
    # 診断機能（維持）
    use_diagnostic_training = True
    log_detection_stats = True
    save_detection_samples = True
    
    # 後処理設定（簡素化）
    use_advanced_postprocessing = False  # True → False (SimpleYOLOには不要)
    postprocessing_config = {
        'use_soft_nms': False,       # 標準NMSで十分
        'use_tta': False,
        'use_multiscale': False,
        'conf_threshold': 0.5,       # 0.3 → 0.5 (標準値)
        'iou_threshold': 0.5,
        'soft_nms_sigma': 0.5,
        'soft_nms_method': 'gaussian'
    }
    
    # ★★★ 期待される劇的改善 ★★★
    print(f"📊 期待される劇的改善:")
    print(f"   予測数: 1,022,112 → 約13,000 (98%削減)")
    print(f"   完璧信頼度検出: 520 → 10-50 (90%以上削減)")
    print(f"   学習安定性: 大幅向上")
    print(f"   メモリ使用量: 大幅削減")
    print(f"   学習速度: 3-5倍高速化")
    
    # SimpleYOLO成功判定基準
    print(f"✅ SimpleYOLO成功判定基準:")
    print(f"   1. 予測数 < 50,000")
    print(f"   2. 完璧信頼度検出 < 100")
    print(f"   3. 平均信頼度 0.1-0.3")
    print(f"   4. 最大信頼度 < 0.95")
    print(f"   5. 学習Loss安定下降")
    
    # 長期的期待
    print(f"🎯 SimpleYOLOでの長期期待:")
    print(f"   Epoch 1-2: 安定した学習")
    print(f"   Epoch 3-5: 初回の意味ある検出")
    print(f"   Epoch 10-15: Val Loss < 20")
    print(f"   Epoch 20-30: 実用的な検出性能")
    print(f"   Epoch 30-50: Val Loss < 5")
    
    # なぜSimpleYOLOが良いか
    print(f"💡 SimpleYOLOの利点:")
    print(f"   1. デバッグしやすい（単一スケール）")
    print(f"   2. 安定した学習（複雑さ最小）")
    print(f"   3. 高速（予測数少ない）")
    print(f"   4. 理解しやすい（YOLO原理そのまま）")
    print(f"   5. 実績ある手法（枯れた技術）")
    
    print(f"🚀 まずはSimpleYOLOで基礎を固めよう！")