import torch

class Config:
    # データパス
    train_img_dir = "/content/FLIR_YOLO_local/images/train"
    train_label_dir = "/content/FLIR_YOLO_local/labels/train"
    save_dir = "/content/drive/MyDrive/IR_obj_detection/modular_yolo/checkpoints"
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ★★★ 学習戦略の根本見直し ★★★
    # 1. 学習率の大胆な調整（8倍増）
    learning_rate = 5e-4    # 6e-5 → 5e-4 (8倍増)
    base_lr = 5e-4
    min_lr = 1e-5           # 最小値も引き上げ
    
    # 2. ウォームアップ追加（重要）
    warmup_epochs = 3       # 0 → 3（学習初期の安定化）
    
    print(f"🔥 学習戦略改善版設定:")
    print(f"   学習率: {learning_rate:.0e} (8倍増)")
    print(f"   ウォームアップ: {warmup_epochs}エポック")
    print(f"   目標: Val Loss 43.45 → 25-30")
    
    # バッチサイズ（メモリ効率重視）
    batch_size = 96         # 安定したサイズ
    img_size = 416
    num_classes = 15
    num_epochs = 50         # 35 → 50（より長期で様子見）
    
    # アーキテクチャ
    use_multiscale_architecture = True
    
    # ★★★ データ拡張の最適化 ★★★
    # 重い処理削除、効果的な拡張のみ
    augment = True
    
    # 基本拡張の強化（サーマル画像特性に最適化）
    brightness_range = 0.5      # 0.3 → 0.5（サーマル画像の特性）
    noise_level = 0.03          # 0.02 → 0.03（少し強化）
    
    # 新規追加: サーマル画像に有効な拡張
    contrast_range = 0.4        # 新規: コントラスト調整
    gaussian_blur_prob = 0.3    # 新規: ガウシアンブラー
    
    # 高度データ拡張の見直し
    use_advanced_augmentation = True
    use_mixup = True           # 軽くて効果的 → 継続
    use_mosaic = False         # 重すぎる → OFF
    use_cutmix = False         # 不要 → OFF
    
    mixup_prob = 0.6           # 0.4 → 0.6（効果的なので増加）
    mosaic_prob = 0.0          # 完全OFF
    
    print(f"🎨 データ拡張最適化:")
    print(f"   MixUp強化: 確率60% (効果的)")
    print(f"   Mosaic停止: 処理時間削減")
    print(f"   新規拡張: コントラスト・ブラー")
    print(f"   予想時間削減: 40-50%")
    
    # ★★★ 損失関数の重み調整 ★★★
    lambda_coord = 10.0         # 8.0 → 10.0（座標により注力）
    lambda_obj = 2.0            # 1.0 → 2.0（物体検出を強化）
    lambda_noobj = 0.3          # 0.5 → 0.3（背景の重要度下げる）
    lambda_cls = 1.0            # 1.5 → 1.0（クラス分類は標準）
    
    print(f"⚖️ 損失関数重み調整:")
    print(f"   座標: 8.0 → 10.0 (+25%)")
    print(f"   物体: 1.0 → 2.0 (2倍)")
    print(f"   背景: 0.5 → 0.3 (-40%)")
    print(f"   → 物体検出により注力")
    
    # 高度損失関数設定
    use_ciou = True             # 継続
    use_focal = True            # 継続  
    use_label_smoothing = False # OFFに（過学習対策）
    
    # 学習パラメータ調整
    weight_decay = 2e-4         # 4e-4 → 2e-4（過学習対策）
    momentum = 0.937
    
    # オプティマイザ
    optimizer_type = "AdamW"
    betas = (0.9, 0.999)
    eps = 1e-8
    
    # スケジューラ（より緩やかに）
    use_scheduler = True
    scheduler_type = "cosine"
    
    # 学習安定化（より忍耐強く）
    gradient_clip = 3.0         # 2.5 → 3.0
    patience = 10               # 6 → 10（より忍耐強く）
    min_improvement = 0.005     # 0.007 → 0.005（より小さな改善も評価）
    
    # EMA設定（安定化強化）
    use_ema = True
    ema_decay = 0.9995          # 0.998 → 0.9995（より安定）
    
    # 検証設定
    validation_split = 0.15
    validate_every = 1
    
    # DataLoader最適化
    dataloader_num_workers = 4
    pin_memory = True
    persistent_workers = True
    prefetch_factor = 2
    
    # メモリ効率化
    mixed_precision = False
    gradient_accumulation_steps = 1
    
    # GPU最適化
    torch_compile = False
    channels_last = True
    memory_debug = False
    target_memory_usage = 8.0
    empty_cache_every_n_batch = 50  # 100 → 50（より頻繁に）
    cudnn_benchmark = True
    
    # デバッグ設定（診断強化）
    debug_mode = True           # False → True（問題特定）
    profile_training = True     # False → True（性能分析）
    
    # 表示・保存設定（診断重視）
    print_interval = 5          # 10 → 5（より詳細に）
    save_interval = 2
    
    # ★★★ 診断機能強化 ★★★
    use_diagnostic_training = True  # 新規追加
    log_detection_stats = True      # 新規追加
    save_detection_samples = True   # 新規追加
    
    # 後処理設定
    use_advanced_postprocessing = True
    postprocessing_config = {
        'use_soft_nms': True,
        'use_tta': False,        # 学習時は時間効率重視
        'use_multiscale': False,
        'conf_threshold': 0.25,  # 0.3 → 0.25（より低い閾値）
        'iou_threshold': 0.5,
        'soft_nms_sigma': 0.5,
        'soft_nms_method': 'gaussian'
    }
    

    dataloader_num_workers = 0
    pin_memory = False


    print(f"🔧 診断機能強化:")
    print(f"   詳細ログ: ON")
    print(f"   検出統計: ON") 
    print(f"   サンプル保存: ON")
    print(f"   信頼度閾値: 0.3 → 0.25")
    
    # ★★★ 期待される改善効果 ★★★
    print(f"📊 期待される改善効果:")
    print(f"   Val Loss: 43.45 → 25-30 (30-40%改善)")
    print(f"   学習安定性: 大幅向上")
    print(f"   検出信頼度: 初回の高信頼度検出期待")
    print(f"   エポック時間: 21分 → 12-15分")
    
    # 成功判定基準
    print(f"✅ 成功判定基準:")
    print(f"   1. Val Loss < 30.0 (今回目標)")
    print(f"   2. conf > 0.5 の検出が出現")
    print(f"   3. Train/Val差 < 2倍")
    print(f"   4. 学習曲線の安定化")
    
    # 次フェーズ判断
    print(f"🚀 次フェーズ判断:")
    print(f"   続行: Val Loss < 30.0")
    print(f"   方針転換: Val Loss > 35.0のまま")
    print(f"   緊急見直し: 5エポックで改善なし")