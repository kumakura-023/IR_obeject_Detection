import torch

class Config:
    # データパス
    train_img_dir = "/content/FLIR_YOLO_local/images/train"
    train_label_dir = "/content/FLIR_YOLO_local/labels/train"
    save_dir = "/content/drive/MyDrive/IR_obj_detection/modular_yolo/checkpoints"
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ★★★ Phase 3統合: 推奨バッチサイズ256 ★★★
    batch_size = 192         # 256の推奨だが、まずは安全に192から
    img_size = 416           # 安定したら512も検討
    num_classes = 15
    num_epochs = 35
    
    # ★★★ Phase 3統合: アーキテクチャ選択 ★★★
    use_multiscale_architecture = True  # True: マルチスケール, False: 従来版
    
    # 学習率（バッチサイズスケーリング）
    base_lr = 2e-4
    learning_rate = base_lr * (batch_size / 32)  # バッチサイズに比例
    
    print(f"🚀 Phase 3統合設定:")
    print(f"   Architecture: {'MultiScale' if use_multiscale_architecture else 'Fallback'}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Learning Rate: {learning_rate:.6f}")
    print(f"   Image Size: {img_size}")
    print(f"   予想GPU使用率: 15-20% (2-3GB)")
    
    # データ拡張 
    augment = True
    brightness_range = 0.35
    noise_level = 0.025
    weight_decay = 2e-4
    momentum = 0.937
    
    # 表示・保存設定
    print_interval = 10
    save_interval = 2
    
    # ★★★ Phase 3統合最適化 ★★★
    use_phase3_optimization = True
    
    # オプティマイザ
    optimizer_type = "AdamW"
    betas = (0.9, 0.999)
    eps = 1e-8
    
    # スケジューラ
    use_scheduler = True
    scheduler_type = "cosine"
    warmup_epochs = 5
    min_lr = learning_rate / 500
    
    # 学習安定化
    gradient_clip = 2.0
    
    # Early Stopping
    patience = 10
    min_improvement = 0.003
    
    # EMA設定
    use_ema = True
    ema_decay = 0.9996
    
    # 検証設定
    validation_split = 0.15
    validate_every = 1
    
    # ★★★ Phase 3統合: マルチスケール専用設定 ★★★
    # 損失重み（マルチスケール用）
    lambda_coord = 5.0      # 座標損失重み
    lambda_obj = 1.0        # 物体信頼度重み  
    lambda_noobj = 0.5      # 背景信頼度重み
    lambda_cls = 1.0        # クラス損失重み
    
    # アンカー設定（Step 1で生成、ここではデフォルト値）
    use_dataset_anchors = True  # True: データセット固有, False: デフォルト
    
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
    channels_last = False
    
    # メモリ使用量確認
    memory_debug = True
    target_memory_usage = 8.0   # 目標メモリ使用量調整 (GB)
    
    # 高速化設定
    cudnn_benchmark = True
    empty_cache_every_n_batch = 100
    
    # デバッグ・プロファイル
    debug_mode = False
    profile_training = False
    
    # ★★★ Phase 3統合予想性能 ★★★
    print(f"📈 Phase 3統合予想改善:")
    print(f"   Val Loss: 20.6 → 5.0以下目標")
    print(f"   GPU使用率: 41% → 15-20%")
    print(f"   1エポック時間: 308秒 → 250秒目標")
    print(f"   検出精度: 大幅向上期待")
    print(f"   過学習: 大幅改善期待")
    
    # フォールバック設定
    print(f"🚨 問題発生時:")
    print(f"   1. use_multiscale_architecture = False")
    print(f"   2. batch_size = 128 に下げる")
    print(f"   3. img_size = 416 維持")
    
    # 成功時の次ステップ
    print(f"🎯 成功したら:")
    print(f"   1. batch_size = 256 に上げる")
    print(f"   2. img_size = 512 に上げる")
    print(f"   3. Phase 4機能追加検討")