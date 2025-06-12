import torch

class Config:
    # データパス
    train_img_dir = "/content/FLIR_YOLO_local/images/train"
    train_label_dir = "/content/FLIR_YOLO_local/labels/train"
    save_dir = "/content/drive/MyDrive/IR_obj_detection/modular_yolo/checkpoints"
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ★★★ T4 GPU 超攻撃的バッチサイズ ★★★
    # 現在4.3GB使用 → 10-12GB目標まで大幅増加
    batch_size = 192         # 96 → 160 (さらに1.67倍増加！)
    img_size = 512
    num_classes = 15
    num_epochs = 35          # さらに短縮（大きなバッチサイズで早期収束期待）
    
    # ★★★ 超大バッチサイズ向け学習率 ★★★
    base_lr = 2e-4
    learning_rate = base_lr * (batch_size / 32)  # 1e-3 (5倍！)
    
    print(f"🚀🚀 T4 GPU 超攻撃的設定:")
    print(f"   Batch Size: {batch_size} (32→160, 5倍増加!!)")
    print(f"   Learning Rate: {learning_rate:.6f}")
    print(f"   予想GPU使用率: 75-85% (10-12GB)")
    print(f"   予想学習速度: 従来の5-6倍！")
    
    # データ拡張 (超大バッチサイズなのでさらに強化)
    augment = True
    brightness_range = 0.35  # 0.3 → 0.35
    noise_level = 0.025     # 0.02 → 0.025
    weight_decay = 2e-4     # 1e-4 → 2e-4 (正則化強化)
    momentum = 0.937
    
    # 表示・保存設定 (バッチ数激減に対応)
    print_interval = 10      # 15 → 10 (さらに頻繁)
    save_interval = 2        # 3 → 2 (さらに頻繁保存)
    
    # ★★★ 超大バッチサイズ専用最適化 ★★★
    use_phase3_optimization = True
    
    # オプティマイザ (超大バッチサイズ向け)
    optimizer_type = "AdamW"
    betas = (0.9, 0.999)     # 0.937 → 0.9 (大きなバッチサイズではデフォルト値の方が良い)
    eps = 1e-8
    
    # スケジューラ (超大バッチサイズ向け)
    use_scheduler = True
    scheduler_type = "cosine"
    warmup_epochs = 5        # 4 → 5 (超大バッチサイズ用ウォームアップ)
    min_lr = learning_rate / 500  # さらに低く
    
    # 学習安定化 (超大バッチサイズ用)
    gradient_clip = 2.0      # 1.5 → 2.0 (さらに強く)
    
    # Early Stopping (効率重視)
    patience = 10           # 12 → 10 (さらに早期判断)
    min_improvement = 0.003  # 0.005 → 0.003 (より細かく)
    
    # 保存戦略
    save_best_only = False
    
    # ===== 超大バッチサイズ専用EMA =====
    use_ema = True
    ema_decay = 0.9996      # 0.9998 → 0.9996 (さらに早めの更新)
    
    # 検証設定
    validation_split = 0.15
    validate_every = 1
    
    # 損失重みの最適化 (超大バッチサイズ用)
    lambda_coord = 7.0      # 6.0 → 7.0 (さらに座標精度重視)
    lambda_noobj = 0.3      # 0.4 → 0.3 (背景検出をさらに緩和)
    lambda_obj = 1.5        # 1.2 → 1.5 (物体検出をさらに強化)
    
    # ★★★ T4メモリ使用量最大化 ★★★
    # DataLoader超最適化
    dataloader_num_workers = 6    # 4 → 6 (さらに高速化)
    pin_memory = True
    persistent_workers = True
    prefetch_factor = 4          # データ先読み強化
    
    # メモリ効率化オプション
    mixed_precision = False       # T4では16GBあるので不要
    gradient_accumulation_steps = 1
    
    # GPU最適化
    torch_compile = False
    channels_last = True
    
    # ★★★ メモリ使用量確認用設定 ★★★
    memory_debug = True          # メモリ使用量を定期的に表示
    target_memory_usage = 12.0   # 目標メモリ使用量 (GB)
    
    # ===== 実験的高速化設定 ===== 
    # さらなる高速化オプション
    cudnn_benchmark = True       # 同じサイズの入力で高速化
    empty_cache_every_n_batch = 50  # 定期的なメモリクリーンアップ
    
    # Phase 4 準備
    use_multiscale = False
    use_anchors = False
    debug_mode = True
    profile_training = False
    
    # ★★★ 超攻撃的予想性能 ★★★
    print(f"📈📈 予想される劇的改善:")
    print(f"   GPU使用率: 75-85% (10-12GB)")
    print(f"   1エポック時間: 1-2分 (従来の1/6)")
    print(f"   目標Loss: < 0.3 (超高性能)")
    print(f"   総学習時間: 35-70分 (従来の1/5)")
    print(f"   Early Stopping: 12-20エポック予想")
    
    # 緊急時のフォールバック設定
    print(f"🚨 OOM時のフォールバック:")
    print(f"   1. batch_size = 128 に下げる")
    print(f"   2. batch_size = 96 に下げる")
    print(f"   3. dataloader_num_workers = 2 に下げる")
    
    # 成功時のさらなる挑戦
    print(f"🎯 成功したらさらに挑戦:")
    print(f"   batch_size = 192 まで試す")
    print(f"   img_size = 512 で高解像度学習")