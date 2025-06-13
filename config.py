import torch

class Config:
    # データパス
    train_img_dir = "/content/FLIR_YOLO_local/images/train"
    train_label_dir = "/content/FLIR_YOLO_local/labels/train"
    save_dir = "/content/drive/MyDrive/IR_obj_detection/modular_yolo/checkpoints"
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # バッチサイズ（成功実績あり）
    batch_size = 96         # 前回成功した設定
    img_size = 416
    num_classes = 15
    num_epochs = 35
    
    # アーキテクチャ
    use_multiscale_architecture = True
    
    # ★★★ 段階的高LR実験 ★★★
    # Phase 1: 現在の1.875倍
    learning_rate = 1.5e-4   # 8e-5 → 1.5e-4 (1.875倍)
    base_lr = 1.5e-4
    
    print(f"🔥 段階的高LR実験 Phase 1:")
    print(f"   前回成功LR: 8e-5 (0.00008)")
    print(f"   今回実験LR: {learning_rate:.6f} (1.875倍)")
    print(f"   狙い: Val 55.4 → 45-50台")
    
    # 実験計画表示
    print(f"📊 実験計画:")
    print(f"   Phase 1 (今回): LR 1.5e-4 → Val < 50 期待")
    print(f"   Phase 2 (次回): LR 3e-4   → Val < 45 期待")
    print(f"   Phase 3 (最終): LR 6e-4   → Val < 40 期待")
    
    # データ拡張（前回と同じ）
    augment = True
    brightness_range = 0.35
    noise_level = 0.025
    weight_decay = 4e-4
    momentum = 0.937
    
    # 表示・保存設定
    print_interval = 10
    save_interval = 2
    
    # Phase 3最適化
    use_phase3_optimization = True
    
    # オプティマイザ（高LR対応）
    optimizer_type = "AdamW"
    betas = (0.9, 0.999)
    eps = 1e-8
    
    # スケジューラ（高LR実験用に調整）
    use_scheduler = True
    scheduler_type = "cosine"
    warmup_epochs = 0       # ウォームアップなし
    min_lr = learning_rate / 200  # 最小LRを調整
    
    # 学習安定化（高LR対応）
    gradient_clip = 2.5     # 2.0 → 2.5 (少し強化)
    
    # Early Stopping（短期実験用）
    patience = 6            # 5 → 6 (少し余裕)
    min_improvement = 0.008 # 0.01 → 0.008 (少し厳しく)
    
    # EMA設定（高LR対応）
    use_ema = True
    ema_decay = 0.998       # 0.999 → 0.998 (高LR用)
    
    # 検証設定
    validation_split = 0.15
    validate_every = 1
    
    # 損失重み
    lambda_coord = 5.0
    lambda_obj = 1.0
    lambda_noobj = 0.5
    lambda_cls = 1.0
    
    # DataLoader設定
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
    
    # メモリ管理
    memory_debug = True
    target_memory_usage = 8.0
    empty_cache_every_n_batch = 100
    cudnn_benchmark = True
    
    # デバッグ設定
    debug_mode = False
    profile_training = False
    
    # ★★★ 実験判定基準 ★★★
    print(f"📋 Phase 1判定基準:")
    print(f"   🎉 大成功: Val < 45 → Phase 2へ (LR 3e-4)")
    print(f"   ✅ 成功:   Val < 50 → Phase 2へ")
    print(f"   🤔 微妙:   Val 50-55 → 継続観察")
    print(f"   ❌ 失敗:   Val > 55 → LR下げる")
    
    print(f"🎯 Phase 1目標:")
    print(f"   主目標: Val Loss < 50")
    print(f"   理想目標: Val Loss < 45")
    print(f"   最低目標: Val Loss < 55 (現状維持)")
    
    # 次フェーズ準備
    print(f"📅 次フェーズ設定:")
    print(f"   Phase 2: learning_rate = 3e-4 (2倍)")
    print(f"   Phase 3: learning_rate = 6e-4 (4倍)")
    print(f"   限界テスト: learning_rate = 1e-3 (6.7倍)")
    
    # 安全対策
    print(f"🚨 安全対策:")
    print(f"   Loss爆発時: 即座にLR半減")
    print(f"   NaN発生時: 前回設定に復帰")
    print(f"   GPU OOM時: batch_size削減")
    
    # 期待成果
    print(f"📈 期待成果:")
    print(f"   Phase 1成功: 収束速度2倍、精度5-10%向上")
    print(f"   全Phase成功: Val Loss 35-40台達成")
    print(f"   副次効果: 最適LR発見、学習時間短縮")