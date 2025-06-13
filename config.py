import torch

class Config:
    # データパス
    train_img_dir = "/content/FLIR_YOLO_local/images/train"
    train_label_dir = "/content/FLIR_YOLO_local/labels/train"
    save_dir = "/content/drive/MyDrive/IR_obj_detection/modular_yolo/checkpoints"
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # バッチサイズ
    batch_size = 96
    img_size = 416
    num_classes = 15
    num_epochs = 35
    
    # アーキテクチャ
    use_multiscale_architecture = True
    
    # ★★★ Step 6 Lite: 時間効率重視 ★★★
    learning_rate = 1.3e-4   # Step 5の中間値
    base_lr = 1.3e-4
    
    print(f"⚡ Step 6 Lite版設定:")
    print(f"   重い処理OFF、軽い拡張のみ")
    print(f"   目標時間: 10-12分/エポック")
    print(f"   目標精度: Val 42.0 → 39-41台")
    
    # ★★★ 軽量データ拡張設定 ★★★
    # 基本データ拡張（従来通り）
    augment = True
    brightness_range = 0.3   # 通常レベル
    noise_level = 0.02       # 通常レベル
    
    # 高度データ拡張（超軽量）
    use_advanced_augmentation = False
    use_mixup = True         # 軽い処理のみ
    use_mosaic = False       # 🚫重い処理OFF
    use_cutmix = False       # 🚫重い処理OFF
    
    mixup_prob = 0.4         # MixUpのみなので確率上げる
    mosaic_prob = 0.0        # 完全OFF
    
    print(f"⚡ 軽量データ拡張:")
    print(f"   MixUp: ON (確率40%)")
    print(f"   Mosaic: OFF (時間節約)")
    print(f"   CutMix: OFF (時間節約)")
    print(f"   予想処理時間削減: 50-60%")
    
    # 学習パラメータ（Step 5ベース）
    weight_decay = 4e-4      # Step 5成功値
    momentum = 0.937
    
    # 表示・保存設定
    print_interval = 10
    save_interval = 2
    
    # Phase 4最適化
    use_phase3_optimization = True
    
    # オプティマイザ
    optimizer_type = "AdamW"
    betas = (0.9, 0.999)
    eps = 1e-8
    
    # スケジューラ
    use_scheduler = True
    scheduler_type = "cosine"
    warmup_epochs = 0
    min_lr = learning_rate / 250
    
    # 学習安定化
    gradient_clip = 2.5      # Step 5レベル
    
    # Early Stopping
    patience = 6             # Step 5レベル
    min_improvement = 0.007
    
    # EMA設定
    use_ema = True
    ema_decay = 0.998        # Step 5レベル
    
    # 検証設定
    validation_split = 0.15
    validate_every = 1
    
    # ★★★ Step 5成功設定完全継続 ★★★
    # 損失関数（Step 5の大成功設定）
    lambda_coord = 8.0       # CIoU効果
    lambda_obj = 1.0
    lambda_noobj = 0.5
    lambda_cls = 1.5         # Focal効果
    
    use_ciou = True          # Step 5成功要因
    use_focal = True         # Step 5成功要因
    use_label_smoothing = False
    
    # ★★★ 時間効率化設定 ★★★
    # DataLoader最適化（速度重視）
    dataloader_num_workers = 6    # 4 → 6 (軽量処理なので増加)
    pin_memory = True
    persistent_workers = True
    prefetch_factor = 3           # 2 → 3 (先読み強化)
    
    # メモリ効率化
    mixed_precision = False
    gradient_accumulation_steps = 1
    
    # GPU最適化（速度重視）
    torch_compile = False
    channels_last = True     # False → True (最適化)
    
    # メモリ管理（軽量化）
    memory_debug = False     # True → False (ログ削減)
    target_memory_usage = 8.0
    empty_cache_every_n_batch = 100  # 50 → 100 (頻度削減)
    cudnn_benchmark = True
    
    # デバッグ設定（速度重視）
    debug_mode = False
    profile_training = False
    

    use_advanced_postprocessing = True
    postprocessing_config = {
    'use_soft_nms': True,
    'use_tta': False,        # 学習時は時間効率重視
    'use_multiscale': False,
    'conf_threshold': 0.3,   # 検証時閾値
    'iou_threshold': 0.5,
    'soft_nms_sigma': 0.5,
    'soft_nms_method': 'gaussian'
    }

    print(f"🔧 Phase 4後処理設定:")
    print(f"   Soft-NMS: {'ON' if postprocessing_config['use_soft_nms'] else 'OFF'}")
    print(f"   TTA: {'OFF' if not postprocessing_config['use_tta'] else 'ON'} (時間効率重視)")
    print(f"   検証時詳細後処理: 5エポックに1回")

    # ★★★ Step 6 Lite予想性能 ★★★
    print(f"📊 Step 6 Lite予想:")
    print(f"   エポック時間: 21分 → 10-12分 (50%削減)")
    print(f"   Val Loss: 42.0 → 39-41台 (軽微改善)")
    print(f"   学習安定性: Step 5レベル維持")
    print(f"   時間効率: 大幅改善")
    
    # 処理時間分析
    print(f"⏱️ 時間短縮の内訳:")
    print(f"   Mosaic停止: -8分 (最大要因)")
    print(f"   軽量MixUp: -1分")
    print(f"   DataLoader最適化: -1分")
    print(f"   その他最適化: -1分")
    print(f"   合計短縮: 約11分 (21分 → 10分)")
    
    # 精度とのトレードオフ
    print(f"🎯 精度とのトレードオフ:")
    print(f"   Mosaic効果損失: -2-3% (データ多様性低下)")
    print(f"   MixUp効果維持: +1-2% (効率的拡張)")
    print(f"   Step 5効果維持: CIoU+Focal効果継続")
    print(f"   実質的影響: 軽微 (時間効率を重視)")
    
    # 成功判定
    print(f"📋 Lite版成功判定:")
    print(f"   ✅ 大成功: Val < 40 かつ 時間 < 12分")
    print(f"   ✅ 成功: Val < 42 かつ 時間 < 15分")
    print(f"   🤔 要検討: 時間 > 15分")
    
    # 次ステップ計画
    print(f"🚀 成功時の次ステップ:")
    print(f"   1. Step 7: 後処理最適化 (Soft-NMS等)")
    print(f"   2. mAP測定システム実装")
    print(f"   3. Phase 4最終調整")
    print(f"   4. 必要に応じてMosaic再導入検討")