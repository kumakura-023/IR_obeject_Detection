# config.py
import torch

class Config:
    # データパス
    train_img_dir = "/content/FLIR_YOLO_local/images/train"
    train_label_dir = "/content/FLIR_YOLO_local/labels/train"
    save_dir = "/content/drive/MyDrive/IR_obj_detection/modular_yolo/checkpoints"
    
    # ハイパーパラメータ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    img_size = 416
    num_classes = 15
    num_epochs = 50
    learning_rate = 1e-3
    
    # 表示設定
    print_interval = 50
    save_interval = 5