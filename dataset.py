import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class YoloInfraredDataset(Dataset):
    def __init__(self, image_dir, label_dir, input_size=(640, 512), num_classes=5, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.transform = transform
        self.num_classes = num_classes

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(('.jpg', '.png'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')

        # Load grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.input_size)
        img_tensor = torch.from_numpy(img).unsqueeze(0).float() / 255.0  # [1, H, W]

        # Load label (YOLO format)
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls, cx, cy, w, h = map(float, line.strip().split())
                    labels.append([cx, cy, w, h, 1.0] + [1 if i == int(cls) else 0 for i in range(self.num_classes)])

        # Convert to tensor [N, 5+C]
        if labels:
            labels = torch.tensor(labels, dtype=torch.float32)
        else:
            labels = torch.zeros((0, 5 + self.num_classes), dtype=torch.float32)

        return img_tensor, labels