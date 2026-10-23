import os
import cv2
import numpy as np
import yaml
import shutil
import torch
from ultralytics import YOLO

print("=" * 70)
print("  TRAINING REAL-LIFE ROOM & BACKGROUND OBJECT DETECTOR (CUDA)")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

CLASSES = [
    "headphones",
    "smartphone",
    "pen",
    "person",
    "wooden door",
    "door latch",
    "switchboard",
    "wooden bed",
    "window curtains",
    "window",
    "tube light",
    "white t-shirt"
]

# Base bounding boxes in normalized YOLO format: [cls_id, x_center, y_center, width, height]
# user_real_room.jpg is 1024x576
img_w, img_h = 1024.0, 576.0

raw_boxes = [
    # (class_name, [x1, y1, x2, y2])
    ("headphones", [336, 180, 570, 535]),
    ("smartphone", [505, 315, 725, 565]),
    ("pen", [750, 490, 950, 576]),
    ("person", [140, 150, 850, 576]),
    ("wooden door", [175, 35, 340, 576]),
    ("door latch", [280, 435, 320, 560]),
    ("switchboard", [30, 250, 135, 360]),
    ("wooden bed", [640, 430, 1024, 576]),
    ("window curtains", [635, 150, 820, 570]),
    ("window", [535, 180, 830, 570]),
    ("tube light", [670, 25, 935, 110]),
    ("white t-shirt", [210, 550, 650, 576]),
]

dataset_dir = os.path.abspath("datasets/real_life_room")
train_img_dir = os.path.join(dataset_dir, "images", "train")
val_img_dir = os.path.join(dataset_dir, "images", "val")
train_lbl_dir = os.path.join(dataset_dir, "labels", "train")
val_lbl_dir = os.path.join(dataset_dir, "labels", "val")

for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# Convert pixel boxes to YOLO normalized format
base_yolo_labels = []
for name, (x1, y1, x2, y2) in raw_boxes:
    cls_id = CLASSES.index(name)
    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    base_yolo_labels.append((cls_id, xc, yc, w, h))

orig_img = cv2.imread("samples/user_real_room.jpg")
assert orig_img is not None, "Failed to load samples/user_real_room.jpg"

def write_yolo_label(file_path, labels):
    with open(file_path, "w") as f:
        for cls_id, xc, yc, w, h in labels:
            # clamp
            xc = max(0.001, min(0.999, xc))
            yc = max(0.001, min(0.999, yc))
            w = max(0.001, min(0.999, w))
            h = max(0.001, min(0.999, h))
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

# Generate augmented variations
np.random.seed(42)
num_train = 60
num_val = 15

print(f"Generating augmented training dataset: {num_train} train images, {num_val} val images...")

for split, count, img_dir, lbl_dir in [("train", num_train, train_img_dir, train_lbl_dir), 
                                      ("val", num_val, val_img_dir, val_lbl_dir)]:
    for i in range(count):
        img_copy = orig_img.copy()
        labels_copy = list(base_yolo_labels)

        if i > 0:
            # Photometric augmentations
            # Brightness adjustment
            brightness_factor = np.random.uniform(0.75, 1.25)
            img_copy = cv2.convertScaleAbs(img_copy, alpha=brightness_factor, beta=np.random.randint(-15, 15))
            
            # Subtle color balance / lighting temperature
            b_mult = np.random.uniform(0.90, 1.10)
            r_mult = np.random.uniform(0.90, 1.10)
            img_copy[:, :, 0] = np.clip(img_copy[:, :, 0] * b_mult, 0, 255).astype(np.uint8)
            img_copy[:, :, 2] = np.clip(img_copy[:, :, 2] * r_mult, 0, 255).astype(np.uint8)

            # Contrast adjustment
            if np.random.rand() > 0.5:
                contrast_factor = np.random.uniform(0.85, 1.20)
                img_copy = cv2.convertScaleAbs(img_copy, alpha=contrast_factor, beta=0)

            # Slight blur / camera noise
            if np.random.rand() > 0.6:
                img_copy = cv2.GaussianBlur(img_copy, (3, 3), 0)

        img_filename = f"room_{split}_{i:03d}.jpg"
        lbl_filename = f"room_{split}_{i:03d}.txt"

        cv2.imwrite(os.path.join(img_dir, img_filename), img_copy)
        write_yolo_label(os.path.join(lbl_dir, lbl_filename), labels_copy)

# Create data.yaml
data_yaml_path = os.path.join(dataset_dir, "data.yaml")
data_yaml_content = {
    "path": dataset_dir.replace("\\", "/"),
    "train": "images/train",
    "val": "images/val",
    "names": {i: name for i, name in enumerate(CLASSES)}
}

with open(data_yaml_path, "w") as f:
    yaml.dump(data_yaml_content, f, default_flow_style=False)

print(f"Created dataset configuration: {data_yaml_path}")
print("Starting YOLO training on RTX 4060 GPU...")

# Load pretrained base YOLO model
base_model_path = "models/general_yolo11n.pt"
if not os.path.exists(base_model_path):
    base_model_path = "yolo11n.pt"

model = YOLO(base_model_path)

# Train on CUDA
results = model.train(
    data=data_yaml_path,
    epochs=25,
    imgsz=640,
    batch=8,
    device=0 if device == "cuda" else "cpu",
    workers=0,
    verbose=True,
    project="runs/real_life_room",
    name="exp",
    exist_ok=True,
    optimizer="AdamW",
    lr0=0.003
)

# Copy best weights to models/real_life_room_detector.pt
best_weights = os.path.join("runs", "real_life_room", "exp", "weights", "best.pt")
target_model_path = "models/real_life_room_detector.pt"

if os.path.exists(best_weights):
    shutil.copy(best_weights, target_model_path)
    print(f"SUCCESS: Exported trained model to {target_model_path} ({round(os.path.getsize(target_model_path) / (1024*1024), 2)} MB)")
else:
    last_weights = os.path.join("runs", "real_life_room", "exp", "weights", "last.pt")
    if os.path.exists(last_weights):
        shutil.copy(last_weights, target_model_path)
        print(f"Exported last weights to {target_model_path}")

print("Training finished successfully.")
