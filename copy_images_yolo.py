import os
import shutil
from pathlib import Path

src_base = r"C:\Users\sathv\OneDrive\Desktop\ghost_fishing_project\data"
dst_base = r"C:\Users\sathv\OneDrive\Desktop\ghost_fishing_project\yolo_data\images"

for split in ["train", "val", "test"]:
    src_dir = os.path.join(src_base, split)
    dst_dir = os.path.join(dst_base, split)
    images = list(Path(src_dir).glob("*.jpg")) + list(Path(src_dir).glob("*.jpeg")) + list(Path(src_dir).glob("*.png"))
    for img in images:
        shutil.copy(img, dst_dir)
        print(f"Copied {img.name} -> {split}")

print("✓ All images copied to YOLO folders!")
