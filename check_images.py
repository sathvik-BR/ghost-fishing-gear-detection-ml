import os
from PIL import Image

data_path = "data"
folders = ["train", "val", "test"]

for folder in folders:
    folder_path = os.path.join(data_path, folder)
    print(f"Checking folder: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(file_path)
                print(f"Loaded {filename} - Size: {img.size}")
            except:
                print(f"Failed to load {filename}")
