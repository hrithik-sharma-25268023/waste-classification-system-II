"""script to fetch number of images"""

import os
import shutil
import random

def fetch_images(dataset_path: str, output_path: str, num_images: int) -> None:
    """
    Copy a fixed number of images per class from dataset to output folder.

    Args:
        dataset_path (str): Root dataset directory (class-wise folders)
        output_path (str): Destination directory
        num_images (int): Number of images per class
    """

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    os.makedirs(output_path, exist_ok=True)
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_path):
            continue

        out_class_path = os.path.join(output_path, class_name)
        os.makedirs(out_class_path, exist_ok=True)

        images = [
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if f.lower().endswith(valid_exts)
        ]

        if len(images) == 0:
            continue

        random.shuffle(images)
        selected_images = images[:num_images]

        for img_path in selected_images:
            filename = os.path.basename(img_path)
            dest_path = os.path.join(out_class_path, filename)

            shutil.copy2(img_path, dest_path)

        print(f"Copied {len(selected_images)} images for class '{class_name}'")
