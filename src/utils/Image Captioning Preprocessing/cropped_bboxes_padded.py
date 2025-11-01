#!pip install pillow

from google.colab import drive
drive.mount('/content/drive')

from PIL import Image, ImageOps
import os

base_input_dir = "/content/drive/MyDrive/Aligned_Sheets1/cropped_bboxes_512"
base_output_dir = "/content/drive/MyDrive/Aligned_Sheets1/cropped_bboxes_padded"

target_size = (512, 512)

def process_image(image_path, save_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")

        img.thumbnail(target_size, Image.LANCZOS)

        delta_w = target_size[0] - img.width
        delta_h = target_size[1] - img.height
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)

        padded_img = ImageOps.expand(img, padding, fill=(255, 255, 255))
        padded_img.save(save_path)

for i in range(1, 48):
    folder_name = f"{i:03d}"  
    input_folder = os.path.join(base_input_dir, folder_name)
    output_folder = os.path.join(base_output_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    print(f"{folder_name} folder is in process...")

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            try:
                process_image(input_path, output_path)
            except Exception as e:
                print(f"Error: {filename} ({e})")

print("Process completed for all folders.")