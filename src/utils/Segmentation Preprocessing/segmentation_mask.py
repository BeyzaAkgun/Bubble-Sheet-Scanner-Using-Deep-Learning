import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from google.colab import drive


drive.mount('/content/drive')

base_path = '/content/drive/My Drive/Aligned_Sheets/'  
json_path = os.path.join(base_path, 'dataset_updated.json')  
output_path = os.path.join(base_path, 'segmentation_masks') 

os.makedirs(output_path, exist_ok=True)

with open(json_path, 'r') as f:
    data = json.load(f)

def create_segmentation_mask(image_shape, answer_area):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    if isinstance(answer_area, list) and len(answer_area) == 4 and all(isinstance(coord, int) for coord in answer_area):
        x1, y1, x2, y2 = answer_area  
        mask[y1:y2, x1:x2] = 1  

    elif isinstance(answer_area, list) and all(isinstance(box, list) for box in answer_area):
        for box in answer_area:
            if len(box) == 4:  
                x1, y1, x2, y2 = box
                mask[y1:y2, x1:x2] = 1  

    else:
        print(f"Warning: Unrecognized answer_area format: {answer_area}")

    return mask

def overlay_mask(image, mask):
    overlay = image.copy()
    overlay[mask == 1] = [0, 255, 0]  
    return overlay

for folder_num in range(1,16):
    folder_prefix = f"{folder_num:03d}/" 

    folder_data = [item for item in data if item['path'].startswith(folder_prefix)]

    if not folder_data:
        print(f"No images found in folder {folder_prefix}")
        continue

    print(f"Processing folder {folder_prefix} - Found {len(folder_data)} images")

    folder_output_path = os.path.join(output_path, f"folder_{folder_num:03d}")
    os.makedirs(folder_output_path, exist_ok=True)

    num_to_visualize = min(5, len(folder_data))

    fig, axes = plt.subplots(num_to_visualize, 3, figsize=(15, num_to_visualize * 5))

    if num_to_visualize == 1:
        axes = [axes]

    for i, item in enumerate(folder_data):
        img_path = os.path.join(base_path, item['path'])
        answer_area = item['answer_area']

        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to read image: {img_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

            mask = create_segmentation_mask(image.shape, answer_area)
            overlay = overlay_mask(image, mask)

            mask_filename = os.path.splitext(os.path.basename(item['path']))[0] + "_mask.npy"
            mask_path = os.path.join(folder_output_path, mask_filename)
            np.save(mask_path, mask)

            if i < num_to_visualize:
                axes[i][0].imshow(image)
                axes[i][0].set_title(f"Original - {os.path.basename(item['path'])}")
                axes[i][0].axis("off")

                axes[i][1].imshow(mask, cmap='gray')
                axes[i][1].set_title(f"Segmentation Mask")
                axes[i][1].axis("off")

                axes[i][2].imshow(overlay)
                axes[i][2].set_title(f"Overlay Mask")
                axes[i][2].axis("off")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    plt.tight_layout()
    plt.suptitle(f"Visualizations for Folder {folder_prefix}", fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.show()

print("Processing complete!")

mask_dataset = {}
for folder_num in range(1, 30):  
    folder_name = f"folder_{folder_num:03d}"
    folder_path = os.path.join(output_path, folder_name)

    if os.path.exists(folder_path):
        mask_files = glob.glob(os.path.join(folder_path, "*_mask.npy"))
        mask_dataset[folder_name] = mask_files

print(f"Total folders processed: {len(mask_dataset)}")
print(f"Total masks saved: {sum(len(files) for files in mask_dataset.values())}")

def load_mask_dataset(folder_name=None):
    masks = {}

    if folder_name and folder_name in mask_dataset:
        print(f"Loading masks from {folder_name}...")
        for mask_path in mask_dataset[folder_name]:
            mask_id = os.path.basename(mask_path)
            masks[mask_id] = np.load(mask_path)
    else:
        for folder, mask_paths in mask_dataset.items():
            masks[folder] = {}
            print(f"Loading masks from {folder}...")
            for mask_path in mask_paths:
                mask_id = os.path.basename(mask_path)
                masks[folder][mask_id] = np.load(mask_path)

    return masks
