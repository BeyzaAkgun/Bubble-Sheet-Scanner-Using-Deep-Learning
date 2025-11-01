#!pip -q install transformers timm torchvision tqdm pillow

from google.colab import drive
drive.mount('/content/drive')

import os, glob, cv2, torch, numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from transformers import SegformerForSemanticSegmentation

BASE_PATH   = "/content/drive/MyDrive/Aligned_Sheets"
MODEL_PATH  = os.path.join(BASE_PATH, "segformer_best_model_512.pth")
OUTPUT_DIR  = os.path.join(BASE_PATH, "cropped_bboxes512_2")
TARGET_SIZE = (512, 512)
THRESH      = 0.5
ROW_THRESH  = 50  

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model(model_path: str, device, num_labels: int = 1):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

_tf = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@torch.no_grad()
def predict_mask(model, img_path, device, target_size=(512, 512)):
    img = Image.open(img_path).convert("RGB").resize(target_size)
    original = np.array(img)
    inp = _tf(img).unsqueeze(0).to(device)
    logits = model(pixel_values=inp).logits
    logits = torch.nn.functional.interpolate(
        logits, size=target_size, mode="bilinear", align_corners=False)
    prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()[0]
    return original, prob

def get_bbox_from_mask(mask, thr=127):
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    binary = (mask > thr).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for c in contours:
        if cv2.contourArea(c) > 100:
            x, y, w, h = cv2.boundingRect(c)
            bboxes.append([x, y, x+w, y+h])
    return bboxes

def sort_bboxes_grid(bboxes, row_threshold=50):
 
    if not bboxes:
        return []

    sorted_bboxes = sorted(bboxes, key=lambda b: (b[1] + b[3]) // 2)

    rows = []
    current_row = [sorted_bboxes[0]]
    current_y_center = (sorted_bboxes[0][1] + sorted_bboxes[0][3]) // 2

    for box in sorted_bboxes[1:]:
        y_center = (box[1] + box[3]) // 2

        if abs(y_center - current_y_center) <= row_threshold:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
            current_y_center = y_center

    rows.append(current_row)

    result = []
    for row in rows:
        sorted_row = sorted(row, key=lambda b: (b[0] + b[2]) // 2)
        result.extend(sorted_row)

    return result

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = load_model(MODEL_PATH, device)
print("Model loaded – device:", device)

folder_list = sorted([f for f in os.listdir(BASE_PATH) if f.isdigit()])

for folder in folder_list:
    img_dir = os.path.join(BASE_PATH, folder)
    out_dir = os.path.join(OUTPUT_DIR, folder)
    os.makedirs(out_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.[jp][pn]g")))

    for img_path in tqdm(img_paths, desc=f"Folder {folder}", leave=False):
        original, pred = predict_mask(model, img_path, device, TARGET_SIZE)

        bboxes = get_bbox_from_mask(pred, 127)
        if not bboxes:
            continue

        ordered = sort_bboxes_grid(bboxes, ROW_THRESH)

        base = os.path.splitext(os.path.basename(img_path))[0]
        for idx, (x1,y1,x2,y2) in enumerate(ordered):
            crop = original[y1:y2, x1:x2]
            cv2.imwrite(
                os.path.join(out_dir, f"{base}_crop{idx}.png"),
                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            )

print("Cropped images loaded in ", OUTPUT_DIR, "directory.")