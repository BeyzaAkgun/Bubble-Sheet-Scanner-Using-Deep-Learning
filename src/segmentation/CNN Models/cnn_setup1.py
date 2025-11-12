# cnn_setup1.py
# CNN (UNet-like) segmentation training
# Place this in a Colab cell and run. Expects splits in BASE_PATH/splits

import os
import json
import csv
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[✔] GPU memory growth enabled")
    except RuntimeError as e:
        print(e)



# -----------------------
# Config (adjust if needed)
# -----------------------

BASE_PATH = '/content/drive/My Drive/Aligned_Sheets/'
JSON_PATH = os.path.join(BASE_PATH, 'dataset_updated.json')
MASKS_PATH = os.path.join(BASE_PATH, 'segmentation_masks')
SPLIT_DIR = os.path.join(BASE_PATH, 'splits')
MODEL_PATH = os.path.join(BASE_PATH, 'bubble_segmentation_model_cnn_halfed.h5')
REPORT_CSV = os.path.join(SPLIT_DIR, 'split_report_setup1.csv')

# Training params
BATCH_SIZE = 2
TARGET_SIZE = (416, 576)
EPOCHS = 50
STEPS_PER_EPOCH = 50
VALIDATION_STEPS = 20

print("BASE_PATH:", BASE_PATH)
print("JSON_PATH:", JSON_PATH)
print("MASKS_PATH:", MASKS_PATH)
print("SPLIT_DIR:", SPLIT_DIR)

# -----------------------
# Metrics
# -----------------------
def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    iou = tf.where(union > 0, intersection / union, tf.ones_like(intersection))
    return tf.reduce_mean(iou)

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)

def precision_metric(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    predicted_positives = tf.reduce_sum(y_pred, axis=[1, 2, 3])
    precision = (true_positives + smooth) / (predicted_positives + smooth)
    return tf.reduce_mean(precision)

def recall_metric(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    possible_positives = tf.reduce_sum(y_true, axis=[1, 2, 3])
    recall = (true_positives + smooth) / (possible_positives + smooth)
    return tf.reduce_mean(recall)

def pixel_accuracy(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    correct = tf.equal(y_true, y_pred)
    return tf.reduce_mean(tf.cast(correct, tf.float32))


# -----------------------
# Data Generator
# -----------------------
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, mask_paths, batch_size=BATCH_SIZE, target_size=TARGET_SIZE, shuffle=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.zeros((len(indexes), self.target_size[0], self.target_size[1], 3), dtype=np.float32)
        y = np.zeros((len(indexes), self.target_size[0], self.target_size[1], 1), dtype=np.float32)

        for i, idx in enumerate(indexes):
            try:
                img = cv2.imread(self.image_paths[idx])
                if img is None:
                    raise ValueError("cv2.imread returned None")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.target_size[1], self.target_size[0]))
                X[i] = img / 255.0

                mask = np.load(self.mask_paths[idx])
                mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
                y[i] = np.expand_dims(mask, axis=-1)
            except Exception as e:
                print(f"Error loading {self.image_paths[idx]} or its mask: {e}")
                X[i] = np.zeros_like(X[i])
                y[i] = np.zeros_like(y[i])
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# -----------------------
# Model (UNet-like)
# -----------------------
def create_unet_model(input_size=(TARGET_SIZE[0], TARGET_SIZE[1], 3)):
    inputs = Input(input_size)

    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    bn4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(bn4)

    up5 = Conv2D(64, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop4))
    merge5 = concatenate([bn3, up5], axis=3)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

    up6 = Conv2D(32, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([bn2, up6], axis=3)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Conv2D(16, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([bn1, up7], axis=3)
    conv7 = Conv2D(16, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv7)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy', iou_metric, dice_coefficient, metrics.Precision(), metrics.Recall(),pixel_accuracy])
    return model

# -----------------------
# Helpers: build lists from splits
# -----------------------
def entry_to_paths(entry):
    img_path = os.path.join(BASE_PATH, entry['path'])
    folder_num = int(entry['path'].split('/')[0])
    mask_filename = os.path.splitext(os.path.basename(entry['path']))[0] + "_mask.npy"
    mask_path = os.path.join(MASKS_PATH, f"folder_{folder_num:03d}", mask_filename)
    return img_path, mask_path

def build_paths_from_entries(entries):
    imgs, masks, missing = [], [], []
    for e in entries:
        img_path, mask_path = entry_to_paths(e)
        img_exists = os.path.exists(img_path)
        mask_exists = os.path.exists(mask_path)
        if img_exists and mask_exists:
            imgs.append(img_path)
            masks.append(mask_path)
        else:
            missing.append({'path': e.get('path'), 'img_exists': img_exists, 'mask_exists': mask_exists})
    return imgs, masks, missing

# -----------------------
# Load splits and prepare data
# -----------------------
train_split_path = os.path.join(SPLIT_DIR, 'train_entries.json')
val_split_path   = os.path.join(SPLIT_DIR, 'val_entries.json')
test_split_path  = os.path.join(SPLIT_DIR, 'test_entries.json')

for p in (train_split_path, val_split_path, test_split_path):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Split file not found: {p}")

with open(train_split_path, 'r', encoding='utf-8') as f:
    train_entries = json.load(f)
with open(val_split_path, 'r', encoding='utf-8') as f:
    val_entries = json.load(f)
with open(test_split_path, 'r', encoding='utf-8') as f:
    test_entries = json.load(f)

train_imgs, train_masks, train_missing = build_paths_from_entries(train_entries)
val_imgs, val_masks, val_missing = build_paths_from_entries(val_entries)
test_imgs, test_masks, test_missing = build_paths_from_entries(test_entries)

print("Train entries:", len(train_entries), "-> sheets with masks:", len(train_imgs), "missing:", len(train_missing))
print("Val entries:", len(val_entries), "-> sheets with masks:", len(val_imgs), "missing:", len(val_missing))
print("Test entries:", len(test_entries), "-> sheets with masks:", len(test_imgs), "missing:", len(test_missing))

# Save missing report
os.makedirs(SPLIT_DIR, exist_ok=True)
with open(REPORT_CSV, 'w', newline='', encoding='utf-8') as csvf:
    writer = csv.writer(csvf)
    writer.writerow(['split','entry_path','img_exists','mask_exists'])
    for e in train_missing:
        writer.writerow(['train', e['path'], e['img_exists'], e['mask_exists']])
    for e in val_missing:
        writer.writerow(['val', e['path'], e['img_exists'], e['mask_exists']])
    for e in test_missing:
        writer.writerow(['test', e['path'], e['img_exists'], e['mask_exists']])
print(f"[✔] Split/missing report saved to: {REPORT_CSV}")

total_missing = len(train_missing) + len(val_missing) + len(test_missing)
if total_missing > 0:
    print(f"Warning: {total_missing} entries missing image or mask files. Check {REPORT_CSV}.")
    # By default we stop to avoid training on incomplete dataset.
    raise RuntimeError("Missing image/mask files detected. Fix or remove entries before training.")

# -----------------------
# Generators
# -----------------------
train_gen = DataGenerator(train_imgs, train_masks, batch_size=BATCH_SIZE, target_size=TARGET_SIZE, shuffle=True)
val_gen = DataGenerator(val_imgs, val_masks, batch_size=BATCH_SIZE, target_size=TARGET_SIZE, shuffle=False)
test_gen = DataGenerator(test_imgs, test_masks, batch_size=BATCH_SIZE, target_size=TARGET_SIZE, shuffle=False)

# -----------------------
# Model, callbacks, training
# -----------------------
model = create_unet_model(input_size=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
model.summary()

checkpoint_path = MODEL_PATH
callbacks = [
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_iou_metric', mode='max'),
    EarlyStopping(patience=10, monitor='val_iou_metric', mode='max', restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=5, monitor='val_iou_metric', mode='max')
]

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    steps_per_epoch=min(STEPS_PER_EPOCH, len(train_gen)),
    validation_data=val_gen,
    validation_steps=min(VALIDATION_STEPS, len(val_gen)),
    callbacks=callbacks
)

# Save final model
model.save(MODEL_PATH)
print(f"[✔] Training finished. Model saved to: {MODEL_PATH}")

# -----------------------
# Evaluate on test set
# -----------------------
print("[*] Evaluating on test set...")
eval_results = model.evaluate(test_gen, steps=min(len(test_gen), VALIDATION_STEPS), return_dict=True)
print("Test evaluation results:")
for k, v in eval_results.items():
    print(f"  {k}: {v:.4f}")

# optionally: save history plots
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.legend(); plt.title('Loss')
    plt.show()
except Exception:
    pass
