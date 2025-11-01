from google.colab import drive
drive.mount('/content/drive')

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras import metrics

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

BASE_PATH = '/content/drive/My Drive/Aligned_Sheets/'
JSON_PATH = os.path.join(BASE_PATH, 'dataset_updated.json')
MASKS_PATH = os.path.join(BASE_PATH, 'segmentation_masks')
MODEL_PATH = os.path.join(BASE_PATH, 'bubble_segmentation_model_halfed.h5')

BATCH_SIZE = 2
TARGET_SIZE = (832, 1176)
EPOCHS = 20
STEPS_PER_EPOCH = 50
VALIDATION_STEPS = 20

""":1654 / 2338 ≈ 0.7076
832 / 1152 = 0.722  ➜ slightly off, but acceptable

"""

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
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.target_size[1], self.target_size[0]))
                X[i] = img / 255.0

                mask = np.load(self.mask_paths[idx])
                mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
                y[i] = np.expand_dims(mask, axis=-1)
            except Exception as e:
                print(f"Error loading {self.image_paths[idx]}: {e}")
                X[i] = np.zeros_like(X[i])
                y[i] = np.zeros_like(y[i])
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

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
                  metrics=['accuracy', iou_metric, dice_coefficient, metrics.Precision(),metrics.Recall()])

    return model

def get_bbox_from_mask(mask, threshold=0.5):
    binary_mask = (mask > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append([x, y, x+w, y+h])
    return bboxes

with open(JSON_PATH, 'r') as f:
    data = json.load(f)

image_paths = []
mask_paths = []

for item in data:
    try:
        img_path = os.path.join(BASE_PATH, item['path'])
        folder_num = int(item['path'].split('/')[0])
        mask_filename = os.path.splitext(os.path.basename(item['path']))[0] + "_mask.npy"
        mask_path = os.path.join(MASKS_PATH, f"folder_{folder_num:03d}", mask_filename)

        if os.path.exists(img_path) and os.path.exists(mask_path):
            image_paths.append(img_path)
            mask_paths.append(mask_path)
    except Exception as e:
        print(f"Skipping invalid entry: {e}")

print(f"Found {len(image_paths)} valid pairs")

train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

train_gen = DataGenerator(train_imgs, train_masks)
val_gen = DataGenerator(val_imgs, val_masks)

model = create_unet_model()
model.summary()

callbacks = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_iou_metric', mode='max'),
    EarlyStopping(patience=10, monitor='val_iou_metric', mode='max'),
    ReduceLROnPlateau(factor=0.1, patience=5)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    steps_per_epoch=min(STEPS_PER_EPOCH, len(train_gen)),
    validation_steps=min(VALIDATION_STEPS, len(val_gen)),
    callbacks=callbacks
)

