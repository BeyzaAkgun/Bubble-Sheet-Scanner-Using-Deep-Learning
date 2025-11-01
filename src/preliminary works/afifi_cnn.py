import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from google.colab import drive

drive.mount('/content/drive')
SAVE_DIR = '/content/drive/MyDrive/BubbleSheetScannerProject/models/'

image_dir = '/content/drive/MyDrive/BubbleSheetScannerProject/Allimages/'
mask_dir  = '/content/drive/MyDrive/BubbleSheetScannerProject/masks/'

images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
masks  = sorted([os.path.join(mask_dir,  f) for f in os.listdir(mask_dir)])
train_imgs, valtest_imgs, train_msks, valtest_msks = train_test_split(images, masks, test_size=0.30, random_state=42)
val_imgs, test_imgs, val_msks, test_msks       = train_test_split(valtest_imgs, valtest_msks, test_size=0.50, random_state=42)

class ImageMaskGen(Sequence):
    def __init__(self, imgs, msks, batch_size=4, sz=(256,256), shuffle=True):
        self.imgs, self.msks = imgs, msks
        self.batch_size, self.sz, self.shuffle = batch_size, sz, shuffle
        self.on_epoch_end()
    def __len__(self): return len(self.imgs)//self.batch_size
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.imgs))
        if self.shuffle: np.random.shuffle(self.indexes)
    def __getitem__(self, idx):
        idxs = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        X, Y = [], []
        for i in idxs:
            img  = cv2.resize(cv2.imread(self.imgs[i]),  self.sz)/255.0
            msk  = cv2.resize(cv2.imread(self.msks[i],0), self.sz)/255.0
            X.append(img); Y.append(msk[...,None])
        return np.stack(X), np.stack(Y)

train_gen = ImageMaskGen(train_imgs, train_msks, shuffle=True)
val_gen   = ImageMaskGen(val_imgs,   val_msks, shuffle=False)
test_gen  = ImageMaskGen(test_imgs,  test_msks, shuffle=False)

def create_unet(inp_shape=(256,256,3)):
    i = Input(inp_shape)
    def conv_block(x, f):
        x = Conv2D(f,3,padding="same",activation="relu")(x)
        x = BatchNormalization()(x)
        return x
    c1 = conv_block(i,32); p1 = MaxPooling2D()(c1)
    c2 = conv_block(p1,64); p2 = MaxPooling2D()(c2)
    c3 = conv_block(p2,128)
    u1 = UpSampling2D()(c3); u1 = concatenate([u1,c2])
    c4 = conv_block(u1,64)
    u2 = UpSampling2D()(c4); u2 = concatenate([u2,c1])
    c5 = conv_block(u2,32)
    out= Conv2D(1,1,activation="sigmoid")(c5)
    return Model(i,out)

def iou_metric(y_true, y_pred, thresh=0.5):
    y_pred = tf.cast(y_pred>thresh, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter
    return inter/(union+1e-6)
def dice_metric(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred>0.5, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred)
    return (2*inter)/(tf.reduce_sum(y_true)+tf.reduce_sum(y_pred)+smooth)

model = create_unet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(ExponentialDecay(1e-3,10000,0.9)),
    loss=dice_metric,
    metrics=['accuracy', iou_metric, dice_metric]
)
callbacks = [
    EarlyStopping('val_iou_metric', mode='max', patience=5, restore_best_weights=True),
    ModelCheckpoint(os.path.join(SAVE_DIR,'unet_best.h5'),
                    monitor='val_iou_metric', mode='max', save_best_only=True)
]

hist = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks
)

y_trues, y_preds = [], []
for Xb, Yb in test_gen:
    P = model.predict(Xb)
    y_trues += list(Yb.ravel()>0.5)
    y_preds += list((P.ravel()>0.5).astype(int))
acc   = np.mean(np.array(y_trues)==np.array(y_preds))
prec  = precision_score(y_trues, y_preds)
rec   = recall_score(y_trues,  y_preds)
f1    = f1_score(y_trues,     y_preds)
inter = np.logical_and(y_trues, y_preds).sum()
union = np.logical_or(y_trues,  y_preds).sum()
dice  = 2*inter/( (np.array(y_trues).sum() + np.array(y_preds).sum()) )
iou   = inter/union
print(f"U‑Net Test ▶ Acc:{acc:.4f} Prec:{prec:.4f} Rec:{rec:.4f} F1:{f1:.4f} Dice:{dice:.4f} IoU:{iou:.4f}")