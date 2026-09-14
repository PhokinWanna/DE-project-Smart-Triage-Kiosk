# import os
# import numpy as np
# import tensorflow as tf
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt

# try:
#     from keras import layers, models, callbacks
# except ImportError:
#     from tensorflow.keras import layers, models, callbacks

# # ==========================================
# # 1. CONFIGURATION & HYPERPARAMETERS
# # ==========================================
# DATASET_DIR = "cnn_dataset-02"     # โฟลเดอร์ที่เก็บ Normal/ และ Flushing/
# IMG_SIZE = (64, 64)
# BATCH_SIZE = 32
# EPOCHS = 40
# SEED = 42
# AUTOTUNE = tf.data.AUTOTUNE

# print("[*] Loading dataset from:", DATASET_DIR)

# # ==========================================
# # 2. LOAD DATASET (60% Train, 20% Val, 20% Test)
# # ==========================================
# train_ds = tf.keras.utils.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.4,
#     subset="training",
#     seed=SEED,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     label_mode="binary"
# )

# remaining_ds = tf.keras.utils.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.4,
#     subset="validation",
#     seed=SEED,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     label_mode="binary"
# )

# class_names = train_ds.class_names
# print(f"[*] Detected classes: {class_names}")

# # แบ่ง 40% ที่เหลือออกเป็น Val (20%) และ Test (20%)
# val_size = int(0.5 * tf.data.experimental.cardinality(remaining_ds).numpy())
# val_ds = remaining_ds.take(val_size)
# test_ds = remaining_ds.skip(val_size)

# # ==========================================
# # 3. DYNAMIC CLASS WEIGHTS (แก้ 3.7 : 1)
# # ==========================================
# # ดึง Label จาก Train set เพื่อนับสัดส่วนจริง
# labels_list = []
# for _, lbls in train_ds:
#     labels_list.extend(lbls.numpy().flatten())

# count_0 = np.sum(np.array(labels_list) == 0)
# count_1 = np.sum(np.array(labels_list) == 1)
# total = count_0 + count_1

# class_weights = {
#     0: (1.0 / count_0) * (total / 2.0),
#     1: (1.0 / count_1) * (total / 2.0)
# }
# print(f"[*] Train counts -> {class_names[0]}: {count_0}, {class_names}: {count_1}")
# print(f"[*] Calculated Class Weights: {class_weights}")

# # Prefetching เพิ่มความเร็ว Pipeline
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# # ==========================================
# # 4. NATIVE DATA AUGMENTATION (RGB Safe)
# # ==========================================
# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.05),     # เอียงศีรษะเล็กน้อย ±5%
#     layers.RandomBrightness(0.1),    # แสงนีออนสว่าง/มืดต่างกันใน รพ.
#     layers.RandomContrast(0.1)
# ], name="data_augmentation")

# # ==========================================
# # 5. LIGHTWEIGHT CNN ARCHITECTURE
# # ==========================================
# def build_skin_cnn():
#     inputs = layers.Input(shape=(64, 64, 3))

#     # 1. Augment & Standard Normalize
#     x = data_augmentation(inputs)
#     x = layers.Rescaling(1./255)(x)

#     # Conv Block 1: 32 Filters
#     x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling2D((2, 2))(x)  # 32x32
#     x = layers.Dropout(0.2)(x)

#     # Conv Block 2: 64 Filters
#     x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling2D((2, 2))(x)  # 16x16
#     x = layers.Dropout(0.2)(x)

#     # Conv Block 3: 128 Filters
#     x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling2D((2, 2))(x)  # 8x8
#     x = layers.Dropout(0.3)(x)

#     # Global Average Pooling (แทน Flatten เพื่อลด Parameters และกัน Overfitting)
#     x = layers.GlobalAveragePooling2D()(x)

#     # Dense Layers
#     x = layers.Dense(64, activation='relu')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.3)(x)

#     x = layers.Dense(32, activation='relu')(x)
#     x = layers.Dropout(0.2)(x)

#     outputs = layers.Dense(1, activation='sigmoid')(x)

#     model = models.Model(inputs=inputs, outputs=outputs, name="Skin_Flushing_CNN")
#     return model

# model = build_skin_cnn()
# model.summary()

# # ==========================================
# # 6. COMPILE & STABLE CALLBACKS
# # ==========================================
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#     loss='binary_crossentropy',
#     metrics=[
#         'accuracy',
#         tf.keras.metrics.Precision(name='precision'),
#         tf.keras.metrics.Recall(name='recall'),
#         tf.keras.metrics.AUC(name='auc')
#     ]
# )

# my_callbacks = [
#     callbacks.EarlyStopping(
#         monitor='val_auc',      # วัดผลผ่าน AUC (เสถียรที่สุดสำหรับข้อมูล Imbalance)
#         patience=7,
#         restore_best_weights=True,
#         verbose=1,
#         mode='max'
#     ),
#     callbacks.ReduceLROnPlateau(
#         monitor='val_loss',
#         factor=0.5,
#         patience=3,
#         min_lr=1e-6,
#         verbose=1
#     ),
#     callbacks.ModelCheckpoint(
#         filepath="best_skin_cnn.keras",
#         monitor='val_auc',
#         mode='max',
#         save_best_only=True,
#         verbose=1
#     )
# ]

# # ==========================================
# # 7. TRAINING
# # ==========================================
# print("\n" + "="*60)
# print("[*] Starting Clean Model Training...")
# print("="*60)

# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=EPOCHS,
#     class_weight=class_weights,
#     callbacks=my_callbacks,
#     verbose=1
# )

# # ==========================================
# # 8. EVALUATION ON TEST SET (Unseen Data)
# # ==========================================
# print("\n" + "="*60)
# print("[*] Evaluating on Test Set...")
# print("="*60)

# test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(test_ds, verbose=0)
# f1_test = 2 * (test_prec * test_rec) / (test_prec + test_rec + 1e-7)

# print(f"\n[+] Test Set Results:")
# print(f"    Accuracy:  {test_acc:.4f}")
# print(f"    Precision: {test_prec:.4f}")
# print(f"    Recall:    {test_rec:.4f}")
# print(f"    F1 Score:  {f1_test:.4f}")
# print(f"    AUC:       {test_auc:.4f}")

# # ดึงข้อมูลมาทำ Confusion Matrix
# y_true = []
# y_pred_probs = []
# for imgs, lbls in test_ds:
#     y_true.extend(lbls.numpy().flatten().astype(int))
#     y_pred_probs.extend(model.predict(imgs, verbose=0).flatten())

# y_pred = (np.array(y_pred_probs) > 0.5).astype(int)

# print(f"\n[*] Classification Report:")
# print(classification_report(y_true, y_pred, target_names=class_names))

# print(f"[*] Confusion Matrix:")
# print(confusion_matrix(y_true, y_pred))

# # ==========================================
# # 9. PLOT TRAINING HISTORY (สำหรับใส่ใน Report)
# # ==========================================
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
# plt.title('Loss History')
# plt.legend()
# plt.grid(True)

# plt.subplot(1, 2, 2)
# plt.plot(history.history['auc'], label='Train AUC')
# plt.plot(history.history['val_auc'], label='Val AUC')
# plt.title('AUC History')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.savefig('training_history.png', dpi=150)
# print("\n[+] Training history saved to 'training_history.png'")

# # ==========================================
# # 10. EXPORT TO TFLITE (Dynamic Range Quantization)
# # ==========================================
# print("\n" + "="*60)
# print("[*] Exporting to TensorFlow Lite (.tflite)...")
# print("="*60)

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]  # บีบอัดขนาดไฟล์ลง 4 เท่าและเร่งความเร็วบน CPU
# tflite_model = converter.convert()

# with open("skin_classifier.tflite", "wb") as f:
#     f.write(tflite_model)

# tflite_kb = len(tflite_model) / 1024
# print(f"[+] Successfully exported 'skin_classifier.tflite' ({tflite_kb:.2f} KB)")



import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from keras import layers, models, callbacks


# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASET_SPLIT_DIR = "cnn_dataset_split"
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 40
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# โหลดข้อมูลตรงจากโฟลเดอร์ที่แบ่งไว้แล้ว
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_SPLIT_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_SPLIT_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_SPLIT_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = train_ds.class_names
print(f"[*] Detected classes: {class_names}")

# ==========================================
# 2. CLASS WEIGHTS (คำนวณจาก Train Set จริง)
# ==========================================
labels_list = []
for _, lbls in train_ds:
    labels_list.extend(lbls.numpy().flatten())

count_0 = np.sum(np.array(labels_list) == 0)
count_1 = np.sum(np.array(labels_list) == 1)
total = count_0 + count_1

class_weights = {
    0: (1.0 / count_0) * (total / 2.0),
    1: (1.0 / count_1) * (total / 2.0)
}
print(f"[*] Train counts -> {class_names[0]}: {count_0}, {class_names}: {count_1}")
print(f"[*] Applied Class Weights: {class_weights}")

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomBrightness(0.1),
    layers.RandomContrast(0.1)
], name="data_augmentation")

def build_skin_cnn():
    inputs = layers.Input(shape=(64, 64, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)

    # Conv Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Conv Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Conv Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Global Pooling & Dense
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs=inputs, outputs=outputs, name="Skin_Flushing_CNN")

model = build_skin_cnn()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

my_callbacks = [
    callbacks.EarlyStopping(monitor='val_auc', patience=7, restore_best_weights=True, mode='max', verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint(filepath="best_skin_cnn.keras", monitor='val_auc', mode='max', save_best_only=True, verbose=1)
]

# ==========================================
# 4. START TRAINING
# ==========================================
print("\n[*] Training Model on Patient-Separated Data...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=my_callbacks,
    verbose=1
)

# ==========================================
# 5. FINAL TEST EVALUATION (คนไข้ใหม่ 100%)
# ==========================================
print("\n[*] Evaluating on Completely Unseen Patients (Test Set)...")
test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(test_ds, verbose=0)
f1_test = 2 * (test_prec * test_rec) / (test_prec + test_rec + 1e-7)

print(f"\n[+] Clean Test Results:")
print(f"    Accuracy:  {test_acc:.4f}")
print(f"    Precision: {test_prec:.4f}")
print(f"    Recall:    {test_rec:.4f}")
print(f"    F1 Score:  {f1_test:.4f}")
print(f"    AUC:       {test_auc:.4f}")

y_true, y_pred_probs = [], []
for imgs, lbls in test_ds:
    y_true.extend(lbls.numpy().flatten().astype(int))
    y_pred_probs.extend(model.predict(imgs, verbose=0).flatten())

y_pred = (np.array(y_pred_probs) > 0.5).astype(int)
print(f"\n[*] Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))
print(f"[*] Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# ==========================================
# 6. EXPORT TO TFLITE
# ==========================================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("skin_classifier.tflite", "wb") as f:
    f.write(tflite_model)

print(f"[+] Exported 'skin_classifier.tflite' ({len(tflite_model)/1024:.2f} KB)")