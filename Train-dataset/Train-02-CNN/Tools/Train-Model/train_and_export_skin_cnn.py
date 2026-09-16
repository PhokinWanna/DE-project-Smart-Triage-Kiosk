"""
Smart Triage Kiosk - Skin Anomaly CNN Training & INT8 Quantization
Author: Phokin Wanna (6630613024)
Refactored: CIELAB conversion, Focal Loss, Patient-wise split, INT8 PTQ.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Config & Hyperparameters
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE = 1e-4

def rgb_to_cielab_normalized(image):
    """Converts RGB tensor to normalized CIELAB float array."""
    # Scale from to uint8 [0, 255] for OpenCV conversion
    img_uint8 = (image * 255.0).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    # L: 0-255 -> 0-1, a: 0-255 -> 0-1, b: 0-255 -> 0-1
    return lab.astype(np.float32) / 255.0

# 2. Binary Focal Loss
class BinaryFocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name="binary_focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        p_t = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1.0), self.alpha, 1.0 - self.alpha)
        
        loss = -alpha_t * tf.pow(1.0 - p_t, self.gamma) * tf.math.log(p_t)
        return tf.reduce_mean(loss)

# 3. Lightweight CNN Backbone
def build_skin_classifier(input_shape=(64, 64, 3)):
    inputs = keras.Input(shape=input_shape)
    
    # Photometric augmentation layers
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomBrightness(factor=0.2)(x)
    x = layers.RandomContrast(factor=0.2)(x)

    # Feature Extractor: Depthwise Separable Convolutions
    x = layers.Conv2D(32, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.SeparableConv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.SeparableConv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="Skin_Anomaly_CNN")
    return model

# 4. INT8 TFLite Converter with Representative Dataset
def export_int8_tflite(keras_model, val_images, export_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        # Calibrate activation dynamic range on 150 validation samples
        for i in range(min(150, len(val_images))):
            sample = np.expand_dims(val_images[i], axis=0).astype(np.float32)
            yield [sample]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32   # Keep float32 input wrapper for easy CV2 feeding
    converter.inference_output_type = tf.float32

    tflite_quant_model = converter.convert()
    with open(export_path, "wb") as f:
        f.write(tflite_quant_model)
    print(f"[SUCCESS] Exported INT8 Quantized Model to {export_path}")