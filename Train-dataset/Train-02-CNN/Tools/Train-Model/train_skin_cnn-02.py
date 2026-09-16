"""
Smart Triage Kiosk - Skin Anomaly CNN Trainer & Edge Quantizer
Author: Phokin Wanna (6630613024)
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE = 1e-4

DATA_DIR = "Train-dataset/Train-02-CNN/split_dataset"
MODEL_SAVE_PATH = "Train-dataset/Train-02-CNN/best_skin_cnn.keras"
TFLITE_EXPORT_PATH = "Train-dataset/Train-02-CNN/skin_classifier.tflite"

# 1. Custom Dataset Loader with CIELAB Color Conversion
def load_split(split_name):
    images = []
    labels = []
    split_path = os.path.join(DATA_DIR, split_name)

    # Binary Classification: Normal (0) vs. Abnormal [Pallor/Flushing] (1)
    class_mapping = {"Normal": 0,"Flushing": 1}

    for cls_name, label in class_mapping.items():
        cls_folder = os.path.join(split_path, cls_name)
        if not os.path.exists(cls_folder):
            continue
        for fname in os.listdir(cls_folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                fpath = os.path.join(cls_folder, fname)
                bgr = cv2.imread(fpath)
                if bgr is None:
                    continue
                resized = cv2.resize(bgr, IMG_SIZE)
                # Convert BGR to CIELAB (isolates luminance from blood volume chromaticity)
                lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
                norm_lab = lab.astype(np.float32) / 255.0

                images.append(norm_lab)
                labels.append(label)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)

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

# 3. Model Architecture
def create_model():
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Data Augmentations
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)

    # Feature Extractor
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

    model = keras.Model(inputs, outputs, name="CIELAB_Skin_CNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=BinaryFocalLoss(),
        metrics=["accuracy", keras.metrics.AUC(name="auc"), keras.metrics.Recall(name="recall")]
    )
    return model

# 4. INT8 TFLite Converter
def export_quantized_tflite(model, val_images, export_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_data_gen():
        for i in range(min(150, len(val_images))):
            yield [np.expand_dims(val_images[i], axis=0)]

    converter.representative_dataset = rep_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    with open(export_path, "wb") as f:
        f.write(tflite_model)
    print(f"[SUCCESS] Exported INT8 Quantized Model to {export_path}")

# 5. Main Execution
if __name__ == "__main__":
    print("Loading datasets...")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    print(f"Train samples: {len(X_train)} | Validation samples: {len(X_val)}")

    model = create_model()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", patience=8, mode="max", restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_auc", mode="max", save_best_only=True)
    ]

    print("Training CIELAB Skin CNN...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    print("Converting to INT8 Edge TFLite...")
    export_quantized_tflite(model, X_val, TFLITE_EXPORT_PATH)

    # ==========================================
    # 6. EVALUATION METRICS & REPORTING
    # ==========================================
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
    
    print("\n" + "="*60)
    print("[*] Generating Evaluation Metrics & Visualizations...")
    print("="*60)

    # Predictions on validation set
    y_val_pred_proba = model.predict(X_val, verbose=0).flatten()
    y_val_pred = (y_val_pred_proba > 0.5).astype(int)

    # ==========================================
    # 6.1 Classification Report
    # ==========================================
    print("\n[*] Classification Report (Validation Set):")
    print("="*60)
    class_names = ["Normal", "Abnormal (Pallor/Flushing)"]
    print(classification_report(y_val, y_val_pred, target_names=class_names, digits=4))

    # ==========================================
    # 6.2 Confusion Matrix
    # ==========================================
    cm = confusion_matrix(y_val, y_val_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n[*] Confusion Matrix (Validation Set):")
    print("="*60)
    print(f"True Negatives (TN):  {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP):  {tp}")
    print(f"\nSensitivity (Recall): {tp / (tp + fn):.4f}")
    print(f"Specificity:          {tn / (tn + fp):.4f}")

    # ==========================================
    # 6.3 Create Visualizations
    # ==========================================
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Training vs Validation Loss
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss Over Epochs', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (Binary Focal Loss)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training vs Validation Accuracy
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy Over Epochs', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: AUC-ROC Curve
    ax3 = plt.subplot(2, 3, 3)
    fpr, tpr, _ = roc_curve(y_val, y_val_pred_proba)
    roc_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC-AUC Curve', fontsize=12, fontweight='bold')
    ax3.legend(loc="lower right")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Confusion Matrix Heatmap
    ax4 = plt.subplot(2, 3, 4)
    im = ax4.imshow(cm, interpolation='nearest', cmap='Blues')
    ax4.figure.colorbar(im, ax=ax4)
    ax4.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]))
    ax4.set_xticklabels(['Normal', 'Abnormal'])
    ax4.set_yticklabels(['Normal', 'Abnormal'])
    ax4.set_ylabel('True Label')
    ax4.set_xlabel('Predicted Label')
    ax4.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax4.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')

    # Plot 5: Precision-Recall Curve
    ax5 = plt.subplot(2, 3, 5)
    precision, recall, _ = precision_recall_curve(y_val, y_val_pred_proba)
    pr_auc = auc(recall, precision)
    ax5.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    ax5.set_xlim([0.0, 1.0])
    ax5.set_ylim([0.0, 1.05])
    ax5.set_xlabel('Recall')
    ax5.set_ylabel('Precision')
    ax5.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    ax5.legend(loc="lower left")
    ax5.grid(True, alpha=0.3)

    # Plot 6: Recall vs Epoch (training stability)
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(history.history['recall'], label='Train Recall', linewidth=2)
    ax6.plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
    ax6.set_title('Model Recall (Sensitivity) Over Epochs', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Recall')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    report_path = "Train-dataset/Train-02-CNN/training_report.png"
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    print(f"\n[+] Training report saved to: {report_path}")
    plt.close()

    # ==========================================
    # 6.4 Model Architecture Summary
    # ==========================================
    arch_report_path = "Train-dataset/Train-02-CNN/model_architecture.txt"
    with open(arch_report_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"[+] Model architecture saved to: {arch_report_path}")

    # ==========================================
    # 6.5 Detailed Metrics Summary
    # ==========================================
    metrics_summary = {
        'Total Validation Samples': len(y_val),
        'True Negatives': int(tn),
        'False Positives': int(fp),
        'False Negatives': int(fn),
        'True Positives': int(tp),
        'Sensitivity (Recall)': f"{tp / (tp + fn):.4f}",
        'Specificity': f"{tn / (tn + fp):.4f}",
        'Precision': f"{tp / (tp + fp):.4f}" if (tp + fp) > 0 else "N/A",
        'F1-Score': f"{2 * tp / (2 * tp + fp + fn):.4f}" if (2 * tp + fp + fn) > 0 else "N/A",
        'Accuracy': f"{(tp + tn) / (tp + tn + fp + fn):.4f}",
        'ROC-AUC': f"{roc_auc:.4f}",
        'PR-AUC': f"{pr_auc:.4f}",
        'Model Size (Keras)': f"{os.path.getsize(MODEL_SAVE_PATH) / (1024*1024):.2f} MB",
        'TFLite Size': f"{os.path.getsize(TFLITE_EXPORT_PATH) / (1024*1024):.2f} MB"
    }

    summary_report_path = "Train-dataset/Train-02-CNN/metrics_summary.txt"
    with open(summary_report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SKIN ANOMALY CNN - VALIDATION METRICS SUMMARY\n")
        f.write("="*60 + "\n\n")
        for key, value in metrics_summary.items():
            f.write(f"{key:.<40} {value}\n")
        f.write("\n" + "="*60 + "\n")

    print(f"\n[+] Metrics summary saved to: {summary_report_path}")

    # ==========================================
    # 6.6 Print Final Summary
    # ==========================================
    print("\n" + "="*60)
    print("[SUCCESS] Training & Evaluation Complete!")
    print("="*60)
    print("\nGenerated Files for Report (Chapter 4):")
    print(f"  ✓ {report_path}")
    print(f"  ✓ {summary_report_path}")
    print(f"  ✓ {arch_report_path}")
    print(f"  ✓ {MODEL_SAVE_PATH}")
    print(f"  ✓ {TFLITE_EXPORT_PATH}")
    print("\nKey Metrics:")
    for key, value in metrics_summary.items():
        print(f"  {key}: {value}")