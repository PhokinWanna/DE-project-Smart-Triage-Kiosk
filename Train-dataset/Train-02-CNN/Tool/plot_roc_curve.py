import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# โหลดโมเดลและชุดข้อมูล Test
model = tf.keras.models.load_model("best_skin_cnn.keras")
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join("cnn_dataset_split", "test"),
    image_size=(64, 64),
    batch_size=32,
    label_mode="binary",
    shuffle=False
)

y_true = []
y_pred_probs = []

for imgs, lbls in test_ds:
    y_true.extend(lbls.numpy().flatten())
    y_pred_probs.extend(model.predict(imgs, verbose=0).flatten())

fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

# วาดกราฟ ROC Curve
plt.figure(figsize=(7, 6), dpi=300)
plt.plot(fpr, tpr, color='#e67e22', lw=2.5, label=f'Skin CNN ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='#7f8c8d', lw=1.5, linestyle='--', label='Random Chance') # Fixed line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, weight='bold')
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, weight='bold')
plt.title('Receiver Operating Characteristic (ROC) on Test Set\n', fontsize=12, weight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('roc_curve_clean.png', dpi=300)
print(f"[+] Saved: 'roc_curve_clean.png' (AUC = {roc_auc:.4f})")