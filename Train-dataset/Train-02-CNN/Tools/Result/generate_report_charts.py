import numpy as np
import matplotlib.pyplot as plt

# ตั้งค่าฟอนต์และสไตล์ให้ดูเป็นมืออาชีพ
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# ==========================================
# 1. PLOT CONFUSION MATRIX HEATMAP
# ==========================================
cm = np.array([
    [711, 94],     # True Flushing -> [TP, FN]
    [654, 2260]    # True Normal   -> [FP, TN]
])

class_names = ['Flushing', 'Normal']
total_samples = np.sum(cm)

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
fig.colorbar(cax, fraction=0.046, pad=0.04)

ax.set(
    xticks=np.arange(cm.shape[1]), # [FIXED] Changed cm.shape to cm.shape[1]
    yticks=np.arange(cm.shape[0]),
    xticklabels=class_names,
    yticklabels=class_names,
    title='Confusion Matrix (Patient-Level Test Set: N=3,719)\n',
    ylabel='True Clinical Label',
    xlabel='Predicted Label by CNN'
)

# ปรับตำแหน่ง Label
plt.setp(ax.get_xticklabels(), fontsize=11, weight='bold')
plt.setp(ax.get_yticklabels(), fontsize=11, weight='bold')

# ใส่ตัวเลขและคำอธิบายทางการแพทย์ในแต่ละช่อง
# [FIXED] Changed {cm} to {cm[0, 1]} and {cm[1, 1]} so it prints the scalar value
annotations = [
    [f"{cm[0, 0]}\n(TP: 88.3%)\nTrue Severe", f"{cm[0, 1]}\n(FN: 11.7%)\nMissed Case"],
    [f"{cm[1, 0]}\n(FP: 22.4%)\nFalse Alarm", f"{cm[1, 1]}\n(TN: 77.6%)\nTrue Normal"]
]

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]): # [FIXED] Changed cm.shape to cm.shape[1]
        color = "white" if cm[i, j] > thresh else "#111111"
        ax.text(j, i, annotations[i][j],
                ha="center", va="center",
                color=color, fontsize=11, weight='bold')

plt.tight_layout()
plt.savefig('confusion_matrix_clean.png', dpi=300)
print("[+] Saved: 'confusion_matrix_clean.png'")

# ==========================================
# 2. PLOT METRICS SUMMARY BAR CHART
# ==========================================
metrics_names = ['Precision', 'Recall', 'F1-Score']
flushing_scores = [0.52, 0.88, 0.66]
normal_scores = [0.96, 0.78, 0.86]

x = np.arange(len(metrics_names))
width = 0.32

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

rects1 = ax.bar(x - width/2, flushing_scores, width, label='Flushing (Minority Class)', color='#e74c3c', edgecolor='black', alpha=0.9)
rects2 = ax.bar(x + width/2, normal_scores, width, label='Normal (Majority Class)', color='#3498db', edgecolor='black', alpha=0.9)

ax.set_ylabel('Score (0.0 - 1.0)', fontsize=11, weight='bold')
ax.set_title('Evaluation Metrics Comparison on Unseen Patients (Test Set)\n', fontsize=12, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=11, weight='bold')
ax.set_ylim(0, 1.15)
ax.legend(loc='upper right', frameon=True)
ax.grid(axis='y', linestyle='--', alpha=0.6)

# แสดงตัวเลขบนแท่งกราฟ
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, weight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('metrics_summary.png', dpi=300)
print("[+] Saved: 'metrics_summary.png'")