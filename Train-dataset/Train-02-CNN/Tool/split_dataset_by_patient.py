import os
import shutil
import random
from collections import defaultdict

# ==========================================
# CONFIGURATION
# ==========================================
SOURCE_DIR = "cnn_dataset-02"       # โฟลเดอร์ที่เก็บรูปสกัด Normal/ และ Flushing/
TARGET_DIR = "cnn_dataset_split"    # โฟลเดอร์ปลายทางที่แบ่ง Train/Val/Test
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(SEED)

classes = ["Normal", "Flushing"]
splits = ["train", "val", "test"]

# สร้างโครงสร้างโฟลเดอร์ปลายทาง
for s in splits:
    for c in classes:
        os.makedirs(os.path.join(TARGET_DIR, s, c), exist_ok=True)

print("="*60)
print("[*] Starting Patient-Level Group Split (No Data Leakage)...")
print("="*60)

for class_name in classes:
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.exists(class_path):
        continue

    files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # 1. จัดกลุ่ม Patch ตามรหัสภาพต้นฉบับ (Base Subject ID)
    subject_groups = defaultdict(list)
    for filename in files:
        # ตัด Suffix โซนออก เช่น 00702_Normal_Zone_L.jpg -> 00702_Normal
        base_id = filename
        for zone in ["_Forehead", "_Zone_L", "_Zone_R"]:
            if zone in filename:
                base_id = filename.split(zone)[0]
                break
        subject_groups[base_id].append(filename)

    unique_subjects = list(subject_groups.keys())
    random.shuffle(unique_subjects)

    total_subjects = len(unique_subjects)
    n_train = int(total_subjects * TRAIN_RATIO)
    n_val = int(total_subjects * VAL_RATIO)

    train_subjects = set(unique_subjects[:n_train])
    val_subjects = set(unique_subjects[n_train:n_train + n_val])
    test_subjects = set(unique_subjects[n_train + n_val:])

    print(f"\n[*] Class: {class_name}")
    print(f"    Total Unique Subjects: {total_subjects}")
    print(f"    Train Subjects: {len(train_subjects)} | Val Subjects: {len(val_subjects)} | Test Subjects: {len(test_subjects)}")

    # 2. คัดลอกไฟล์ Patch ไปยังโฟลเดอร์ตามกลุ่ม
    counts = {"train": 0, "val": 0, "test": 0}
    for sub_id, patch_list in subject_groups.items():
        if sub_id in train_subjects:
            target_split = "train"
        elif sub_id in val_subjects:
            target_split = "val"
        else:
            target_split = "test"

        for patch_file in patch_list:
            src = os.path.join(class_path, patch_file)
            dst = os.path.join(TARGET_DIR, target_split, class_name, patch_file)
            shutil.copy2(src, dst)
            counts[target_split] += 1

    print(f"    [+] Patches Copied -> Train: {counts['train']}, Val: {counts['val']}, Test: {counts['test']}")

print("\n" + "="*60)
print("[+] DATASET SPLIT COMPLETE! All patches cleanly segregated.")
print(f"[+] Output Directory: {TARGET_DIR}")
print("="*60)