"""
Patient-Wise 3-Way Dataset Splitter (Train / Val / Test)
Author: Phokin Wanna (6630613024)

Resolves paths dynamically and reports exact patient/image distribution.
"""

import os
import shutil
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split

# 1. Dynamic Path Resolution (Works from any terminal directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # .../Tool
CNN_ROOT = os.path.dirname(SCRIPT_DIR)                  # .../Train-02-CNN

def find_dataset_root():
    candidates = [
        os.path.join(CNN_ROOT, "cnn_dataset-02"),
        os.path.abspath("Train-dataset/Train-02-CNN/cnn_dataset-02"),
        os.path.abspath("Train-02-CNN/cnn_dataset-02"),
        os.path.abspath("cnn_dataset-02")
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"\n[ERROR] Could not locate 'cnn_dataset-02' folder!\nSearched locations:\n" +
        "\n".join(f" - {c}" for c in candidates)
    )

DATASET_ROOT = find_dataset_root()
OUTPUT_ROOT = os.path.join(CNN_ROOT, "split_dataset")

# Target Classes: Normal and Flushing (Pallor excluded per scope)
TARGET_CLASSES = ["Normal", "Flushing"]

def extract_patient_id(filename):
    # Strips _Forehead, _Zone_L, _Zone_R to obtain unique base subject ID
    return re.sub(r'(_Forehead|_Zone_L|_Zone_R)\.(jpg|jpeg|png)$', '', filename, flags=re.IGNORECASE)

def run_3way_split(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    print(f"[INFO] Source Dataset Location : {DATASET_ROOT}")
    print(f"[INFO] Output Split Location   : {OUTPUT_ROOT}\n")

    # Wipe prior split directory to guarantee a clean slate
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)

    total_images_processed = 0

    for cls in TARGET_CLASSES:
        # Case-insensitive class directory match
        matched_dirs = [d for d in os.listdir(DATASET_ROOT) if d.lower() == cls.lower()]
        if not matched_dirs:
            print(f"[WARNING] Directory for class '{cls}' not found in {DATASET_ROOT}. Skipping.")
            continue
        
        cls_dir = os.path.join(DATASET_ROOT, matched_dirs[0])

        # Group image crops by patient ID
        patient_files = defaultdict(list)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                pid = extract_patient_id(fname)
                patient_files[pid].append(fname)

        unique_patients = list(patient_files.keys())
        total_class_images = sum(len(f) for f in patient_files.values())

        if len(unique_patients) == 0:
            print(f"[WARNING] Class '{cls}' has 0 images. Skipping.")
            continue

        # Step 1: Split Train (70%) vs Temp (30%)
        train_pids, temp_pids = train_test_split(
            unique_patients,
            test_size=(val_ratio + test_ratio),
            random_state=random_seed
        )

        # Step 2: Split Temp into Val (15%) and Test (15%)
        val_pids, test_pids = train_test_split(
            temp_pids,
            test_size=0.50,
            random_state=random_seed
        )

        # Step 3: Copy physical files into train, val, test
        splits = [("train", train_pids), ("val", val_pids), ("test", test_pids)]
        counts = {}

        for split_name, pids in splits:
            target_dir = os.path.join(OUTPUT_ROOT, split_name, cls)
            os.makedirs(target_dir, exist_ok=True)
            copied = 0
            for pid in pids:
                for fname in patient_files[pid]:
                    shutil.copy2(os.path.join(cls_dir, fname), os.path.join(target_dir, fname))
                    copied += 1
            counts[split_name] = (len(pids), copied)
            total_images_processed += copied

        print(f"[{cls.upper()}] Summary:")
        print(f"  • Total Patients: {len(unique_patients)} | Total Images: {total_class_images}")
        print(f"  • Train : {counts['train'][0]} patients ({counts['train']} images)")
        print(f"  • Val   : {counts['val'][0]} patients ({counts['val']} images)")
        print(f"  • Test  : {counts['test'][0]} patients ({counts['test']} images)\n")

    print("--------------------------------------------------")
    print(f"[SUCCESS] Copied {total_images_processed} total images into 3 splits.")
    print(f"[OUTPUT] Check directory: {OUTPUT_ROOT}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    run_3way_split()