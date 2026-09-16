"""
Patient-Wise Dataset Splitter
Guarantees all facial patches (_Forehead, _Zone_L, _Zone_R) from the same patient
stay strictly in the same split to avoid data leakage.
"""

import os
import shutil
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split

DATASET_ROOT = "Train-dataset/Train-02-CNN/cnn_dataset-02"
OUTPUT_ROOT = "Train-dataset/Train-02-CNN/split_dataset"

CLASSES = ["Normal", "Flushing"]

def extract_patient_id(filename):
    # Strips out _Forehead, _Zone_L, _Zone_R to get unique base patient ID
    base = re.sub(r'(_Forehead|_Zone_L|_Zone_R)\.jpg$', '', filename)
    return base

def run_split(val_ratio=0.20, random_seed=42):
    for cls in CLASSES:
        cls_dir = os.path.join(DATASET_ROOT, cls)
        if not os.path.exists(cls_dir):
            continue

        # Group all images by patient ID
        patient_files = defaultdict(list)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                pid = extract_patient_id(fname)
                patient_files[pid].append(fname)

        unique_patients = list(patient_files.keys())
        train_pids, val_pids = train_test_split(unique_patients, test_size=val_ratio, random_state=random_seed)

        # Copy to output directories
        for split_name, pids in [("train", train_pids), ("val", val_pids)]:
            target_dir = os.path.join(OUTPUT_ROOT, split_name, cls)
            os.makedirs(target_dir, exist_ok=True)
            for pid in pids:
                for fname in patient_files[pid]:
                    src = os.path.join(cls_dir, fname)
                    dst = os.path.join(target_dir, fname)
                    shutil.copy2(src, dst)

        print(f"[{cls}] {len(train_pids)} train patients, {len(val_pids)} val patients.")

if __name__ == "__main__":
    run_split()
    print(">>> Patient-wise split complete with zero leakage!")