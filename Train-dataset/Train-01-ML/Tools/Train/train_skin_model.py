import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 1. โหลดข้อมูลจาก CSV ล่าสุด
DATA_FILE = "E:\SmartTriageAI\Train-dataset\skin_dataset_v2.csv"  # อัปเดตชื่อไฟล์ให้ตรงกับ Output V4 ของคุณ
print(f"กำลังโหลดข้อมูลจาก {DATA_FILE}...")

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"❌ ไม่พบไฟล์ {DATA_FILE} กรุณาตรวจสอบชื่อไฟล์อีกครั้ง")
    exit()

# 2. จัดเตรียมข้อมูล (Data Preparation)
# เลือกลักษณะเฉพาะที่เกี่ยวกับ "เม็ดสี" และ "ประเภทผิว" เท่านั้น 
# (ตัด L_mean เรื่องความสว่าง และ Used_ROIs เรื่องทิศทางหน้าทิ้งไปเลย)
X = df[['A_mean', 'B_mean', 'ITA_Angle']]

# แปลง Label ให้เป็นตัวเลข
le = LabelEncoder()
y = le.fit_transform(df['Label'])

# 3. แบ่ง Dataset (Train 70% | Val 15% | Test 15%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1764, random_state=42, stratify=y_temp)

print(f"\n📊 สรุปจำนวนข้อมูลที่ถูกแบ่ง:")
print(f" - Train Set: {len(X_train)} รูป (ใช้สอน AI)")
print(f" - Validation Set: {len(X_val)} รูป (ใช้ปรับจูนระหว่างสอน)")
print(f" - Test Set: {len(X_test)} รูป (ใช้สอบไล่ วัดผลจริง)")

# 4. ปรับสเกลข้อมูล (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 5. สร้างและเทรนโมเดล (Random Forest - Optimized Parameters)
print("\n🧠 กำลังเทรนโมเดล Random Forest...")
model = RandomForestClassifier(
    n_estimators=300, 
    criterion='entropy',
    max_depth=None, 
    min_samples_split=4,
    random_state=42, 
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train)

# 6. ประเมินผลโมเดล
val_preds = model.predict(X_val_scaled)
val_acc = accuracy_score(y_val, val_preds)

test_preds = model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, test_preds)

print("\n" + "="*40)
print(f"📈 ผลลัพธ์ความแม่นยำ (Accuracy):")
print(f" - Validation Accuracy : {val_acc * 100:.2f}%")
print(f" - Test Accuracy       : {test_acc * 100:.2f}%")
print("="*40)

print("\n🔍 Confusion Matrix (บน Test Set):")
print(confusion_matrix(y_test, test_preds))

print("\n📑 Classification Report:")
print(classification_report(y_test, test_preds, target_names=le.classes_))

# 7. Feature Importance
importances = model.feature_importances_
print("\n🌟 ความสำคัญของตัวแปร (Feature Importance):")
for feature, imp in zip(X.columns, importances):
    print(f" - {feature}: {imp * 100:.2f}%")

# 8. บันทึกโมเดล
MODEL_NAME = "skin_rf_model.pkl"
SCALER_NAME = "skin_scaler.pkl"
ENCODER_NAME = "skin_encoder.pkl"

joblib.dump(model, MODEL_NAME)
joblib.dump(scaler, SCALER_NAME)
joblib.dump(le, ENCODER_NAME)

print(f"\n💾 บันทึกโมเดลสำเร็จ! พร้อมเอาไปใช้ในกล้องจริงแล้วครับ")