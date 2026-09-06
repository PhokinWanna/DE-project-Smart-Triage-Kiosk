import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 1. โหลดข้อมูลจาก CSV
DATA_FILE = "E:\SmartTriageAI\Train-dataset\skin_dataset_dynamic1.csv"  # เปลี่ยนชื่อไฟล์ให้ตรงกับของคุณ
print(f"กำลังโหลดข้อมูลจาก {DATA_FILE}...")

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"❌ ไม่พบไฟล์ {DATA_FILE} กรุณาตรวจสอบชื่อไฟล์อีกครั้ง")
    exit()

# 2. จัดเตรียมข้อมูล (Data Preparation)

# ตัด L_mean ทิ้ง บังคับให้ AI โฟกัสแค่เม็ดสี (A, B) และประเภทผิว (ITA)
X = df[['A_mean', 'B_mean', 'ITA_Angle']]

# คำตอบ (y) ที่ AI ต้องทายให้ถูก
# แปลง Label ('Normal', 'Flushing') ให้เป็นตัวเลข (0, 1)
le = LabelEncoder()
y = le.fit_transform(df['Label'])

# 3. แบ่ง Dataset (Train 70% | Validation 15% | Test 15%)
# แบ่งครั้งที่ 1: ดึง Test ออกมาก่อน 15% (เหลือ 85% สำหรับ Train+Val)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# แบ่งครั้งที่ 2: แบ่ง 85% ที่เหลือ เป็น Train กับ Validation
# (0.15 / 0.85 ≈ 0.1764)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1764, random_state=42, stratify=y_temp)

print(f"\n📊 สรุปจำนวนข้อมูลที่ถูกแบ่ง:")
print(f" - Train Set: {len(X_train)} รูป (ใช้สอน AI)")
print(f" - Validation Set: {len(X_val)} รูป (ใช้ปรับจูนระหว่างสอน)")
print(f" - Test Set: {len(X_test)} รูป (ใช้สอบไล่ วัดผลจริง)")

# 4. ปรับสเกลข้อมูล (Standardization)
# เพื่อให้ค่า L, A, B, ITA มีสเกลที่ใกล้เคียงกัน AI จะได้ไม่เอนเอียง
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 5. สร้างและเทรนโมเดล (Training Random Forest)
print("\n🧠 กำลังเทรนโมเดล Random Forest...")
# เพิ่มต้นไม้เป็น 300 ต้น (จาก 100), ปล่อยให้เรียนรู้ลึกขึ้น (ไม่มี max_depth), 
# และใช้ entropy ช่วยเกลี่ย Information Gain
model = RandomForestClassifier(
    n_estimators=300, 
    criterion='entropy',
    max_depth=None, 
    min_samples_split=4,
    random_state=42, 
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train)

# 6. ประเมินผลโมเดล (Evaluation)
# วัดด้วย Validation Set ก่อน
val_preds = model.predict(X_val_scaled)
val_acc = accuracy_score(y_val, val_preds)

# วัดผลจริงด้วย Test Set (ข้อมูลที่ AI ไม่เคยเห็นมาก่อน)
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

# 7. ดูว่า Feature ไหนสำคัญที่สุดในสายตา AI
importances = model.feature_importances_
print("\n🌟 ความสำคัญของตัวแปร (Feature Importance):")
for feature, imp in zip(X.columns, importances):
    print(f" - {feature}: {imp * 100:.2f}%")

# 8. บันทึกโมเดลไว้ใช้งานในกล้อง (Export Model)
MODEL_NAME = "skin_rf_model.pkl"
SCALER_NAME = "skin_scaler.pkl"
ENCODER_NAME = "skin_encoder.pkl"

joblib.dump(model, MODEL_NAME)
joblib.dump(scaler, SCALER_NAME)
joblib.dump(le, ENCODER_NAME)

print(f"\n💾 บันทึกโมเดลสำเร็จ! (เซฟเป็นไฟล์ {MODEL_NAME}, {SCALER_NAME}, {ENCODER_NAME})")