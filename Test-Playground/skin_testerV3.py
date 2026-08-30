import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def get_indices_from_edges(edges):
    indices = set()
    for edge in edges:
        indices.add(edge[0])
        indices.add(edge[1])
    return list(indices)

def get_pure_skin_lab(image_path):
    image = cv2.imread(image_path)
    if image is None: return None, None, None, None
    h_img, w_img, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks: return None, None, None, None

    landmarks = results.multi_face_landmarks[0].landmark
    mask = np.zeros((h_img, w_img), dtype=np.uint8)

    def draw_feature_hull(indices, color):
        pts = np.array([[int(landmarks[idx].x * w_img), int(landmarks[idx].y * h_img)] for idx in indices], np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, color)

    # 1. วาดโครงหน้า (ขาว) แล้วเจาะ ตา คิ้ว ปาก (ดำ) ทิ้งไป
    face_oval = get_indices_from_edges(mp_face_mesh.FACEMESH_FACE_OVAL)
    left_eye = get_indices_from_edges(mp_face_mesh.FACEMESH_LEFT_EYE)
    right_eye = get_indices_from_edges(mp_face_mesh.FACEMESH_RIGHT_EYE)
    lips = get_indices_from_edges(mp_face_mesh.FACEMESH_LIPS)
    left_eyebrow = get_indices_from_edges(mp_face_mesh.FACEMESH_LEFT_EYEBROW)
    right_eyebrow = get_indices_from_edges(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)

    draw_feature_hull(face_oval, 255)
    draw_feature_hull(left_eye, 0)
    draw_feature_hull(right_eye, 0)
    draw_feature_hull(lips, 0)
    draw_feature_hull(left_eyebrow, 0)
    draw_feature_hull(right_eyebrow, 0)

    # 2. แปลงเป็น LAB Color Space (The Secret Weapon)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 3. หาค่าเฉลี่ย "เฉพาะ" พื้นที่หน้ากากผิวหนัง
    mean_color = cv2.mean(lab_image, mask=mask)
    avg_L = mean_color[0] # ความสว่าง
    avg_A = mean_color[1] # *** แกน เขียว-แดง (ยิ่งมาก ยิ่งแดง) ***
    avg_B = mean_color[2] # แกน น้ำเงิน-เหลือง

    skin_only = cv2.bitwise_and(image, image, mask=mask)
    return avg_L, avg_A, avg_B, skin_only

# ==========================================
# ส่วนทดสอบ
# ==========================================
normal_img = "E:\SmartTriageAI\Face_normal.png"   # รูปหน้าปกติ
symptom_img = "E:\SmartTriageAI\Face_red.png" # รูปตอนมีอาการ (ซีด/แดง)

print("--- AI LAB SKIN COLOR TESTER ---")
base_L, base_A, base_B, img_normal = get_pure_skin_lab(normal_img)
symp_L, symp_A, symp_B, img_symp = get_pure_skin_lab(symptom_img)

if base_A is not None and symp_A is not None:
    print(f"[1] NORMAL  -> Lightness (L): {base_L:.2f} | Redness (A-Channel): {base_A:.2f}")
    print(f"[2] SYMPTOM -> Lightness (L): {symp_L:.2f} | Redness (A-Channel): {symp_A:.2f}")
    
    # คำนวณความเปลี่ยนแปลงของแกนสีแดง (A-channel)
    a_shift = symp_A - base_A
    
    print(f"    -> Redness shifted by : {a_shift:+.2f} units")
    
    # ตรรกะใหม่: สนใจแค่ A-Channel
    if a_shift > 1.5:  # ถ้าค่าแดงเพิ่มขึ้นเกิน 1.5 หน่วย ถือว่าหน้าแดง
        print("\n>>> RESULT: FLUSHING (RED) DETECTED! <<<")
    elif a_shift < -1.5: # ถ้าค่าความแดงหายไปเกิน 1.5 หน่วย ถือว่าหน้าซีด
        print("\n>>> RESULT: PALLOR (PALE) DETECTED! <<<")
    else:
        print("\n>>> RESULT: NORMAL SKIN <<<")
        
    # โชว์ภาพผิวที่ถูกเจาะแล้ว
    cv2.imshow("Mask Normal", img_normal)
    cv2.imshow("Mask Red", img_symp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()