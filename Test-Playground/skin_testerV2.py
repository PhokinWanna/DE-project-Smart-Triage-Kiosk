import cv2
import mediapipe as mp
import numpy as np

# เรียกใช้ Face Mesh ที่มีความละเอียด 468 จุด
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def get_indices_from_edges(edges):
    """ฟังก์ชันดึงเลข Index จุด จากเส้นเชื่อมของ MediaPipe"""
    indices = set()
    for edge in edges:
        indices.add(edge[0])
        indices.add(edge[1])
    return list(indices)

def get_pure_skin_color(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: หาไฟล์ {image_path} ไม่เจอ!")
        return None, None, None

    h_img, w_img, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print(f"Error: จับใบหน้าในรูป {image_path} ไม่ได้!")
        return None, None, None

    landmarks = results.multi_face_landmarks[0].landmark

    # 1. สร้างหน้ากากดำๆ ว่างๆ ขนาดเท่ารูปภาพ
    mask = np.zeros((h_img, w_img), dtype=np.uint8)

    def draw_feature_hull(indices, color):
        """ฟังก์ชันวาด Polygon ครอบจุดอวัยวะ"""
        pts = np.array([[int(landmarks[idx].x * w_img), int(landmarks[idx].y * h_img)] for idx in indices], np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, color)

    # 2. ดึง Index ของอวัยวะต่างๆ จาก MediaPipe
    face_oval = get_indices_from_edges(mp_face_mesh.FACEMESH_FACE_OVAL)
    left_eye = get_indices_from_edges(mp_face_mesh.FACEMESH_LEFT_EYE)
    right_eye = get_indices_from_edges(mp_face_mesh.FACEMESH_RIGHT_EYE)
    lips = get_indices_from_edges(mp_face_mesh.FACEMESH_LIPS)
    left_eyebrow = get_indices_from_edges(mp_face_mesh.FACEMESH_LEFT_EYEBROW)
    right_eyebrow = get_indices_from_edges(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)

    # 3. ระบายสีขาว (255) ให้เต็มโครงหน้า
    draw_feature_hull(face_oval, 255)
    
    # 4. เจาะรู ตา คิ้ว ปาก โดยระบายสีดำ (0) ทับลงไป (นี่คือวิธีกำจัดจุดที่กวนใจเรา!)
    draw_feature_hull(left_eye, 0)
    draw_feature_hull(right_eye, 0)
    draw_feature_hull(lips, 0)
    draw_feature_hull(left_eyebrow, 0)
    draw_feature_hull(right_eyebrow, 0)

    # นำหน้ากากไปทาบกับรูปจริง เพื่อให้เห็นเฉพาะผิวหนัง (ส่วนอื่นจะดำสนิท)
    skin_only = cv2.bitwise_and(image, image, mask=mask)

    # 5. แปลงเป็น HSV และหาค่าเฉลี่ย "เฉพาะพิกเซลที่หน้ากากเป็นสีขาว"
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_color = cv2.mean(hsv_image, mask=mask) # สุดยอดคำสั่ง: หาค่าเฉลี่ยข้ามพิกเซลสีดำไปเลย
    
    avg_h = mean_color[0]
    avg_s = mean_color[1]

    return avg_h, avg_s, skin_only


normal_img_path = "E:\SmartTriageAI\\Normal2.png"   
symptom_img_path = "E:\SmartTriageAI\\Red2.png" 

print("--- AI PURE SKIN COLOR TESTER ---")
base_h, base_s, img_normal_skin = get_pure_skin_color(normal_img_path)

if base_h is not None:
    print(f"NORMAL   Baseline S: {base_s:.2f}, Baseline H: {base_h:.2f}")
    
    symp_h, symp_s, img_symp_skin = get_pure_skin_color(symptom_img_path)
    if symp_h is not None:
        print(f"SYMPTOM  Current  S: {symp_s:.2f}, Current  H: {symp_h:.2f}")
        
        
        s_drop_percent = ((base_s - symp_s) / base_s) * 100 if base_s > 0 else 0
        h_shift = base_h - symp_h
        
        print(f" S Dropped by : {s_drop_percent:.1f}%")
        print(f" H Shifted by : {h_shift:.1f} units")
        
        if s_drop_percent > 5.0:  
            print("\n>>> RESULT: PALLOR (PALE) DETECTED! <<<")
        elif h_shift > -5.0 and base_s >= symp_s: #* 1.05:
            print("\n>>> RESULT: FLUSHING (RED) DETECTED! <<<")
        else:
            print("\n>>> RESULT: NORMAL SKIN <<<")

        cv2.imshow(" Baseline", img_normal_skin)
        cv2.imshow(" Tested", img_symp_skin)
        cv2.waitKey(0)
        cv2.destroyAllWindows()