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

def analyze_single_skin_image(image_path):
    image = cv2.imread(image_path)
    if image is None: 
        print(f"Error: หาไฟล์ {image_path} ไม่เจอ!")
        return
        
    h_img, w_img, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks: 
        print(f"Error: ไม่พบใบหน้า")
        return

    landmarks = results.multi_face_landmarks[0].landmark
    mask = np.zeros((h_img, w_img), dtype=np.uint8)

    def draw_feature_hull(indices, color):
        pts = np.array([[int(landmarks[idx].x * w_img), int(landmarks[idx].y * h_img)] for idx in indices], np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, color)

    face_oval = get_indices_from_edges(mp_face_mesh.FACEMESH_FACE_OVAL)
    left_eye = get_indices_from_edges(mp_face_mesh.FACEMESH_LEFT_EYE)
    right_eye = get_indices_from_edges(mp_face_mesh.FACEMESH_RIGHT_EYE)
    lips = get_indices_from_edges(mp_face_mesh.FACEMESH_LIPS)
    l_brow = get_indices_from_edges(mp_face_mesh.FACEMESH_LEFT_EYEBROW)
    r_brow = get_indices_from_edges(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)

    draw_feature_hull(face_oval, 255)
    draw_feature_hull(left_eye, 0)
    draw_feature_hull(right_eye, 0)
    draw_feature_hull(lips, 0)
    draw_feature_hull(l_brow, 0)
    draw_feature_hull(r_brow, 0)

    # แปลงสีและหาค่าเฉพาะบริเวณผิว
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    mean_color = cv2.mean(lab_image, mask=mask)
    
    avg_L = mean_color[0] # ความสว่าง
    avg_A = mean_color[1] # ความแดง (เป้าหมายหลัก)
    
    print("\n" + "="*40)
    print(f"File: {image_path}")
    print(f"-> Lightness (L) : {avg_L:.2f}")
    print(f"-> Redness   (A) : {avg_A:.2f}")
    print("="*40)

    # -----------------------------------------------------
    # ตรรกะ ABSOLUTE THRESHOLD (เช็คจากค่า A ตายตัว)
    # คุณต้องหาตัวเลขที่เหมาะสมที่สุด (ตอนนี้ผมสมมติที่ 132 กับ 148)
    # -----------------------------------------------------
    if avg_A > 148.0:
        print(">>> RESULT: FLUSHING (RED) DETECTED! <<<")
    elif avg_A < 132.0:
        print(">>> RESULT: PALLOR (PALE) DETECTED! <<<")
    else:
        print(">>> RESULT: NORMAL SKIN <<<")

    skin_only = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow(f"Analysis: {image_path}", skin_only)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==========================================
# ใส่ชื่อไฟล์ตรงนี้ได้เลย (เทสทีละรูป)
# ==========================================
# test_image = "E:\SmartTriageAI\Face_red.png" 
# test_image2 = "E:\SmartTriageAI\Face_normal.png"
# test_image3 = "E:\SmartTriageAI\\" + "Normal2.png"
# test_image4 = "E:\SmartTriageAI\\" + "Red2.png"
test_image5 = "E:\SmartTriageAI\\" + "Red3.jpg"

# analyze_single_skin_image(test_image)
# analyze_single_skin_image(test_image2)
# analyze_single_skin_image(test_image3)
# analyze_single_skin_image(test_image4)
analyze_single_skin_image(test_image5)