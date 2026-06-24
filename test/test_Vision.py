import cv2  # OpenCV: ยักษ์ใหญ่แห่งวงการจัดการภาพ
import mediapipe as mp # พระเอกของเรา

# 1. เตรียมเครื่องมือวาดรูป (Drawing Utils)
# เอาไว้วาดเส้นเชื่อมจุดต่างๆ บนตัว (ถ้าไม่ใช้ตัวนี้ เราต้องนั่งวาดเส้นเองทีละเส้น)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 2. เปิดกล้อง Webcam (0 คือกล้องตัวแรกในเครื่อง)
cap = cv2.VideoCapture(0)

# เช็คว่ากล้องเปิดติดไหม
if not cap.isOpened():
    print("❌ เปิดกล้องไม่ติดเช็คสายด่วน!")
    exit()

print("✅ เปิดกล้องสำเร็จ... กด 'q' เพื่อปิดโปรแกรม")

# 3. เรียกใช้โมเดล Pose
# min_detection_confidence=0.5: ต้องมั่นใจเกิน 50% ถึงจะนับว่าเจอคน
# min_tracking_confidence=0.5: ต้องมั่นใจเกิน 50% ว่าจุดที่ตามอยู่คือจุดเดิม
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        # อ่านภาพจากกล้องทีละเฟรม (Frame by Frame)
        success, image = cap.read()
        if not success:
            continue

        # --- ขั้นตอน Pre-processing ---
        # OpenCV อ่านสีเป็น BGR (Blue-Green-Red) แต่ MediaPipe ชอบ RGB
        # เราต้องแปลงสีไม่งั้นสีเพี้ยนและโมเดลจะงง
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- ขั้นตอน AI Processing (ส่งภาพให้โมเดลคิด) ---
        results = pose.process(image)

        # --- ขั้นตอน Post-processing ---
        # แปลงสีกลับเป็น BGR เพื่อแสดงผลบนจอให้คนดู
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ถ้า AI หาตัวคนเจอ (results.pose_landmarks ไม่เป็นค่าว่าง)
        if results.pose_landmarks:
            # สั่งให้วาด "จุด (Landmarks)" และ "เส้นเชื่อม (Connections)" ลงบนภาพ
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # --- ส่วนนี้สำคัญ: การดึงข้อมูลพิกัด (Data Extraction) ---
            # ลองดึงพิกัด "จมูก (Nose)" ซึ่งคือจุดหมายเลข 0
            # landmarks จะเก็บค่า x, y เป็น % ของภาพ (0.0 ถึง 1.0)
            nose = results.pose_landmarks.landmark[0] 
            
            # ลอง print ดูค่าดิบๆ (Uncomment บรรทัดล่างดูถ้าอยากเห็น)
            # print(f"Nose: x={nose.x:.2f}, y={nose.y:.2f}")

        # 4. แสดงผลภาพออกหน้าจอ
        # พลิกภาพกระจก (Flip) ให้เหมือนเราส่องกระจก จะได้ไม่งงซ้ายขวา
        cv2.imshow('MediaPipe Pose - The Eye', cv2.flip(image, 1))

        # ถ้ากดปุ่ม 'q' (Quit) ให้หยุดลูป
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# 5. คืนทรัพยากรให้เครื่อง (ปิดกล้อง)
cap.release()
cv2.destroyAllWindows()