import os
import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_DIR = "dataset"            # Source images: dataset/Normal, dataset/Flushing
OUTPUT_DIR = "cnn_dataset"      # Destination folder
CATEGORIES = ["Normal", "Flushing"]
PATCH_SIZE = 64

FOREHEAD_HEIGHT_RATIO = 0.35
FOREHEAD_TOP_EXPAND = 0.08

MIN_ZONE_WIDTH = 25
MIN_ZONE_HEIGHT = 20
MIN_SKIN_PIXELS = 350

NOSE_TIP_IDX = 1
LEFT_EYE_OUTER_IDX = 33
RIGHT_EYE_OUTER_IDX = 263

Y_AXIS = 1
X_AXIS = 0

mp_face_mesh = mp.solutions.face_mesh


def create_dirs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for cat in CATEGORIES:
        path = os.path.join(OUTPUT_DIR, cat)
        if not os.path.exists(path):
            os.makedirs(path)


def get_indices_from_edges(edges):
    indices = set()
    for start_node, end_node in edges:
        indices.add(int(start_node))
        indices.add(int(end_node))
    return list(indices)


def get_expanded_face_points(landmarks, face_oval_indices, w_img, h_img, top_expand_ratio=0.0):
    pts = []
    for idx in face_oval_indices:
        pts.append([int(landmarks[idx].x * w_img), int(landmarks[idx].y * h_img)])
    pts = np.array(pts, dtype=np.int32)

    if top_expand_ratio > 0:
        _, y, _, h = cv2.boundingRect(pts)
        dy = int(h * top_expand_ratio)
        threshold_y = y + int(h * 0.25)
        for row in pts:
            curr_y = row[Y_AXIS]
            if curr_y < threshold_y:
                factor = (threshold_y - curr_y) / max(1, (threshold_y - y))
                row[Y_AXIS] = max(0, curr_y - int(dy * factor))
    return pts


def draw_polygon_hull(mask, pts, color):
    if len(pts) > 2:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, color)


def draw_feature_hull(mask, landmarks, indices, w_img, h_img, color):
    pts = []
    for idx in indices:
        pts.append([int(landmarks[idx].x * w_img), int(landmarks[idx].y * h_img)])
    pts = np.array(pts, dtype=np.int32)
    draw_polygon_hull(mask, pts, color)


def get_bbox_from_mask(mask):
    rows = np.any(mask, axis=Y_AXIS)
    cols = np.any(mask, axis=X_AXIS)
    if not (np.any(rows) and np.any(cols)):
        return None

    row_indices = np.where(rows)[X_AXIS]
    col_indices = np.where(cols)[X_AXIS]
    min_x = int(col_indices[X_AXIS])
    max_x = int(col_indices[-1])
    min_y = int(row_indices[X_AXIS])
    max_y = int(row_indices[-1])
    return min_x, min_y, max_x - min_x, max_y - min_y


def detect_head_turn(landmarks, w_img, f_w):
    nose_x = landmarks[NOSE_TIP_IDX].x * w_img
    l_eye_x = landmarks[LEFT_EYE_OUTER_IDX].x * w_img
    r_eye_x = landmarks[RIGHT_EYE_OUTER_IDX].x * w_img

    d_left = nose_x - l_eye_x
    d_right = r_eye_x - nose_x

    if d_right < 0.06 * f_w or (d_left > 0 and (d_right / max(1.0, d_left)) < 0.22):
        return "TURNED_RIGHT"
    elif d_left < 0.06 * f_w or (d_right > 0 and (d_left / max(1.0, d_right)) < 0.22):
        return "TURNED_LEFT"
    else:
        return "FRONTAL"


def extract_masked_patches():
    create_dirs()
    print(f"Starting Pose-Aware Patch Extraction ({PATCH_SIZE}x{PATCH_SIZE})...\n")

    total_extracted = 0
    total_skipped_zones = 0
    failed_files = []

    face_oval = get_indices_from_edges(mp_face_mesh.FACEMESH_FACE_OVAL)
    left_eye = get_indices_from_edges(mp_face_mesh.FACEMESH_LEFT_EYE)
    right_eye = get_indices_from_edges(mp_face_mesh.FACEMESH_RIGHT_EYE)
    lips = get_indices_from_edges(mp_face_mesh.FACEMESH_LIPS)
    l_brow = get_indices_from_edges(mp_face_mesh.FACEMESH_LEFT_EYEBROW)
    r_brow = get_indices_from_edges(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)

    

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        for category in CATEGORIES:
            folder_path = os.path.join(INPUT_DIR, category)
            if not os.path.exists(folder_path):
                continue

            for filename in os.listdir(folder_path):
                if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                img_path = os.path.join(folder_path, filename)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                h_img, w_img, _ = image.shape
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)

                if not results.multi_face_landmarks:
                    print(f"[-] Face not found: {filename}")
                    failed_files.append(f"{category}/{filename}")
                    continue

                landmarks = results.multi_face_landmarks[0].landmark

                # Base mask with forehead expansion
                expanded_face_pts = get_expanded_face_points(
                    landmarks, face_oval, w_img, h_img, top_expand_ratio=FOREHEAD_TOP_EXPAND
                )
                base_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                draw_polygon_hull(base_mask, expanded_face_pts, 255)

                for feature in [left_eye, right_eye, lips, l_brow, r_brow]:
                    draw_feature_hull(base_mask, landmarks, feature, w_img, h_img, 0)

                f_x, f_y, f_w, f_h = cv2.boundingRect(expanded_face_pts)
                forehead_bottom = f_y + int(f_h * FOREHEAD_HEIGHT_RATIO)
                cheek_top = forehead_bottom

                pose_state = detect_head_turn(landmarks, w_img, f_w)
                nose_x = int(landmarks[NOSE_TIP_IDX].x * w_img)

                mask_forehead = base_mask.copy()
                mask_forehead[forehead_bottom:, :] = 0

                mask_left = None
                mask_right = None

                if pose_state == "TURNED_RIGHT":
                    mask_left = base_mask.copy()
                    mask_left[:cheek_top, :] = 0
                elif pose_state == "TURNED_LEFT":
                    mask_right = base_mask.copy()
                    mask_right[:cheek_top, :] = 0
                else:
                    mask_left = base_mask.copy()
                    mask_left[:cheek_top, :] = 0
                    mask_left[:, nose_x:] = 0

                    mask_right = base_mask.copy()
                    mask_right[:cheek_top, :] = 0
                    mask_right[:, :nose_x] = 0

                zones = [
                    ("Forehead", mask_forehead),
                    ("Zone_L", mask_left),
                    ("Zone_R", mask_right),
                ]

                base_name = os.path.splitext(filename)[0]

                for zone_name, zone_mask in zones:
                    if zone_mask is None:
                        total_skipped_zones += 1
                        continue

                    bbox = get_bbox_from_mask(zone_mask)
                    if bbox is None:
                        total_skipped_zones += 1
                        continue

                    # Unpack bbox dimensions cleanly
                    x_box, y_box, w_box, h_box = bbox
                    skin_pixels = np.count_nonzero(zone_mask)

                    # Filter out narrow slivers and obstructed patches
                    if w_box < MIN_ZONE_WIDTH or h_box < MIN_ZONE_HEIGHT or skin_pixels < MIN_SKIN_PIXELS:
                        total_skipped_zones += 1
                        continue

                    cropped_img = image[y_box : y_box + h_box, x_box : x_box + w_box]
                    cropped_mask = zone_mask[y_box : y_box + h_box, x_box : x_box + w_box]

                    cropped_lab = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2LAB)
                    skin_lab_masked = cv2.bitwise_and(cropped_lab, cropped_lab, mask=cropped_mask)

                    mean_color = cv2.mean(cropped_lab, mask=cropped_mask)[:3]
                    skin_lab_masked[cropped_mask == 0] = mean_color

                    h_crop, w_crop = skin_lab_masked.shape[:2]
                    side = max(h_crop, w_crop)

                    pad_y_top = (side - h_crop) // 2
                    pad_y_bottom = side - h_crop - pad_y_top
                    pad_x_left = (side - w_crop) // 2
                    pad_x_right = side - w_crop - pad_x_left

                    padded_patch = cv2.copyMakeBorder(
                        skin_lab_masked,
                        pad_y_top,
                        pad_y_bottom,
                        pad_x_left,
                        pad_x_right,
                        cv2.BORDER_CONSTANT,
                        value=mean_color,
                    )

                    final_patch = cv2.resize(padded_patch, (PATCH_SIZE, PATCH_SIZE))

                    save_name = f"{base_name}_{zone_name}.jpg"
                    save_path = os.path.join(OUTPUT_DIR, category, save_name)
                    cv2.imwrite(save_path, final_patch)
                    total_extracted += 1

                print(f"[+] Processed: {filename} ({pose_state})")
    if failed_files:
        log_path = os.path.join(OUTPUT_DIR, "skipped_images.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            for item in failed_files:
                f.write(item + "\n")
        print(f"Recorded {len(failed_files)} undetected images in: {log_path}")
        
    print("\n" + "=" * 50)
    print(f"Extraction Completed!")
    print(f"Total pure patches extracted: {total_extracted}")
    print(f"Total occluded/corrupted zones safely skipped: {total_skipped_zones}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    extract_masked_patches()