import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PATCH_SIZE = 64
FOREHEAD_HEIGHT_RATIO = 0.35    # Top 35% of face height
CHEEK_HEIGHT_RATIO = 0.50       # Middle 50% of face height
FOREHEAD_TOP_EXPAND = 0.11      # 8% upward expansion to cover full forehead

# Minimum thresholds for usable skin (filters out narrow slivers)
MIN_ZONE_WIDTH = 25             # Minimum width in pixels
MIN_ZONE_HEIGHT = 20            # Minimum height in pixels
MIN_SKIN_PIXELS = 350           # Minimum non-masked skin pixels

# Landmark indices
NOSE_TIP_IDX = 1
LEFT_EYE_OUTER_IDX = 33
RIGHT_EYE_OUTER_IDX = 263

Y_AXIS = 1
X_AXIS = 0

mp_face_mesh = mp.solutions.face_mesh


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

    d_left = nose_x - l_eye_x     # Distance from left eye to nose
    d_right = r_eye_x - nose_x    # Distance from nose to right eye

    if d_right < 0.06 * f_w or (d_left > 0 and (d_right / max(1.0, d_left)) < 0.22):
        return "TURNED_RIGHT", d_left, d_right
    elif d_left < 0.06 * f_w or (d_right > 0 and (d_left / max(1.0, d_right)) < 0.22):
        return "TURNED_LEFT", d_left, d_right
    else:
        return "FRONTAL", d_left, d_right


def preview_face_crop(image_path, save_preview=True):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    h_img, w_img, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_oval = get_indices_from_edges(mp_face_mesh.FACEMESH_FACE_OVAL)
    left_eye = get_indices_from_edges(mp_face_mesh.FACEMESH_LEFT_EYE)
    right_eye = get_indices_from_edges(mp_face_mesh.FACEMESH_RIGHT_EYE)
    lips = get_indices_from_edges(mp_face_mesh.FACEMESH_LIPS)
    l_brow = get_indices_from_edges(mp_face_mesh.FACEMESH_LEFT_EYEBROW)
    r_brow = get_indices_from_edges(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            print("No face detected.")
            return

        landmarks = results.multi_face_landmarks[0].landmark

        # 1. Base mask with forehead expansion
        expanded_face_pts = get_expanded_face_points(
            landmarks, face_oval, w_img, h_img, top_expand_ratio=FOREHEAD_TOP_EXPAND
        )
        base_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        draw_polygon_hull(base_mask, expanded_face_pts, 255)

        for feature in [left_eye, right_eye, lips, l_brow, r_brow]:
            draw_feature_hull(base_mask, landmarks, feature, w_img, h_img, 0)

        # 2. Geometric bounds
        f_x, f_y, f_w, f_h = cv2.boundingRect(expanded_face_pts)
        forehead_bottom = f_y + int(f_h * FOREHEAD_HEIGHT_RATIO)
        cheek_top = forehead_bottom

        # 3. Detect head orientation
        pose_state, d_l, d_r = detect_head_turn(landmarks, w_img, f_w)
        nose_x = int(landmarks[NOSE_TIP_IDX].x * w_img)
        print(f"Detected Pose: {pose_state} (d_left={d_l:.1f}, d_right={d_r:.1f})")

        # 4. Generate zone masks adaptively
        mask_forehead = base_mask.copy()
        mask_forehead[forehead_bottom:, :] = 0

        mask_left = np.zeros_like(base_mask)
        mask_right = np.zeros_like(base_mask)

        if pose_state == "TURNED_RIGHT":
            # Right cheek is occluded; Left cheek occupies the entire visible cheek area
            mask_left = base_mask.copy()
            mask_left[:cheek_top, :] = 0
            mask_right = None
        elif pose_state == "TURNED_LEFT":
            # Left cheek is occluded; Right cheek occupies the entire visible cheek area
            mask_right = base_mask.copy()
            mask_right[:cheek_top, :] = 0
            mask_left = None
        else:
            # Frontal view: split down the nose
            mask_left = base_mask.copy()
            mask_left[:cheek_top, :] = 0
            mask_left[:, nose_x:] = 0

            mask_right = base_mask.copy()
            mask_right[:cheek_top, :] = 0
            mask_right[:, :nose_x] = 0

        candidate_zones = [
            ("Forehead", mask_forehead, (255, 150, 50)),
            ("Zone_L", mask_left, (50, 200, 50)),
            ("Zone_R", mask_right, (50, 50, 220)),
        ]

        # 5. Overlay on full image
        vis_img = image.copy()
        overlay = image.copy()

        for zone_name, z_mask, color in candidate_zones:
            if z_mask is not None and np.count_nonzero(z_mask) > 0:
                overlay[z_mask > 0] = color

        vis_img = cv2.addWeighted(overlay, 0.35, vis_img, 0.65, 0)
        cv2.rectangle(vis_img, (f_x, f_y), (f_x + f_w, f_y + f_h), (0, 255, 255), 2)
        cv2.line(vis_img, (f_x, forehead_bottom), (f_x + f_w, forehead_bottom), (255, 255, 0), 2)

        if pose_state == "FRONTAL":
            cv2.line(vis_img, (nose_x, cheek_top), (nose_x, f_y + f_h), (255, 0, 255), 2)

        cv2.putText(
            vis_img,
            f"Pose: {pose_state}",
            (f_x, max(25, f_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        # 6. Extract valid patches
        processed_patches_bgr = []
        for zone_name, zone_mask, _ in candidate_zones:
            if zone_mask is None:
                skipped_patch = np.zeros((128, 128, 3), dtype=np.uint8)
                cv2.putText(skipped_patch, f"{zone_name}:", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                cv2.putText(skipped_patch, "SKIPPED", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                processed_patches_bgr.append(skipped_patch)
                continue

            bbox = get_bbox_from_mask(zone_mask)
            if bbox is None:
                skipped_patch = np.zeros((128, 128, 3), dtype=np.uint8)
                cv2.putText(skipped_patch, f"{zone_name}:", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                cv2.putText(skipped_patch, "EMPTY", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                processed_patches_bgr.append(skipped_patch)
                continue

            x_box, y_box, w_box, h_box = bbox
            skin_pixels = np.count_nonzero(zone_mask)

            # Check unpacked dimensions and skin pixel count
            if w_box < MIN_ZONE_WIDTH or h_box < MIN_ZONE_HEIGHT or skin_pixels < MIN_SKIN_PIXELS:
                print(f"Skipping {zone_name}: Insufficient skin area ({skin_pixels} px)")
                skipped_patch = np.zeros((128, 128, 3), dtype=np.uint8)
                cv2.putText(skipped_patch, f"{zone_name}:", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
                cv2.putText(skipped_patch, "TOO SMALL", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
                processed_patches_bgr.append(skipped_patch)
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

            final_patch_lab = cv2.resize(padded_patch, (PATCH_SIZE, PATCH_SIZE))
            final_patch_bgr = cv2.cvtColor(final_patch_lab, cv2.COLOR_LAB2BGR)

            enlarged = cv2.resize(final_patch_bgr, (128, 128), interpolation=cv2.INTER_NEAREST)
            cv2.putText(enlarged, zone_name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            processed_patches_bgr.append(enlarged)

        if processed_patches_bgr:
            patch_strip = np.hstack(processed_patches_bgr)
            target_width = vis_img.shape[Y_AXIS]
            strip_h, strip_w = patch_strip.shape[:2]
            scaled_strip_h = int(strip_h * (target_width / strip_w))
            patch_strip_resized = cv2.resize(patch_strip, (target_width, scaled_strip_h))
            final_preview = np.vstack([vis_img, patch_strip_resized])
        else:
            final_preview = vis_img

        if save_preview:
            cv2.imwrite("preview_result3.jpg", final_preview)
            print("Preview saved to 'preview_result.jpg'")

        cv2.imshow("Pose-Aware Preview (Press any key to close)", final_preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sample_path = r"E:\SmartTriageAI\Train-dataset\dataset\Normal\04960.png"
    preview_face_crop(sample_path)