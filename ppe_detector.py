import os
import sys
import time
import uuid
import torch
import cv2
import numpy as np
from pathlib import Path

# Add YOLOX to sys.path
YOLOX_DIR = Path(r"C:\ppe detection\YOLOX")
if str(YOLOX_DIR) not in sys.path:
    sys.path.append(str(YOLOX_DIR))

from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.data.data_augment import ValTransform

from face_recognizer import FaceRecognitionSystem
from db_logger import log_violation_to_db, SNAPSHOTS_DIR

# SafeWatch AI: Strict 5-Class Target Mapping
CLASSES = ["Helmet", "No_helmet", "Safety_shoes", "No_SafetyShoes", "Person"]

CLASS_COLORS = {
    "Helmet": (0, 220, 0),          # Green
    "No_helmet": (0, 0, 255),        # Red
    "Safety_shoes": (0, 200, 0),     # Green
    "No_SafetyShoes": (0, 140, 255), # Orange
    "Person": (255, 255, 255),       # White
}


def box_iou(box1, box2):
    """Computes Intersection over Union of two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def box_containment(child_box, parent_box):
    """Computes what fraction of child_box is inside parent_box."""
    x1 = max(child_box[0], parent_box[0])
    y1 = max(child_box[1], parent_box[1])
    x2 = min(child_box[2], parent_box[2])
    y2 = min(child_box[3], parent_box[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    child_area = max(0, child_box[2] - child_box[0]) * max(0, child_box[3] - child_box[1])
    return inter / child_area if child_area > 0 else 0.0


def detect_slippers_or_barefoot(foot_crop):
    """
    Analyzes footwear region for exposed human skin tones (toes, bridge, ankles)
    to decisively separate casual slippers, flip-flops, sandals, and bare feet
    from heavy industrial steel-toe leather boots (which have 0% exposed skin).
    """
    if foot_crop is None or foot_crop.size == 0:
        return False, 0.0

    h, w, _ = foot_crop.shape
    if h < 14 or w < 14:
        return False, 0.0

    # 1. YCrCb Skin Tone Space (Industry standard for human skin segmentation)
    ycrcb = cv2.cvtColor(foot_crop, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    skin_ycrcb = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    skin_ratio_ycrcb = float(np.mean(skin_ycrcb))

    # 2. HSV Skin Tone Space
    hsv = cv2.cvtColor(foot_crop, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    skin_hsv = (h_ch <= 25) & (s_ch >= 35) & (s_ch <= 210) & (v_ch >= 45)
    skin_ratio_hsv = float(np.mean(skin_hsv))

    # If substantial skin is exposed on foot/toes (> 7.5%), it is BARE FEET OR SLIPPERS!
    is_slipper = (skin_ratio_ycrcb > 0.075) or (skin_ratio_hsv > 0.085)
    return is_slipper, max(skin_ratio_ycrcb, skin_ratio_hsv)


def is_architectural_pillar(crop, box, score=0.5):
    """
    Filters out static vertical entrance pillars, columns, and building posts.
    """
    x1, y1, x2, y2 = box
    pw = x2 - x1
    ph = y2 - y1

    if pw <= 0 or ph <= 0:
        return True

    aspect_ratio = ph / float(pw)

    # 1. Vertical aspect ratio check: entrance columns are extremely tall and narrow
    if aspect_ratio > 3.0:
        return True

    # 2. Check for uniform vertical edges & monolithic texture across column
    if crop is not None and crop.size > 0:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h > 30 and w > 10:
            top_slice = gray[:int(h * 0.25), :]
            mid_slice = gray[int(h * 0.4):int(h * 0.6), :]
            bot_slice = gray[int(h * 0.75):, :]

            mean_top = np.mean(top_slice)
            mean_mid = np.mean(mid_slice)
            mean_bot = np.mean(bot_slice)

            # Monolithic architectural pillars have nearly identical row averages from top to bottom
            col_std = np.std([mean_top, mean_mid, mean_bot])
            if col_std < 10.0 and aspect_ratio > 2.3:
                return True

    # 3. Reject marginal detections that look like pillars
    if score < 0.55 and aspect_ratio > 2.4:
        return True

    return False


def verify_industrial_hard_hat(head_crop):
    """
    Distinguishes genuine industrial safety helmets (Hard Hats - OSHA/EN 397 standard)
    from casual headwear (baseball caps, cloth hats, beanies) and bicycle helmets.
    """
    if head_crop is None or head_crop.size == 0:
        return False, "NO_HEADWEAR"

    h, w, _ = head_crop.shape
    if h < 20 or w < 20:
        return False, "NO_HEADWEAR"

    dome = head_crop[:max(int(h * 0.65), 14), :]
    gray = cv2.cvtColor(dome, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(dome, cv2.COLOR_BGR2HSV)

    # Vent density: Bike helmets have large air vents causing extreme Laplacian variance
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var > 720:
        return False, "BIKE_HELMET"

    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # White polymer shell
    is_white_hardhat = (np.mean(s_channel) < 45) and (np.mean(v_channel) > 150)

    # Hi-vis safety colors (Yellow, Orange, Blue, Red, Green)
    yellow_mask = (h_channel >= 18) & (h_channel <= 40) & (s_channel > 60) & (v_channel > 95)
    orange_mask = (h_channel >= 7) & (h_channel < 18) & (s_channel > 70) & (v_channel > 100)
    blue_mask = (h_channel >= 90) & (h_channel <= 135) & (s_channel > 55) & (v_channel > 75)
    red_mask = ((h_channel <= 7) | (h_channel >= 170)) & (s_channel > 70) & (v_channel > 80)
    green_mask = (h_channel >= 40) & (h_channel <= 85) & (s_channel > 55) & (v_channel > 75)

    safety_color_ratio = np.mean(yellow_mask | orange_mask | blue_mask | red_mask | green_mask)

    if is_white_hardhat or safety_color_ratio > 0.28:
        return True, "HARD_HAT"

    if safety_color_ratio < 0.15 and not is_white_hardhat:
        return False, "BASEBALL_CAP"

    if lap_var < 350 and np.mean(v_channel) > 110:
        return True, "HARD_HAT"

    return False, "BASEBALL_CAP"


class SpatialWorkerTracker:
    def __init__(self, max_disappeared=75, re_alert_interval=180.0):
        self.next_id = 1
        self.tracks = {}
        self.max_disappeared = max_disappeared
        self.re_alert_interval = re_alert_interval

    def update(self, detected_boxes):
        matched = []
        unmatched_dets = list(range(len(detected_boxes)))

        for tid, track in list(self.tracks.items()):
            best_iou = 0.0
            best_idx = -1
            tb = track["bbox"]

            for idx in unmatched_dets:
                db = detected_boxes[idx]
                iou = box_iou(tb, db)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou > 0.25:
                track["bbox"] = detected_boxes[best_idx]
                track["disappeared"] = 0
                matched.append((tid, track, detected_boxes[best_idx]))
                unmatched_dets.remove(best_idx)
            else:
                track["disappeared"] += 1
                if track["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]

        for idx in unmatched_dets:
            tid = self.next_id
            self.next_id += 1
            new_track = {
                "bbox": detected_boxes[idx],
                "disappeared": 0,
                "logged": False,
                "name": "Unidentified Worker",
                "conf": 0.0,
                "last_time": 0.0,
                "face_res": None
            }
            self.tracks[tid] = new_track
            matched.append((tid, new_track, detected_boxes[idx]))

        return matched


class PPEDetector:
    def __init__(self, 
                 exp_path=r"C:\ppe detection\YOLOX\exps\custom\yolox_s_ppe.py",
                 ckpt_path=r"C:\ppe detection\YOLOX\YOLOX_outputs\yolox_s_ppe\latest_ckpt.pth",
                 conf_thre=0.25,
                 nms_thre=0.45):
        self.exp_path = exp_path
        self.ckpt_path = ckpt_path
        self.conf_thre = conf_thre
        self.nms_thre = nms_thre
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = (self.device == "cuda")
        print(f"[PPE DETECTOR] Initializing on {self.device} (FP16={self.use_fp16})")
        
        self.exp = get_exp(self.exp_path, None)
        self.model = self.exp.get_model()
        self.model.to(self.device)
        self.model.eval()
        
        if os.path.exists(self.ckpt_path):
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            print(f"[PPE DETECTOR] Loaded weights from: {self.ckpt_path}")
        else:
            print(f"[PPE DETECTOR WARNING] Checkpoint not found at {self.ckpt_path}")

        if self.use_fp16:
            self.model.half()

        self.preproc = ValTransform(legacy=False)
        self.test_size = self.exp.test_size
        
        self.face_system = FaceRecognitionSystem()
        self.tracker = SpatialWorkerTracker(max_disappeared=75, re_alert_interval=180.0)
        self.frame_count = 0

    def detect(self, frame, camera_id="Cam_01", save_snapshots=True):
        """
        Runs edge-accelerated YOLOX detection, pillar rejection, slipper/barefoot inspection,
        hard hat verification, and deduplicated snapshot capture.
        """
        self.frame_count += 1
        orig_h, orig_w, _ = frame.shape
        blob, _ = self.preproc(frame, None, self.test_size)
        blob = torch.from_numpy(blob).unsqueeze(0).to(self.device)
        blob = blob.half() if self.use_fp16 else blob.float()

        with torch.no_grad():
            outputs = self.model(blob)
            outputs = postprocess(outputs, self.exp.num_classes, self.conf_thre, self.nms_thre)

        scale = min(self.test_size[0] / orig_h, self.test_size[1] / orig_w)
        
        detected_person_boxes = []
        detected_ppe = []

        if outputs[0] is not None:
            output = outputs[0].cpu().numpy()
            for row in output:
                box = row[:4] / scale
                x1 = max(0, min(orig_w - 1, int(box[0])))
                y1 = max(0, min(orig_h - 1, int(box[1])))
                x2 = max(0, min(orig_w - 1, int(box[2])))
                y2 = max(0, min(orig_h - 1, int(box[3])))
                
                score = float(row[4] * row[5])
                cls_id = int(row[6])
                cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else "Unknown"
                
                if cls_name == "Person":
                    # --- REJECT POSTERS, T-SHIRT PRINTS & ARCHITECTURAL PILLARS ---
                    # 1. Require strong person confidence score (>= 0.44)
                    if score < 0.44:
                        continue
                    
                    pw = x2 - x1
                    ph = y2 - y1
                    # 2. Minimum human physical dimensions (at least 105px tall and 40px wide)
                    if ph < 105 or pw < 40:
                        continue
                    
                    # 3. Reject flat posters/prints (aspect < 1.15)
                    aspect = ph / float(pw)
                    if aspect < 1.15:
                        continue

                    # 4. Reject entrance pillars, columns, and structural posts
                    person_crop = frame[y1:y2, x1:x2]
                    if is_architectural_pillar(person_crop, [x1, y1, x2, y2], score=score):
                        continue
                        
                    detected_person_boxes.append([x1, y1, x2, y2])

                else:
                    if score >= 0.28:
                        detected_ppe.append({
                            "bbox": [x1, y1, x2, y2],
                            "score": round(score, 2),
                            "class_id": cls_id,
                            "class_name": cls_name
                        })

        annotated = frame.copy()
        now = time.time()
        violations_in_frame = []
        compliant_in_frame = []

        # Run spatial tracker across detected person boxes
        tracked_workers = self.tracker.update(detected_person_boxes)

        for tid, track, pbox in tracked_workers:
            px1, py1, px2, py2 = pbox
            pw = px2 - px1
            ph = py2 - py1
            
            head_box = [px1, max(0, py1 - int(ph * 0.10)), px2, py1 + int(ph * 0.45)]
            feet_box = [px1, py1 + int(ph * 0.65), px2, py2]
            
            has_raw_helmet = False
            has_no_helmet = False
            has_verified_shoes = False
            has_no_shoes_flag = False
            slipper_or_barefoot_detected = False
            raw_helmet_box = None
            
            for ppe in detected_ppe:
                p_box = ppe["bbox"]
                p_cls = ppe["class_name"]
                
                if p_cls in ("Helmet", "No_helmet"):
                    if box_containment(p_box, head_box) or box_iou(p_box, head_box) > 0.08:
                        if p_cls == "Helmet":
                            has_raw_helmet = True
                            raw_helmet_box = p_box
                        if p_cls == "No_helmet":
                            has_no_helmet = True
                        
                elif p_cls in ("Safety_shoes", "No_SafetyShoes"):
                    if box_containment(p_box, feet_box) or box_iou(p_box, feet_box) > 0.05:
                        if p_cls == "Safety_shoes":
                            # Inspect the shoes bounding box for exposed skin (slippers/barefoot)
                            sx1, sy1, sx2, sy2 = p_box
                            shoe_crop = frame[sy1:sy2, sx1:sx2]
                            is_slipper, _ = detect_slippers_or_barefoot(shoe_crop)
                            if is_slipper:
                                slipper_or_barefoot_detected = True
                            else:
                                has_verified_shoes = True

                        if p_cls == "No_SafetyShoes":
                            has_no_shoes_flag = True

            # If not detected as shoes, also inspect feet zone of the worker directly
            if not has_verified_shoes and not slipper_or_barefoot_detected:
                feet_crop = frame[py1 + int(ph * 0.70):py2, px1:px2]
                is_slipper, _ = detect_slippers_or_barefoot(feet_crop)
                if is_slipper:
                    slipper_or_barefoot_detected = True

            # --- HARD HAT VS BASEBALL CAP / BIKE HELMET VERIFICATION ---
            has_verified_hardhat = False
            headwear_type = "NO_HEADWEAR"

            if has_raw_helmet:
                hx1, hy1, hx2, hy2 = raw_helmet_box if raw_helmet_box else head_box
                hx1 = max(0, min(orig_w - 1, hx1))
                hy1 = max(0, min(orig_h - 1, hy1))
                hx2 = max(0, min(orig_w - 1, hx2))
                hy2 = max(0, min(orig_h - 1, hy2))
                
                head_crop = frame[hy1:hy2, hx1:hx2]
                is_valid_hardhat, headwear_type = verify_industrial_hard_hat(head_crop)
                has_verified_hardhat = is_valid_hardhat

            # Determine PPE Compliance / Violations
            missing_items = []

            # 1. Helmet checks
            if has_no_helmet:
                missing_items.append("No Helmet")
            elif not has_verified_hardhat:
                if headwear_type == "BASEBALL_CAP":
                    missing_items.append("Non-compliant Cap (Hard Hat Required)")
                elif headwear_type == "BIKE_HELMET":
                    missing_items.append("Non-compliant Bike Helmet (Hard Hat Required)")
                else:
                    missing_items.append("No Helmet")

            # 2. Footwear checks (Strict: slippers/barefoot are immediate critical violations)
            if slipper_or_barefoot_detected:
                missing_items.append("Slippers/Barefoot (Safety Shoes Required)")
            elif has_no_shoes_flag or not has_verified_shoes:
                # If feet are in frame (person reaches lower screen)
                if py2 > int(orig_h * 0.80):
                    missing_items.append("No Safety Shoes")

            is_violation = len(missing_items) > 0
            
            # Identity Recognition Cadence
            needs_face_id = (track["name"] in ("Unidentified Worker", "Unknown Worker")) or (self.frame_count % 8 == 0)
            if needs_face_id:
                person_crop = frame[py1:py2, px1:px2]
                face_result = self.face_system.recognize_in_crop(person_crop)
                track["name"] = face_result["name"]
                track["conf"] = face_result["confidence"]
                track["face_res"] = face_result

            worker_name = track["name"]
            face_conf = track["conf"]
            face_result = track["face_res"]

            is_target = any(k in worker_name.lower() for k in ["aman", "vishesh"])

            if is_violation:
                violations_in_frame.append({
                    "track_id": tid,
                    "worker_name": worker_name,
                    "missing_items": missing_items,
                    "bbox": [px1, py1, px2, py2],
                    "face_conf": face_conf
                })
                
                time_since_last = now - track["last_time"]
                is_fresh_incident = not track["logged"]
                re_alert_due = (track["logged"] and time_since_last > self.tracker.re_alert_interval)

                if save_snapshots and (is_fresh_incident or re_alert_due):
                    uid = uuid.uuid4().hex[:8]
                    ts_str = time.strftime("%Y%m%d_%H%M%S")
                    
                    snap_filename = f"snap_{ts_str}_{uid}.jpg"
                    face_filename = f"face_{ts_str}_{uid}.jpg"
                    
                    cv2.imwrite(str(SNAPSHOTS_DIR / snap_filename), frame)
                    
                    if face_result and face_result.get("aligned_face") is not None:
                        cv2.imwrite(str(SNAPSHOTS_DIR / face_filename), face_result["aligned_face"])
                    else:
                        person_crop = frame[py1:py2, px1:px2]
                        if person_crop.size > 0:
                            upper_crop = person_crop[:int(ph * 0.4), :]
                            if upper_crop.size > 0:
                                cv2.imwrite(str(SNAPSHOTS_DIR / face_filename), upper_crop)

                    log_violation_to_db(
                        status="VIOLATION",
                        missing_items=missing_items,
                        camera_id=camera_id,
                        worker_name=worker_name,
                        snapshot_file=snap_filename,
                        face_crop_file=face_filename,
                        confidence=face_conf
                    )
                    
                    track["logged"] = True
                    track["last_time"] = now
                    print(f"[INCIDENT LOGGED] Track #{tid} ({worker_name}) - Violation: {missing_items}")

                # Draw violation box (Red)
                color = (40, 40, 235)
                cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 3)
                
                name_tag = f"VIOLATION: {worker_name}" + (" [MATCH]" if is_target else "")
                sub_label = f"Violation: {', '.join(missing_items)}"
                
                cv2.rectangle(annotated, (px1, max(0, py1 - 44)), (px1 + 400, py1), color, -1)
                cv2.putText(annotated, name_tag, (px1 + 6, max(16, py1 - 25)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
                cv2.putText(annotated, sub_label, (px1 + 6, max(36, py1 - 6)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1)

            else:
                if track["logged"]:
                    track["logged"] = False
                    
                compliant_in_frame.append({
                    "track_id": tid,
                    "worker_name": worker_name,
                    "bbox": [px1, py1, px2, py2]
                })
                color = (60, 200, 40)
                cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 2)
                
                name_tag = f"COMPLIANT: {worker_name}" + (" [VERIFIED]" if is_target else "")
                sub_tag = "Hard Hat & Shoes Verified"
                
                cv2.rectangle(annotated, (px1, max(0, py1 - 26)), (px1 + 340, py1), color, -1)
                cv2.putText(annotated, f"{name_tag} ({sub_tag})", (px1 + 6, max(18, py1 - 7)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2)

        # Draw PPE items bounding boxes
        for ppe in detected_ppe:
            bx1, by1, bx2, by2 = ppe["bbox"]
            pcls = ppe["class_name"]
            pcolor = CLASS_COLORS.get(pcls, (255, 255, 0))
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), pcolor, 1)
            cv2.putText(annotated, f"{pcls} {ppe['score']}", (bx1, max(12, by1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, pcolor, 1)

        summary = {
            "persons_count": len(tracked_workers),
            "violations_count": len(violations_in_frame),
            "compliant_count": len(compliant_in_frame),
            "violations": violations_in_frame,
            "compliant": compliant_in_frame
        }
        return annotated, summary
