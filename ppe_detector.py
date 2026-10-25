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


def evaluate_footwear_certainty(crop, score, cls_name):
    """
    Enforces strict footwear certainty policy:
    'if you are unable to classify them into safety shoes or no safety shoes,
    just do not detect them; only detect them when you are certain.'

    Returns:
        (verified_class, confidence_score) or (None, 0.0) if uncertain.
    """
    if crop is None or crop.size == 0:
        return None, 0.0
    h, w, _ = crop.shape
    if h < 14 or w < 14:
        return None, 0.0

    # 1. YCrCb Skin Tone Space
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    skin_ycrcb = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    skin_ratio_ycrcb = float(np.mean(skin_ycrcb))

    # 2. HSV Skin Tone Space
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    skin_hsv = (h_ch <= 25) & (s_ch >= 35) & (s_ch <= 210) & (v_ch >= 45)
    skin_ratio_hsv = float(np.mean(skin_hsv))

    skin_ratio = max(skin_ratio_ycrcb, skin_ratio_hsv)

    # Decisive Slippers / Flip-Flops / Barefoot:
    # If substantial skin is exposed on the foot (>= 12%), it CANNOT be a safety shoe!
    if skin_ratio >= 0.12:
        return "No_SafetyShoes", round(max(score, skin_ratio), 2)

    # Decisive Industrial Safety Shoes (Steel-toe leather/rubber boots):
    # Must have high model confidence (>= 0.58) AND virtually zero exposed skin (< 3.5%)
    if cls_name == "Safety_shoes" and score >= 0.58 and skin_ratio < 0.035:
        return "Safety_shoes", round(score, 2)

    # Decisive Explicit No_SafetyShoes:
    # Must have high confidence (>= 0.55)
    if cls_name == "No_SafetyShoes" and score >= 0.55:
        return "No_SafetyShoes", round(score, 2)

    # If unable to classify with certainty, DO NOT DETECT!
    return None, 0.0


def is_architectural_pillar(crop, box, orig_h, score=0.5):
    """
    Distinguishes architectural columns, entrance pillars, and static posts
    from living human workers.
    """
    x1, y1, x2, y2 = box
    pw = x2 - x1
    ph = y2 - y1

    if pw <= 0 or ph <= 0:
        return True

    aspect_ratio = ph / float(pw)

    # 1. Structural pillars attached to top ceiling / entrance frame (y1 <= 6% frame height)
    if y1 <= int(orig_h * 0.06) and aspect_ratio > 1.8 and score < 0.65:
        return True

    # 2. Check vertical texture: architectural columns have continuous vertical Sobel lines
    if crop is not None and crop.size > 0 and aspect_ratio > 2.0:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h > 30 and w > 10:
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            mean_sx = np.mean(np.abs(sobelx))
            mean_sy = np.mean(np.abs(sobely))
            vert_edge_ratio = mean_sx / max(mean_sy, 1e-5)
            # Pillars have dominant parallel vertical boundaries and flat horizontal cross-section
            if vert_edge_ratio > 1.40 and score < 0.58:
                return True

    # 3. Static extreme aspect ratio column with modest score
    if aspect_ratio > 3.6:
        return True

    return False


def verify_industrial_hard_hat(head_crop, yolo_score=0.5):
    """
    Distinguishes genuine industrial safety helmets (Hard Hats - OSHA/EN 397 standard)
    from casual headwear (baseball caps, cloth hats, beanies) and bicycle helmets.
    """
    if head_crop is None or head_crop.size == 0:
        return False, "NO_HEADWEAR"

    h, w, _ = head_crop.shape
    if h < 16 or w < 16:
        return False, "NO_HEADWEAR"

    # If YOLOX model predicted Helmet with strong confidence (>= 0.48), trust it as genuine Hard Hat
    if yolo_score >= 0.48:
        return True, "HARD_HAT"

    dome = head_crop[:max(int(h * 0.70), 12), :]
    gray = cv2.cvtColor(dome, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(dome, cv2.COLOR_BGR2HSV)

    # Vent density: Bike helmets have large air vents causing extreme Laplacian variance
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var > 950:
        return False, "BIKE_HELMET"

    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # White polymer shell
    is_white_hardhat = (np.mean(s_channel) < 55) and (np.mean(v_channel) > 125)

    # Hi-vis safety colors (Yellow, Orange, Blue, Red, Green)
    h_channel = hsv[:, :, 0]
    yellow_mask = (h_channel >= 15) & (h_channel <= 42) & (s_channel > 45) & (v_channel > 75)
    orange_mask = (h_channel >= 6) & (h_channel < 18) & (s_channel > 55) & (v_channel > 80)
    blue_mask = (h_channel >= 88) & (h_channel <= 138) & (s_channel > 40) & (v_channel > 60)
    red_mask = ((h_channel <= 8) | (h_channel >= 168)) & (s_channel > 55) & (v_channel > 65)
    green_mask = (h_channel >= 40) & (h_channel <= 85) & (s_channel > 40) & (v_channel > 60)

    safety_color_ratio = np.mean(yellow_mask | orange_mask | blue_mask | red_mask | green_mask)

    if is_white_hardhat or safety_color_ratio > 0.16:
        return True, "HARD_HAT"

    if yolo_score >= 0.38 and lap_var < 550:
        return True, "HARD_HAT"

    return False, "BASEBALL_CAP"


def is_valid_human_subject(person_crop, head_crop, face_system, score=0.5, has_helmet=False):
    """
    Distinguishes genuine living human workers from non-human objects:
    hanging jackets, coats on coat racks, pillows on couches, chairs, cushions, curtains.
    
    A valid human subject must exhibit:
    1. Human physical aspect ratio (height >= 1.25 * width)
    2. A human face (detected by YuNet) OR
    3. An industrial safety helmet OR
    4. Human skin tones in the head/neck area OR
    5. High model confidence (score >= 0.72) with human proportions.
    """
    if person_crop is None or person_crop.size == 0:
        return False

    ph, pw = person_crop.shape[:2]
    if ph < 80 or pw < 25:
        return False

    aspect = ph / float(pw)
    # Reject horizontal or square objects (pillows, cushions, folded jackets)
    if aspect < 1.22:
        return False

    if has_helmet:
        return True

    # 1. Face detection in upper region
    if head_crop is not None and head_crop.size > 0:
        hh, hw = head_crop.shape[:2]
        if hh >= 20 and hw >= 20:
            face_result = face_system.recognize_in_crop(head_crop)
            if face_result and face_result.get("face_box") is not None:
                return True

            # 2. Skin tone analysis in head/neck
            ycrcb = cv2.cvtColor(head_crop, cv2.COLOR_BGR2YCrCb)
            cr = ycrcb[:, :, 1]
            cb = ycrcb[:, :, 2]
            skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
            skin_ratio = float(np.mean(skin_mask))
            if skin_ratio >= 0.035:
                return True

    # 3. High model confidence with standing/walking human proportions
    if score >= 0.72 and 1.35 <= aspect <= 3.8:
        return True

    return False


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
                "face_res": None,
                # Anti-flicker temporal state
                "helmet_state": "UNCERTAIN",
                "consecutive_helmet": 0,
                "consecutive_no_helmet": 0,
                "violation_confirmed": False,
                "violation_frames": 0,
                "recent_votes": []
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
        
        raw_person_candidates = []
        raw_ppe_candidates = []

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
                    pw = x2 - x1
                    ph = y2 - y1
                    if score >= 0.48 and ph >= 80 and pw >= 25:
                        raw_person_candidates.append([x1, y1, x2, y2, score])
                else:
                    # Raw candidate PPE items (Helmet, No_helmet, Safety_shoes, No_SafetyShoes)
                    if score >= 0.28:
                        raw_ppe_candidates.append({
                            "bbox": [x1, y1, x2, y2],
                            "score": round(score, 2),
                            "class_id": cls_id,
                            "class_name": cls_name
                        })

        # --- STEP 1: VALIDATE LIVING HUMAN WORKERS (REJECT COATS, JACKETS, PILLOWS) ---
        detected_person_boxes = []
        for cand in raw_person_candidates:
            cx1, cy1, cx2, cy2, c_score = cand
            c_crop = frame[cy1:cy2, cx1:cx2]
            head_h = max(int((cy2 - cy1) * 0.35), 18)
            h_crop = frame[cy1:min(orig_h, cy1 + head_h), cx1:cx2]

            # Check if there is an overlapping Helmet candidate in head region
            has_helmet_cand = any(
                p["class_name"] == "Helmet" and p["score"] >= 0.38 and
                (box_containment(p["bbox"], [cx1, cy1, cx2, cy1 + head_h]) > 0.20 or
                 box_iou(p["bbox"], [cx1, cy1, cx2, cy1 + head_h]) > 0.08)
                for p in raw_ppe_candidates
            )

            # Check architectural pillar or static vertical column
            if is_architectural_pillar(c_crop, [cx1, cy1, cx2, cy2], orig_h, score=c_score):
                continue

            # Verify genuine living human worker (eliminates jackets, coats on racks, pillows)
            if not is_valid_human_subject(c_crop, h_crop, self.face_system, score=c_score, has_helmet=has_helmet_cand):
                continue

            detected_person_boxes.append([cx1, cy1, cx2, cy2])

        annotated = frame.copy()
        now = time.time()
        violations_in_frame = []
        compliant_in_frame = []
        final_ppe_to_draw = []

        # Run spatial tracker across validated human workers
        tracked_workers = self.tracker.update(detected_person_boxes)

        for tid, track, pbox in tracked_workers:
            px1, py1, px2, py2 = pbox
            pw = px2 - px1
            ph = py2 - py1
            
            head_box = [px1, max(0, py1 - int(ph * 0.12)), px2, py1 + int(ph * 0.38)]
            feet_box = [px1, py1 + int(ph * 0.65), px2, py2]

            # --- STEP 2: HEADWEAR ANALYSIS WITH MUTUAL EXCLUSION & HELMET BIAS ---
            best_helmet = None
            best_helmet_score = 0.0
            best_no_helmet = None
            best_no_helmet_score = 0.0

            for ppe in raw_ppe_candidates:
                p_box = ppe["bbox"]
                p_cls = ppe["class_name"]
                p_score = ppe["score"]

                if p_cls in ("Helmet", "No_helmet"):
                    overlap = box_containment(p_box, head_box) > 0.22 or box_iou(p_box, head_box) > 0.07
                    if overlap:
                        if p_cls == "Helmet" and p_score > best_helmet_score:
                            best_helmet = ppe
                            best_helmet_score = p_score
                        elif p_cls == "No_helmet" and p_score > best_no_helmet_score:
                            best_no_helmet = ppe
                            best_no_helmet_score = p_score

            # Single-frame verdict on headwear:
            frame_verdict = "UNCERTAIN"
            matched_head_ppe = None

            if best_helmet is not None and best_no_helmet is not None:
                # Both detected on head: Helmet takes priority unless No_helmet is overwhelmingly dominant
                if best_helmet_score >= best_no_helmet_score - 0.15:
                    frame_verdict = "HELMET"
                    matched_head_ppe = best_helmet
                elif best_no_helmet_score >= 0.65 and best_helmet_score < 0.35:
                    frame_verdict = "NO_HELMET"
                    matched_head_ppe = best_no_helmet
                else:
                    frame_verdict = "HELMET"
                    matched_head_ppe = best_helmet

            elif best_helmet is not None:
                if best_helmet_score >= 0.35:
                    frame_verdict = "HELMET"
                    matched_head_ppe = best_helmet

            elif best_no_helmet is not None:
                # Explicit No_helmet requires score >= 0.48 AND confirmed face or skin tone presence
                head_crop = frame[max(0, head_box[1]):min(orig_h, head_box[3]), max(0, head_box[0]):min(orig_w, head_box[2])]
                has_skin_or_face = False
                if head_crop.size > 0:
                    ycrcb = cv2.cvtColor(head_crop, cv2.COLOR_BGR2YCrCb)
                    skin_cr = ycrcb[:, :, 1]
                    skin_cb = ycrcb[:, :, 2]
                    skin_mask = (skin_cr >= 133) & (skin_cr <= 173) & (skin_cb >= 77) & (skin_cb <= 127)
                    if float(np.mean(skin_mask)) >= 0.035:
                        has_skin_or_face = True
                    elif track.get("face_res") and track["face_res"].get("face_box") is not None:
                        has_skin_or_face = True

                if best_no_helmet_score >= 0.48 and has_skin_or_face:
                    frame_verdict = "NO_HELMET"
                    matched_head_ppe = best_no_helmet

            # --- STEP 3: ANTI-FLICKER TEMPORAL DEBOUNCE ENGINE ---
            if "recent_votes" not in track:
                track["recent_votes"] = []
            track["recent_votes"].append(frame_verdict)
            if len(track["recent_votes"]) > 15:
                track["recent_votes"].pop(0)

            if frame_verdict == "HELMET":
                track["consecutive_helmet"] += 1
                track["consecutive_no_helmet"] = 0
                if track["consecutive_helmet"] >= 3:
                    track["helmet_state"] = "HELMET"
                    track["violation_confirmed"] = False
                    track["violation_frames"] = 0

            elif frame_verdict == "NO_HELMET":
                track["consecutive_no_helmet"] += 1
                track["consecutive_helmet"] = 0

                # Anti-flicker: If the person was confirmed wearing a helmet,
                # do NOT switch to NO_HELMET on momentary jitter!
                # Require at least 8 consecutive frames of No_helmet and 65%+ recent votes!
                if track["helmet_state"] == "HELMET":
                    recent_no_pct = track["recent_votes"].count("NO_HELMET") / max(len(track["recent_votes"]), 1)
                    if track["consecutive_no_helmet"] >= 8 and recent_no_pct >= 0.65:
                        track["helmet_state"] = "NO_HELMET"
                        track["violation_confirmed"] = True
                else:
                    if track["consecutive_no_helmet"] >= 4:
                        track["helmet_state"] = "NO_HELMET"
                        track["violation_confirmed"] = True

            elif frame_verdict == "UNCERTAIN":
                # Maintain confirmed HELMET status! No resetting or flickering!
                pass

            # --- STEP 4: FOOTWEAR EVALUATION (STRICT CERTAINTY POLICY) ---
            has_verified_shoes = False
            has_no_shoes_flag = False
            matched_foot_ppe = None

            for ppe in raw_ppe_candidates:
                p_box = ppe["bbox"]
                p_cls = ppe["class_name"]
                p_score = ppe["score"]

                if p_cls in ("Safety_shoes", "No_SafetyShoes"):
                    if box_containment(p_box, feet_box) > 0.20 or box_iou(p_box, feet_box) > 0.05:
                        fcrop = frame[max(0, p_box[1]):min(orig_h, p_box[3]), max(0, p_box[0]):min(orig_w, p_box[2])]
                        v_cls, v_score = evaluate_footwear_certainty(fcrop, p_score, p_cls)
                        if v_cls == "Safety_shoes":
                            has_verified_shoes = True
                            matched_foot_ppe = {"bbox": p_box, "score": v_score, "class_name": "Safety_shoes"}
                        elif v_cls == "No_SafetyShoes":
                            has_no_shoes_flag = True
                            matched_foot_ppe = {"bbox": p_box, "score": v_score, "class_name": "No_SafetyShoes"}

            # Auxiliary check on bottom feet region for exposed skin (slippers/barefoot)
            if not has_no_shoes_flag and not has_verified_shoes:
                feet_crop = frame[py1 + int(ph * 0.88):py2, px1:px2]
                if feet_crop.size > 0:
                    ycrcb = cv2.cvtColor(feet_crop, cv2.COLOR_BGR2YCrCb)
                    cr = ycrcb[:, :, 1]
                    cb = ycrcb[:, :, 2]
                    skin_cr = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
                    hsv = cv2.cvtColor(feet_crop, cv2.COLOR_BGR2HSV)
                    skin_h = (hsv[:, :, 0] <= 25) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 1] <= 210) & (hsv[:, :, 2] >= 45)
                    feet_skin_ratio = max(float(np.mean(skin_cr)), float(np.mean(skin_h)))
                    if feet_skin_ratio >= 0.14:
                        has_no_shoes_flag = True
                        s_box = [px1, py1 + int(ph * 0.85), px2, py2]
                        matched_foot_ppe = {"bbox": s_box, "score": round(feet_skin_ratio, 2), "class_name": "No_SafetyShoes"}

            # --- STEP 5: COMPLIANCE & VIOLATION ASSESSMENT ---
            missing_items = []
            if track["helmet_state"] == "NO_HELMET" and track["violation_confirmed"]:
                missing_items.append("No Helmet")

            if has_no_shoes_flag:
                missing_items.append("No Safety Shoes (Slippers/Casual Footwear)")

            is_violation = len(missing_items) > 0

            # Attach verified PPE items to draw list
            if track["helmet_state"] == "HELMET" and matched_head_ppe and matched_head_ppe["class_name"] == "Helmet":
                final_ppe_to_draw.append(matched_head_ppe)
            elif track["helmet_state"] == "NO_HELMET" and track["violation_confirmed"] and matched_head_ppe and matched_head_ppe["class_name"] == "No_helmet":
                final_ppe_to_draw.append(matched_head_ppe)

            if matched_foot_ppe:
                final_ppe_to_draw.append(matched_foot_ppe)

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
            is_target = worker_name not in ("Unidentified Worker", "Unknown Worker", "Unknown", "")

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

                # Only save snapshot if confirmed sustained violation
                if save_snapshots and track["violation_confirmed"] and (is_fresh_incident or re_alert_due):
                    uid = uuid.uuid4().hex[:8]
                    ts_str = time.strftime("%Y%m%d_%H%M%S")
                    
                    snap_filename = f"snap_{ts_str}_{uid}.jpg"
                    face_filename = f"face_{ts_str}_{uid}.jpg"
                    
                    # Save pristine high-resolution evidence snapshot (98% JPEG quality for deep zoom inspection)
                    cv2.imwrite(str(SNAPSHOTS_DIR / snap_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 98])
                    
                    if face_result and face_result.get("aligned_face") is not None:
                        cv2.imwrite(str(SNAPSHOTS_DIR / face_filename), face_result["aligned_face"], [cv2.IMWRITE_JPEG_QUALITY, 98])
                    else:
                        person_crop = frame[py1:py2, px1:px2]
                        if person_crop.size > 0:
                            upper_crop = person_crop[:int(ph * 0.4), :]
                            if upper_crop.size > 0:
                                cv2.imwrite(str(SNAPSHOTS_DIR / face_filename), upper_crop, [cv2.IMWRITE_JPEG_QUALITY, 98])

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

                # Draw violation box (Solid Red)
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
                if track.get("consecutive_helmet", 0) >= 5:
                    track["logged"] = False
                    
                compliant_in_frame.append({
                    "track_id": tid,
                    "worker_name": worker_name,
                    "bbox": [px1, py1, px2, py2]
                })
                # Draw compliant box (Solid Green)
                color = (60, 200, 40)
                cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 2)
                
                name_tag = f"COMPLIANT: {worker_name}" + (" [VERIFIED]" if is_target else "")
                sub_tag = "Hard Hat & Shoes Verified" if has_verified_shoes else "Hard Hat Verified"
                
                cv2.rectangle(annotated, (px1, max(0, py1 - 26)), (px1 + 340, py1), color, -1)
                cv2.putText(annotated, f"{name_tag} ({sub_tag})", (px1 + 6, max(18, py1 - 7)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2)

        # Draw ONLY validated PPE items attached to confirmed workers (NEVER orphaned objects)
        for ppe in final_ppe_to_draw:
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
