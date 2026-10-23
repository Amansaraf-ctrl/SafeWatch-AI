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


class SpatialWorkerTracker:
    """
    Tracks workers across frames using spatial IoU.
    Prevents duplicate snapshot flood while a worker moves around in frame.
    """
    def __init__(self, max_disappeared=75, re_alert_interval=180.0):
        self.next_id = 1
        self.tracks = {}  # {track_id: {'bbox': box, 'disappeared': 0, 'logged': bool, 'name': str, 'conf': float, 'last_time': float, 'face_res': dict}}
        self.max_disappeared = max_disappeared
        self.re_alert_interval = re_alert_interval

    def update(self, detected_boxes):
        """
        Matches detected person boxes to existing tracks.
        Returns list of (track_id, track_dict, det_box).
        """
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

            # Spatial match threshold
            if best_iou > 0.25:
                track["bbox"] = detected_boxes[best_idx]
                track["disappeared"] = 0
                matched.append((tid, track, detected_boxes[best_idx]))
                unmatched_dets.remove(best_idx)
            else:
                track["disappeared"] += 1
                if track["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]

        # Allocate new tracks for entering personnel
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

        # Half precision for Ampere Tensor Cores
        if self.use_fp16:
            self.model.half()

        self.preproc = ValTransform(legacy=False)
        self.test_size = self.exp.test_size
        
        # Face Recognizer integration
        self.face_system = FaceRecognitionSystem()
        
        # Spatial Worker Tracker for single-snapshot incident deduplication
        self.tracker = SpatialWorkerTracker(max_disappeared=75, re_alert_interval=180.0)
        self.frame_count = 0

    def detect(self, frame, camera_id="Cam_01", save_snapshots=True):
        """
        Runs edge-accelerated YOLOX detection, spatial tracking, and deduplicated snapshot capture.
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
                    detected_person_boxes.append([x1, y1, x2, y2])
                else:
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
            
            if pw < 20 or ph < 30:
                continue
                
            head_box = [px1, py1, px2, py1 + int(ph * 0.45)]
            feet_box = [px1, py1 + int(ph * 0.65), px2, py2]
            
            has_helmet = False
            has_no_helmet = False
            has_shoes = False
            has_no_shoes = False
            
            for ppe in detected_ppe:
                p_box = ppe["bbox"]
                p_cls = ppe["class_name"]
                
                if p_cls in ("Helmet", "No_helmet"):
                    if box_containment(p_box, head_box) or box_iou(p_box, head_box) > 0.08:
                        if p_cls == "Helmet": has_helmet = True
                        if p_cls == "No_helmet": has_no_helmet = True
                        
                elif p_cls in ("Safety_shoes", "No_SafetyShoes"):
                    if box_containment(p_box, feet_box) or box_iou(p_box, feet_box) > 0.05:
                        if p_cls == "Safety_shoes": has_shoes = True
                        if p_cls == "No_SafetyShoes": has_no_shoes = True

            missing_items = []
            if has_no_helmet or not has_helmet:
                missing_items.append("No Helmet")
            if has_no_shoes:
                missing_items.append("No Safety Shoes")

            is_violation = len(missing_items) > 0
            
            # Identity Recognition Cadence
            needs_face_id = (track["name"] in ("Unidentified Worker", "Unknown Worker")) or (self.frame_count % 10 == 0)
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
                
                # --- DEDUPLICATED SNAPSHOT ENGINE ---
                # A snapshot is ONLY created if:
                # 1. This worker has NOT logged a snapshot yet for this incident, OR
                # 2. More than 180 seconds have elapsed for an ongoing persistent violation.
                time_since_last = now - track["last_time"]
                is_fresh_incident = not track["logged"]
                re_alert_due = (track["logged"] and time_since_last > self.tracker.re_alert_interval)

                if save_snapshots and (is_fresh_incident or re_alert_due):
                    uid = uuid.uuid4().hex[:8]
                    ts_str = time.strftime("%Y%m%d_%H%M%S")
                    
                    snap_filename = f"snap_{ts_str}_{uid}.jpg"
                    face_filename = f"face_{ts_str}_{uid}.jpg"
                    
                    # Save single full incident snapshot
                    cv2.imwrite(str(SNAPSHOTS_DIR / snap_filename), frame)
                    
                    # Save cropped face
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
                    print(f"[INCIDENT LOGGED] Track #{tid} ({worker_name}) - Missing: {missing_items}")

                # Draw violation box (Red)
                color = (40, 40, 235)
                cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 3)
                
                name_tag = f"VIOLATION: {worker_name}" + (" [MATCH]" if is_target else "")
                sub_label = f"Missing: {', '.join(missing_items)}"
                
                cv2.rectangle(annotated, (px1, max(0, py1 - 44)), (px1 + 340, py1), color, -1)
                cv2.putText(annotated, name_tag, (px1 + 6, max(16, py1 - 25)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
                cv2.putText(annotated, sub_label, (px1 + 6, max(36, py1 - 6)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1)

            else:
                # If worker has resolved their violation while on camera, reset logged flag
                if track["logged"]:
                    track["logged"] = False
                    
                compliant_in_frame.append({
                    "track_id": tid,
                    "worker_name": worker_name,
                    "bbox": [px1, py1, px2, py2]
                })
                # Compliant box (Green)
                color = (60, 200, 40)
                cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 2)
                
                name_tag = f"COMPLIANT: {worker_name}" + (" [VERIFIED]" if is_target else "")
                cv2.rectangle(annotated, (px1, max(0, py1 - 26)), (px1 + 280, py1), color, -1)
                cv2.putText(annotated, name_tag, (px1 + 6, max(18, py1 - 7)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

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
