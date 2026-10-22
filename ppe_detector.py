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
    "Helmet": (0, 200, 0),         # Green
    "No_helmet": (0, 0, 255),       # Red
    "Safety_shoes": (0, 180, 0),    # Green
    "No_SafetyShoes": (0, 140, 255),# Orange
    "Person": (255, 255, 255),      # White
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
        print(f"[PPE DETECTOR] Initializing on device: {self.device}")
        
        self.exp = get_exp(self.exp_path, None)
        self.model = self.exp.get_model()
        self.model.to(self.device)
        self.model.eval()
        
        if os.path.exists(self.ckpt_path):
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            print(f"[PPE DETECTOR] Loaded weights from: {self.ckpt_path}")
        else:
            print(f"[PPE DETECTOR WARNING] Checkpoint not found at {self.ckpt_path}, model uninitialized!")

        self.preproc = ValTransform(legacy=False)
        self.test_size = self.exp.test_size
        
        # Face Recognizer integration
        self.face_system = FaceRecognitionSystem()
        
        # Debounce dictionary for snapshot throttling: {worker_name_or_box: last_timestamp}
        self.last_logged_time = {}
        self.log_cooldown = 6.0  # seconds

    def detect(self, frame, camera_id="Cam_01", save_snapshots=True):
        """
        Runs YOLOX detection and Face Recognition on a frame.
        
        Args:
            frame (np.ndarray): BGR image.
            camera_id (str): ID of camera stream.
            save_snapshots (bool): Whether to persist violation snapshots and DB logs.
            
        Returns:
            annotated_frame (np.ndarray): Visualization with high-contrast boxes and labels.
            frame_summary (dict): Detections, violations, and recognized worker details.
        """
        orig_h, orig_w, _ = frame.shape
        blob, _ = self.preproc(frame, None, self.test_size)
        blob = torch.from_numpy(blob).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(blob)
            outputs = postprocess(outputs, self.exp.num_classes, self.conf_thre, self.nms_thre)

        scale = min(self.test_size[0] / orig_h, self.test_size[1] / orig_w)
        
        detected_persons = []
        detected_ppe = []

        if outputs[0] is not None:
            output = outputs[0].cpu().numpy()
            for row in output:
                # [x1, y1, x2, y2, obj_conf, cls_conf, cls_pred]
                box = row[:4] / scale
                x1 = max(0, min(orig_w - 1, int(box[0])))
                y1 = max(0, min(orig_h - 1, int(box[1])))
                x2 = max(0, min(orig_w - 1, int(box[2])))
                y2 = max(0, min(orig_h - 1, int(box[3])))
                
                score = float(row[4] * row[5])
                cls_id = int(row[6])
                cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else "Unknown"
                
                det_obj = {
                    "bbox": [x1, y1, x2, y2],
                    "score": round(score, 2),
                    "class_id": cls_id,
                    "class_name": cls_name
                }
                
                if cls_name == "Person":
                    detected_persons.append(det_obj)
                else:
                    detected_ppe.append(det_obj)

        annotated = frame.copy()
        now = time.time()
        violations_in_frame = []
        compliant_in_frame = []

        # If persons detected, associate PPE items to each person
        for person in detected_persons:
            px1, py1, px2, py2 = person["bbox"]
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
                pbox = ppe["bbox"]
                pcls = ppe["class_name"]
                
                if pcls in ("Helmet", "No_helmet"):
                    overlap = box_containment(pbox, head_box) or box_iou(pbox, head_box) > 0.1
                    if overlap:
                        if pcls == "Helmet": has_helmet = True
                        if pcls == "No_helmet": has_no_helmet = True
                        
                elif pcls in ("Safety_shoes", "No_SafetyShoes"):
                    overlap = box_containment(pbox, feet_box) or box_iou(pbox, feet_box) > 0.05
                    if overlap:
                        if pcls == "Safety_shoes": has_shoes = True
                        if pcls == "No_SafetyShoes": has_no_shoes = True

            # Determine violations
            missing_items = []
            if has_no_helmet or not has_helmet:
                missing_items.append("No Helmet")
            if has_no_shoes:
                missing_items.append("No Safety Shoes")

            is_violation = len(missing_items) > 0
            
            # Crop person area
            person_crop = frame[py1:py2, px1:px2]
            face_result = self.face_system.recognize_in_crop(person_crop)
            worker_name = face_result["name"]
            face_conf = face_result["confidence"]

            if is_violation:
                violations_in_frame.append({
                    "worker_name": worker_name,
                    "missing_items": missing_items,
                    "bbox": [px1, py1, px2, py2],
                    "face_conf": face_conf
                })
                
                # Check snapshot cooldown
                debounce_key = f"{worker_name}_{camera_id}"
                last_time = self.last_logged_time.get(debounce_key, 0)
                
                snap_filename = None
                face_filename = None
                
                if save_snapshots and (now - last_time > self.log_cooldown):
                    uid = uuid.uuid4().hex[:8]
                    ts_str = time.strftime("%Y%m%d_%H%M%S")
                    
                    # Save full frame snapshot
                    snap_filename = f"snap_{ts_str}_{uid}.jpg"
                    cv2.imwrite(str(SNAPSHOTS_DIR / snap_filename), frame)
                    
                    # Save cropped face if available, otherwise upper person crop
                    if face_result.get("aligned_face") is not None:
                        face_filename = f"face_{ts_str}_{uid}.jpg"
                        cv2.imwrite(str(SNAPSHOTS_DIR / face_filename), face_result["aligned_face"])
                    elif person_crop.size > 0:
                        face_filename = f"face_{ts_str}_{uid}.jpg"
                        upper_crop = person_crop[:int(ph * 0.4), :]
                        if upper_crop.size > 0:
                            cv2.imwrite(str(SNAPSHOTS_DIR / face_filename), upper_crop)

                    # Persist event
                    log_violation_to_db(
                        status="VIOLATION",
                        missing_items=missing_items,
                        camera_id=camera_id,
                        worker_name=worker_name,
                        snapshot_file=snap_filename,
                        face_crop_file=face_filename,
                        confidence=face_conf
                    )
                    self.last_logged_time[debounce_key] = now

                # Draw violation styling: Red (#EF4444)
                color = (40, 40, 235)  # BGR Red
                cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 3)
                
                # Overlay tag header
                tag_label = f"VIOLATION: {worker_name}"
                sub_label = f"Missing: {', '.join(missing_items)}"
                
                # Background banner for text
                cv2.rectangle(annotated, (px1, max(0, py1 - 42)), (px1 + 320, py1), color, -1)
                cv2.putText(annotated, tag_label, (px1 + 6, max(16, py1 - 24)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(annotated, sub_label, (px1 + 6, max(36, py1 - 6)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1)

            else:
                compliant_in_frame.append({
                    "worker_name": worker_name,
                    "bbox": [px1, py1, px2, py2]
                })
                # Compliant styling: Emerald Green (#10B981)
                color = (60, 180, 40)  # BGR Green
                cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 2)
                
                tag_label = f"COMPLIANT: {worker_name}"
                cv2.rectangle(annotated, (px1, max(0, py1 - 24)), (px1 + 240, py1), color, -1)
                cv2.putText(annotated, tag_label, (px1 + 6, max(16, py1 - 6)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

        # Draw PPE items bounding boxes
        for ppe in detected_ppe:
            bx1, by1, bx2, by2 = ppe["bbox"]
            pcls = ppe["class_name"]
            pcolor = CLASS_COLORS.get(pcls, (255, 255, 0))
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), pcolor, 1)
            cv2.putText(annotated, f"{pcls} {ppe['score']}", (bx1, max(12, by1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, pcolor, 1)

        summary = {
            "persons_count": len(detected_persons),
            "violations_count": len(violations_in_frame),
            "compliant_count": len(compliant_in_frame),
            "violations": violations_in_frame,
            "compliant": compliant_in_frame
        }
        return annotated, summary
