import os
import cv2
import json
import numpy as np
from pathlib import Path

DEFAULT_DET_MODEL = r"C:\ppe detection\models\face_recognition\face_detection_yunet.onnx"
DEFAULT_REC_MODEL = r"C:\ppe detection\models\face_recognition\face_recognition_sface.onnx"
DEFAULT_KNOWN_DIR = r"C:\ppe detection\known_faces"

# Cosine similarity threshold for SFace (recommended threshold: 0.363)
COSINE_THRESHOLD = 0.363


class FaceRecognitionSystem:
    def __init__(self, 
                 det_model_path=DEFAULT_DET_MODEL, 
                 rec_model_path=DEFAULT_REC_MODEL, 
                 known_dir=DEFAULT_KNOWN_DIR,
                 threshold=COSINE_THRESHOLD):
        self.det_model_path = det_model_path
        self.rec_model_path = rec_model_path
        self.known_dir = Path(known_dir)
        self.threshold = threshold
        
        self.known_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize detector with dummy size (will be updated dynamically per image)
        self.detector = cv2.FaceDetectorYN.create(
            self.det_model_path, 
            "", 
            (320, 320), 
            score_threshold=0.5, 
            nms_threshold=0.3, 
            top_k=5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(self.rec_model_path, "")
        
        # Enrolled faces cache: list of dicts: {"name": str, "id": str, "feature": np.ndarray, "img_path": str}
        self.enrolled_faces = []
        self.reload_enrolled_faces()

    def reload_enrolled_faces(self):
        """Scans the known_faces directory and computes embeddings for all registered faces."""
        self.enrolled_faces = []
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        
        for file in self.known_dir.iterdir():
            if file.suffix.lower() in valid_exts:
                # Name format: Name_Surname_ID or Name
                stem = file.stem.replace("_", " ")
                img = cv2.imread(str(file))
                if img is None:
                    continue
                
                h, w, _ = img.shape
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(img)
                
                if faces is not None and len(faces) > 0:
                    # Pick largest face
                    best_face = max(faces, key=lambda f: f[2] * f[3])
                    aligned = self.recognizer.alignCrop(img, best_face)
                    feat = self.recognizer.feature(aligned)
                    
                    self.enrolled_faces.append({
                        "name": stem,
                        "filename": file.name,
                        "feature": feat,
                        "img_path": str(file)
                    })
                    print(f"[FACE ENROLL] Enrolled: {stem} from {file.name}")
                else:
                    print(f"[FACE ENROLL WARNING] No face detected in enrolled image: {file.name}")
                    
        print(f"[FACE SYSTEM] Total enrolled workers: {len(self.enrolled_faces)}")

    def enroll_face(self, image_np, name, worker_id=""):
        """Enrolls a new worker face directly from image data."""
        clean_name = f"{name}_{worker_id}".strip("_").replace(" ", "_")
        filename = f"{clean_name}.jpg"
        save_path = self.known_dir / filename
        
        h, w, _ = image_np.shape
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image_np)
        
        if faces is None or len(faces) == 0:
            return False, "No face detected in the provided image. Please provide a clear frontal photo."
            
        cv2.imwrite(str(save_path), image_np)
        best_face = max(faces, key=lambda f: f[2] * f[3])
        aligned = self.recognizer.alignCrop(image_np, best_face)
        feat = self.recognizer.feature(aligned)
        
        # Remove previous entry with same filename or name to prevent duplicates
        display_name = f"{name} ({worker_id})" if worker_id else name
        self.enrolled_faces = [e for e in self.enrolled_faces if e["filename"] != filename and e["name"].lower() != display_name.lower()]
        
        self.enrolled_faces.append({
            "name": display_name,
            "filename": filename,
            "feature": feat,
            "img_path": str(save_path)
        })
        print(f"[FACE ENROLL] Enrolled: {display_name} ({filename}) - Total enrolled: {len(self.enrolled_faces)}")
        return True, f"Successfully enrolled worker: {display_name}"

    def delete_face(self, filename):
        """Removes a registered worker face from disk and updates the in-memory embeddings."""
        clean_name = Path(filename).name
        target_path = self.known_dir / clean_name
        if target_path.exists():
            target_path.unlink()
            print(f"[FACE SYSTEM] Deleted worker file: {clean_name}")
        self.reload_enrolled_faces()
        return True, f"Successfully removed {clean_name}"

    def has_face(self, crop):
        """
        Ultra-fast face presence verification (YuNet detection only).
        Avoids alignment and SFace feature extraction overhead.
        """
        if crop is None or crop.size == 0:
            return False
        h, w = crop.shape[:2]
        if h < 16 or w < 16:
            return False
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(crop)
        return faces is not None and len(faces) > 0

    def recognize_in_crop(self, person_crop):
        """
        Detects and recognizes the face inside a cropped Person bounding box.
        
        Returns:
            dict: {
                "name": str ("Unknown Worker" if unregistered or no match),
                "confidence": float (similarity score or 0.0),
                "face_box": list [x, y, w, h] relative to person_crop, or None,
                "aligned_face": np.ndarray or None
            }
        """
        if person_crop is None or person_crop.size == 0:
            return {"name": "Unknown Worker", "confidence": 0.0, "face_box": None, "aligned_face": None}
            
        h, w, _ = person_crop.shape
        # Focus especially on upper 60% of person crop for face detection
        upper_h = max(int(h * 0.6), 50)
        search_region = person_crop[:upper_h, :]
        
        sh, sw, _ = search_region.shape
        self.detector.setInputSize((sw, sh))
        _, faces = self.detector.detect(search_region)
        
        if faces is None or len(faces) == 0:
            # Fallback: try whole person crop
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(person_crop)
            source_img = person_crop
        else:
            source_img = search_region
            
        if faces is None or len(faces) == 0:
            return {"name": "Unidentified Worker", "confidence": 0.0, "face_box": None, "aligned_face": None}
            
        # Select highest-confidence / largest face
        best_face = max(faces, key=lambda f: f[2] * f[3] * f[-1])
        face_box = [int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])]
        
        aligned = self.recognizer.alignCrop(source_img, best_face)
        query_feat = self.recognizer.feature(aligned)
        
        # Compare with enrolled faces
        best_match_name = "Unknown Worker"
        best_score = 0.0
        
        for enrolled in self.enrolled_faces:
            try:
                score = float(self.recognizer.match(query_feat, enrolled["feature"], cv2.FaceRecognizerSF_FR_COSINE))
            except Exception:
                v1 = query_feat.flatten()
                v2 = enrolled["feature"].flatten()
                denom = np.linalg.norm(v1) * np.linalg.norm(v2)
                score = float(np.dot(v1, v2) / denom) if denom > 0 else 0.0

            if score > best_score:
                best_score = score
                if score >= self.threshold:
                    best_match_name = enrolled["name"]
                    
        return {
            "name": best_match_name,
            "confidence": round(float(best_score), 3),
            "face_box": face_box,
            "aligned_face": aligned
        }

    def list_enrolled(self):
        """Returns metadata for all registered workers."""
        return [{"name": e["name"], "filename": e["filename"]} for e in self.enrolled_faces]
