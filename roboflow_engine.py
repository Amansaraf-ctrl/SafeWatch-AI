import os
import io
import time
import json
import base64
import zipfile
import requests
import cv2
import numpy as np
from pathlib import Path

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "g7qCc2fPeBfayBK9z3Mc")
DEFAULT_WORKSPACE = "aman-saraf-zldzj"
BASE_URL = "https://api.roboflow.com"
INFER_URL = "https://serverless.roboflow.com"

SNAPSHOTS_DIR = Path(r"C:\ppe detection\snapshots")
ROBOFLOW_DATA_DIR = Path(r"C:\ppe detection\custom_dataset\roboflow")
ROBOFLOW_DATA_DIR.mkdir(parents=True, exist_ok=True)


class RoboflowEngine:
    def __init__(self, api_key=ROBOFLOW_API_KEY, workspace=DEFAULT_WORKSPACE):
        self.api_key = api_key
        self.workspace = workspace
        self._cached_overview = None
        self._cache_time = 0.0

    def get_status(self, force_refresh=False):
        """Authenticates and retrieves high-level workspace metrics."""
        now = time.time()
        if not force_refresh and self._cached_overview and (now - self._cache_time < 30):
            return self._cached_overview

        try:
            url = f"{BASE_URL}/{self.workspace}?api_key={self.api_key}"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return {
                    "connected": False,
                    "error": f"HTTP {resp.status_code}: {resp.text[:120]}",
                    "workspace": self.workspace
                }

            data = resp.json().get("workspace", {})
            projects = data.get("projects", [])

            project_summaries = []
            for p in projects:
                pid = p.get("id", "").split("/")[-1]
                classes = p.get("classes", {})
                versions = p.get("versions", 0)
                project_summaries.append({
                    "id": pid,
                    "full_id": p.get("id"),
                    "name": p.get("name"),
                    "type": p.get("type"),
                    "images": p.get("images", 0),
                    "unannotated": p.get("unannotated", 0),
                    "versions_count": versions,
                    "classes": classes,
                    "splits": p.get("splits", {}),
                    "has_model": versions > 0 and p.get("images", 0) > 0
                })

            result = {
                "connected": True,
                "workspace": self.workspace,
                "workspace_name": data.get("name", "Amans Workspace"),
                "members": data.get("members", 1),
                "projects_count": len(projects),
                "projects": project_summaries,
                "last_synced": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            self._cached_overview = result
            self._cache_time = now
            return result

        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "workspace": self.workspace
            }

    def get_project_versions(self, project_id):
        """Retrieves versions and trained serverless models for a project."""
        clean_id = project_id.split("/")[-1]
        try:
            url = f"{BASE_URL}/{self.workspace}/{clean_id}?api_key={self.api_key}"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return {"success": False, "error": f"HTTP {resp.status_code}"}

            proj_data = resp.json().get("project", {})
            num_versions = proj_data.get("versions", 0)
            
            versions = []
            for v_idx in range(1, num_versions + 1):
                v_url = f"{BASE_URL}/{self.workspace}/{clean_id}/{v_idx}?api_key={self.api_key}"
                v_resp = requests.get(v_url, timeout=8)
                if v_resp.status_code == 200:
                    v_info = v_resp.json().get("version", {})
                    m_info = v_info.get("model", {})
                    versions.append({
                        "version": v_idx,
                        "name": v_info.get("name", f"v{v_idx}"),
                        "images": v_info.get("images", 0),
                        "model_endpoint": m_info.get("endpoint"),
                        "has_model": bool(m_info.get("endpoint"))
                    })

            return {
                "success": True,
                "project_id": clean_id,
                "name": proj_data.get("name"),
                "versions": versions
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def upload_snapshot(self, filename, project_id="safety-shoes-detection-5qgkg-nhiun", split="train"):
        """Uploads a recorded violation snapshot to Roboflow for active annotation."""
        clean_id = project_id.split("/")[-1]
        snap_path = SNAPSHOTS_DIR / filename
        if not snap_path.exists():
            return {"success": False, "error": f"Snapshot file not found: {filename}"}

        try:
            with open(snap_path, "rb") as f:
                img_bytes = f.read()

            b64_str = base64.b64encode(img_bytes).decode("ascii")
            url = f"{BASE_URL}/dataset/{clean_id}/upload?api_key={self.api_key}&name={filename}&split={split}"
            resp = requests.post(
                url, 
                data=b64_str, 
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15
            )

            if resp.status_code == 200:
                res_json = resp.json()
                return {
                    "success": True,
                    "message": f"Successfully uploaded {filename} to Roboflow project '{clean_id}'",
                    "roboflow_id": res_json.get("id"),
                    "project_id": clean_id
                }
            else:
                return {
                    "success": False,
                    "error": f"Roboflow API HTTP {resp.status_code}: {resp.text}"
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def predict_cloud(self, frame_bgr, project_id="safety-shoes-detection-5qgkg-nhiun", version=3, confidence=0.35):
        """Runs fast auxiliary inference using Roboflow hosted serverless model."""
        clean_id = project_id.split("/")[-1]
        try:
            ret, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                return {"success": False, "error": "Frame compression failed"}

            b64_str = base64.b64encode(buf.tobytes()).decode("ascii")
            url = f"{INFER_URL}/{clean_id}/{version}?api_key={self.api_key}&confidence={confidence}"
            
            t0 = time.time()
            resp = requests.post(
                url,
                data=b64_str,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10
            )
            elapsed_ms = round((time.time() - t0) * 1000, 1)

            if resp.status_code == 200:
                data = resp.json()
                preds = data.get("predictions", [])
                return {
                    "success": True,
                    "predictions": preds,
                    "inference_ms": elapsed_ms,
                    "predictions_count": len(preds),
                    "model_id": data.get("resolved_model", {}).get("model_id")
                }
            else:
                return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:150]}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def sync_dataset_to_local(self, project_id, version=None):
        """
        Downloads a dataset version from Roboflow in YOLO format and unpacks it
        directly into SafeWatch AI's local training directory.
        """
        clean_id = project_id.split("/")[-1]
        v = version or 1
        try:
            export_url = f"{BASE_URL}/{self.workspace}/{clean_id}/{v}/yolov5pytorch?api_key={self.api_key}"
            resp = requests.get(export_url, timeout=12)
            if resp.status_code != 200:
                return {"success": False, "error": f"Failed to get export URL: HTTP {resp.status_code}"}

            exp_data = resp.json().get("export", {})
            download_link = exp_data.get("link")
            if not download_link:
                return {"success": False, "error": "No download link provided by Roboflow for this version."}

            print(f"[ROBOFLOW SYNC] Downloading dataset from: {download_link}")
            zip_resp = requests.get(download_link, stream=True, timeout=60)
            if zip_resp.status_code != 200:
                return {"success": False, "error": f"Failed to download zip: HTTP {zip_resp.status_code}"}

            target_dir = ROBOFLOW_DATA_DIR / f"{clean_id}_v{v}"
            target_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as z:
                z.extractall(target_dir)

            image_count = len(list(target_dir.glob("**/*.jpg"))) + len(list(target_dir.glob("**/*.png")))
            print(f"[ROBOFLOW SYNC] Successfully extracted {image_count} images to {target_dir}")

            return {
                "success": True,
                "project_id": clean_id,
                "version": v,
                "extracted_path": str(target_dir),
                "images_synced": image_count,
                "message": f"Successfully synced {image_count} images from Roboflow '{clean_id}' (v{v}) to local training storage!"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


# Global instance singleton
global_roboflow = RoboflowEngine()
