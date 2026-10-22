# SafeWatch AI — Autonomous PPE Compliance & Worker Face Recognition Command Center

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5%20CUDA-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLOX](https://img.shields.io/badge/Model-YOLOX--Small-red.svg)](https://github.com/Megvii-BaseDetection/YOLOX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, edge-accelerated computer vision command system designed for industrial manufacturing, construction sites, and hazardous fabrication plants. **SafeWatch AI** provides real-time detection of Personal Protective Equipment (PPE) violations combined with high-accuracy facial biometric identification to attribute safety violations to specific authorized personnel.

---

## 🎯 Target 5-Class Safety Protocol

SafeWatch AI is trained specifically on an optimized **5-Class target mapping**:

| Class ID | Target Class Name | Color Code | Enforcement Policy |
|:---:|:---|:---:|:---|
| `0` | **Helmet** | 🟩 Green | Compliant Hard Hat Verified |
| `1` | **No_helmet** | 🟥 Crimson | **CRITICAL VIOLATION**: Immediate Alert |
| `2` | **Safety_shoes** | 🟩 Green | Compliant Steel-Toe Shoes Verified |
| `3` | **No_SafetyShoes** | 🟧 Orange | **HIGH VIOLATION**: Safety Shoes Missing |
| `4` | **Person** | ⬜ Blue/White | Worker Bounding Box for Spatial Tracking |

---

## ⚡ Core Architecture

```
                                  [ CCTV Site Stream / Laptop Webcam ]
                                                  │
                                                  ▼
                                      [ YOLOX-Small (FP16 CUDA) ]
                                                  │
                      ┌───────────────────────────┴───────────────────────────┐
                      ▼                                                       ▼
            [ Worker Bounding Boxes ]                              [ PPE Detection Bounding Boxes ]
                      │                                                       │
                      └───────────────────────────┬───────────────────────────┘
                                                  ▼
                                    [ Spatial Association Engine ]
                              (Head Zone IoU / Lower Feet Containment)
                                                  │
                                   Is Violation Detected? (Yes/No)
                                                  │
                                   ┌──────────────┴──────────────┐
                                [Yes]                          [No]
                                  │                              │
                                  ▼                              ▼
                       [ OpenCV YuNet (Face Detect) ]    [ Mark Worker Compliant ]
                                  │                      (Green HUD Tag: PPE Verified)
                                  ▼
                     [ OpenCV SFace (128-d Vector) ]
                                  │
                      [ Cosine Similarity Match ]
                     (Authorized: Aman & Vishesh)
                                  │
                                  ▼
                     [ Incident Snapshot Pipeline ]
               (Crops Face & Full Incident Frame to Disk)
                                  │
                      ┌───────────┴───────────┐
                      ▼                       ▼
            [ SQLite Local Store ]   [ MongoDB Atlas Cloud ]
                      │                       │
                      └───────────┬───────────┘
                                  ▼
                   [ SafeWatch AI Web Command Center ]
               (Live MJPEG Feed, Snapshots Gallery, Lightbox)
```

---

## ✨ Key Features

1. **Cyber-Industrial Command Center Dashboard**:
   - Modern dark UI inspired by Palantir and Tesla Vision command interfaces.
   - Live stream canvas with HUD corner brackets and real-time bounding boxes.
   - Instant toggle between **Industrial Site Footage** (`.mp4`, `.avi`, `.mov`) and **Live Laptop Camera**.
   - Zero stock/foreign face artifacts on CCTV feed.

2. **Biometric Face Verification System**:
   - Integrated **OpenCV YuNet** (face detection) and **OpenCV SFace** (128-dimensional deep feature embeddings).
   - Restricted to authorized personnel directory: **Aman (EMP-01)** and **Vishesh (EMP-02)**.
   - Automatic matching with cosine similarity thresholding (`0.363`) and high-visibility cyan highlight tags.

3. **Automated Evidence Snapshot Gallery**:
   - Real-time debounced capture of violators.
   - Dual-crop architecture: high-resolution full-frame incident snapshot + aligned face crop.
   - Click-to-enlarge lightbox modal with download and resolution actions.

4. **Dual Persistence Architecture**:
   - Zero-configuration local **SQLite WAL database** for immediate offline operation.
   - Native TLS-encrypted sync with **MongoDB Atlas** for centralized compliance management.

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Amansaraf-ctrl/SafeWatch-AI.git
cd SafeWatch-AI
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create and activate Python environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install SafeWatch AI dependencies
pip install -r requirements.txt
```

### 3. Launch the Web Command Center
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 📸 Personnel Face Enrollment

To register the authorized personnel (**Aman** and **Vishesh**):

### Option A: Via Command Line
```bash
python enroll_target_faces.py --aman "path/to/aman.jpg" --vishesh "path/to/vishesh.jpg"
```

### Option B: Via Web Dashboard
1. Open the dashboard at `http://localhost:8000`.
2. Click **"Enroll Face"** in the top navigation bar.
3. Select **Aman** or **Vishesh**, upload their frontal photo, and click **Enroll Face**.

---

## 📹 Industrial Site Video Ingest

To stream your own industrial site videos:
- Drop `.mp4`, `.avi`, or `.mov` files into the `custom_footage/` folder, or
- Click the **"Upload Site Video"** button on the dashboard header.

---

## 📡 REST API Reference

| Endpoint | Method | Description |
|:---|:---:|:---|
| `/` | `GET` | Main SafeWatch AI Command Center dashboard |
| `/video_feed` | `GET` | MJPEG real-time video stream with AI detection overlays |
| `/api/stats` | `GET` | Real-time compliance rate and incident statistics |
| `/api/violations` | `GET` | Recent violations list with snapshot and face crop links |
| `/api/resolve/{id}` | `POST` | Marks an active safety incident as resolved |
| `/api/workers` | `GET` | Enrolled personnel directory |
| `/api/enroll` | `POST` | Enrolls a new worker face into SFace database |
| `/api/footages` | `GET` | Lists available industrial site footage files |
| `/api/upload_footage` | `POST` | Uploads an industrial site video to the stream engine |
| `/api/audit` | `POST` | Runs instant PPE & face detection on an uploaded still photo |

---

## 👥 Authors & Team
* **Aman** — Lead Developer & Architecture ([GitHub](https://github.com/Amansaraf-ctrl))
* **Vishesh** — Computer Vision & Safety Engineering

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
