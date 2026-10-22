import sys
import argparse
from pathlib import Path
import cv2
from face_recognizer import FaceRecognitionSystem, DEFAULT_KNOWN_DIR

def enroll_faces(aman_img_path=None, vishesh_img_path=None):
    known_dir = Path(DEFAULT_KNOWN_DIR)
    known_dir.mkdir(parents=True, exist_ok=True)

    fr = FaceRecognitionSystem()
    print(f"\n==========================================")
    print(f"  SafeWatch AI: Target Face Enrollment")
    print(f"  Allowed Identities: Aman & Vishesh")
    print(f"==========================================\n")

    # 1. Check for Aman
    if aman_img_path:
        p = Path(aman_img_path)
    else:
        # Check standard default filenames in known_dir or root
        candidates = [
            known_dir / "aman.jpg", known_dir / "aman.png", known_dir / "aman.jpeg",
            Path(r"C:\ppe detection\aman.jpg"), Path(r"C:\ppe detection\aman.png")
        ]
        p = next((c for c in candidates if c.exists()), None)

    if p and p.exists():
        img = cv2.imread(str(p))
        if img is not None:
            ok, msg = fr.enroll_face(img, name="Aman", worker_id="EMP-01")
            print(f"[AMAN ENROLLMENT] {msg}")
        else:
            print(f"[ERROR] Could not read Aman image from {p}")
    else:
        print(f"[PENDING] Aman face photo not found yet. Place 'aman.jpg' in {known_dir} or pass --aman <path>")

    # 2. Check for Vishesh
    if vishesh_img_path:
        p = Path(vishesh_img_path)
    else:
        candidates = [
            known_dir / "vishesh.jpg", known_dir / "vishesh.png", known_dir / "vishesh.jpeg",
            Path(r"C:\ppe detection\vishesh.jpg"), Path(r"C:\ppe detection\vishesh.png")
        ]
        p = next((c for c in candidates if c.exists()), None)

    if p and p.exists():
        img = cv2.imread(str(p))
        if img is not None:
            ok, msg = fr.enroll_face(img, name="Vishesh", worker_id="EMP-02")
            print(f"[VISHESH ENROLLMENT] {msg}")
        else:
            print(f"[ERROR] Could not read Vishesh image from {p}")
    else:
        print(f"[PENDING] Vishesh face photo not found yet. Place 'vishesh.jpg' in {known_dir} or pass --vishesh <path>")

    fr.reload_enrolled_faces()
    print("\nCurrently Enrolled Workers:")
    for w in fr.list_enrolled():
        print(f" - {w['name']} ({w['filename']})")
    print("==========================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll Aman and Vishesh into SafeWatch AI Face Recognition")
    parser.add_argument("--aman", type=str, help="Path to Aman's face image")
    parser.add_argument("--vishesh", type=str, help="Path to Vishesh's face image")
    args = parser.parse_args()

    enroll_faces(args.aman, args.vishesh)
