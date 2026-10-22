import os
import cv2
import json
import shutil
import argparse
from pathlib import Path

# SafeWatch AI: Strict 5-Class Target Mapping
CLASSES = ["Helmet", "No_helmet", "Safety_shoes", "No_SafetyShoes", "Person"]

def convert_yolo_to_coco(root_dir, out_dir):
    root_dir = Path(root_dir)
    out_dir = Path(out_dir)
    
    # Setup YOLOX expected directory structure
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations").mkdir(exist_ok=True)

    print(f"Reading dataset from: {root_dir}")
    print(f"Building COCO structure at: {out_dir}")

    # Standardize split naming for Kaggle (train/valid/test) to YOLOX (train2017/val2017)
    splits = [('train', 'train2017'), ('valid', 'val2017'), ('val', 'val2017')]
    processed_splits = set()

    for in_split, out_split in splits:
        if out_split in processed_splits:
            continue

        # Detect Kaggle's folder structure (either /train/images or /images/train)
        img_dir = root_dir / in_split / "images"
        lbl_dir = root_dir / in_split / "labels"
        
        if not img_dir.exists():
            img_dir = root_dir / "images" / in_split
            lbl_dir = root_dir / "labels" / in_split
        
        if not img_dir.exists():
            continue
            
        print(f"\nProcessing {in_split} split -> {out_split}...")
        
        out_img_dir = out_dir / out_split
        out_img_dir.mkdir(exist_ok=True)

        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": name} for i, name in enumerate(CLASSES)]
        }

        img_id, ann_id = 0, 0

        for img_name in os.listdir(img_dir):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')): 
                continue
            
            img_path = img_dir / img_name
            img = cv2.imread(str(img_path))
            if img is None: 
                continue
            
            h, w, _ = img.shape

            coco_data["images"].append({
                "id": img_id,
                "file_name": img_name,
                "width": w,
                "height": h
            })

            lbl_name = img_path.stem + ".txt"
            lbl_path = lbl_dir / lbl_name

            if lbl_path.exists():
                with open(lbl_path, "r") as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) < 5: continue
                        
                        cls_id = int(parts[0])
                        # Filter strictly to the SafeWatch AI classes
                        if cls_id >= len(CLASSES): continue

                        x_c, y_c, bw, bh = map(float, parts[1:5])
                        
                        # YOLO to COCO math: (normalized center -> absolute top-left)
                        abs_bw, abs_bh = bw * w, bh * h
                        abs_x = (x_c * w) - (abs_bw / 2)
                        abs_y = (y_c * h) - (abs_bh / 2)

                        coco_data["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cls_id,
                            "bbox": [abs_x, abs_y, abs_bw, abs_bh],
                            "area": abs_bw * abs_bh,
                            "iscrowd": 0
                        })
                        ann_id += 1
            
            # Copy image to YOLOX folder
            shutil.copy2(img_path, out_img_dir / img_name)
            img_id += 1

        out_json = out_dir / "annotations" / f"instances_{out_split}.json"
        with open(out_json, "w") as f:
            json.dump(coco_data, f)
        
        print(f"Success: Copied {img_id} images and mapped {ann_id} annotations to {out_json}")
        processed_splits.add(out_split)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to raw Kaggle dataset")
    parser.add_argument("--out", type=str, required=True, help="Path to output COCO dataset")
    args = parser.parse_args()
    convert_yolo_to_coco(args.root, args.out)
