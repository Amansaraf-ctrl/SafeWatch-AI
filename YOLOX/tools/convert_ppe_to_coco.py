import os
import cv2
import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

# SafeWatch AI: Strict 5-Class Target Mapping
CLASSES = ["Helmet", "No_helmet", "Safety_shoes", "No_SafetyShoes", "Person"]

# Mapping from raw Kaggle dataset YOLO class IDs to SafeWatch AI target IDs:
# 0: Helmet -> 0 (Helmet)
# 4: NO_helmet -> 1 (No_helmet)
# 3: Safety_shoes -> 2 (Safety_shoes)
# 7: No_SafetyShoes -> 3 (No_SafetyShoes)
# 8: Person -> 4 (Person)
CLASS_MAPPING = {
    0: 0,  # Helmet -> Helmet
    4: 1,  # NO_helmet -> No_helmet
    3: 2,  # Safety_shoes -> Safety_shoes
    7: 3,  # No_SafetyShoes -> No_SafetyShoes
    8: 4,  # Person -> Person
}


def get_image_size(img_path):
    """Fast image dimension retrieval with fallback."""
    img = cv2.imread(str(img_path))
    if img is not None:
        h, w, _ = img.shape
        return w, h
    return None, None


def link_or_copy(src, dst, use_hardlink=True):
    """Link file if on the same drive to save disk space and time; fallback to copy."""
    if dst.exists():
        return
    if use_hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def convert_yolo_to_coco(root_dir, out_dir, use_hardlinks=True):
    root_dir = Path(root_dir)
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations").mkdir(exist_ok=True)

    print(f"Reading dataset from: {root_dir}")
    print(f"Building COCO structure at: {out_dir}")
    print(f"Target SafeWatch AI classes: {CLASSES}")
    print(f"Class mapping: {CLASS_MAPPING}")

    splits = [("train", "train2017"), ("valid", "val2017"), ("val", "val2017"), ("test", "test2017")]
    processed_splits = set()

    for in_split, out_split in splits:
        if out_split in processed_splits:
            continue

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
            "categories": [{"id": i, "name": name} for i, name in enumerate(CLASSES)],
        }

        img_id, ann_id = 0, 0
        img_names = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        class_ann_counts = {c: 0 for c in range(len(CLASSES))}

        for img_name in tqdm(img_names, desc=f"Converting {out_split}"):
            img_path = img_dir / img_name
            w, h = get_image_size(img_path)
            if w is None or h is None:
                continue

            coco_data["images"].append(
                {
                    "id": img_id,
                    "file_name": img_name,
                    "width": w,
                    "height": h,
                }
            )

            lbl_name = img_path.stem + ".txt"
            lbl_path = lbl_dir / lbl_name

            if lbl_path.exists():
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue

                        raw_cls_id = int(parts[0])
                        # Filter and remap strictly to SafeWatch AI 5 classes
                        if raw_cls_id not in CLASS_MAPPING:
                            continue

                        target_cls_id = CLASS_MAPPING[raw_cls_id]
                        class_ann_counts[target_cls_id] += 1

                        x_c, y_c, bw, bh = map(float, parts[1:5])

                        # YOLO normalized center -> COCO absolute top-left
                        abs_bw, abs_bh = bw * w, bh * h
                        abs_x = (x_c * w) - (abs_bw / 2)
                        abs_y = (y_c * h) - (abs_bh / 2)

                        # Clamp bounding box coordinates within image bounds
                        abs_x = max(0.0, abs_x)
                        abs_y = max(0.0, abs_y)
                        abs_bw = min(float(w) - abs_x, abs_bw)
                        abs_bh = min(float(h) - abs_y, abs_bh)

                        if abs_bw <= 0 or abs_bh <= 0:
                            continue

                        coco_data["annotations"].append(
                            {
                                "id": ann_id,
                                "image_id": img_id,
                                "category_id": target_cls_id,
                                "bbox": [round(abs_x, 2), round(abs_y, 2), round(abs_bw, 2), round(abs_bh, 2)],
                                "area": round(abs_bw * abs_bh, 2),
                                "iscrowd": 0,
                            }
                        )
                        ann_id += 1

            # Hardlink or copy image into YOLOX split folder
            link_or_copy(img_path, out_img_dir / img_name, use_hardlink=use_hardlinks)
            img_id += 1

        out_json = out_dir / "annotations" / f"instances_{out_split}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(coco_data, f)

        print(f"Success: Processed {img_id} images and mapped {ann_id} annotations to {out_json}")
        print("Class distribution:")
        for idx, name in enumerate(CLASSES):
            print(f"  - [{idx}] {name}: {class_ann_counts[idx]} annotations")

        processed_splits.add(out_split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to raw Kaggle dataset")
    parser.add_argument("--out", type=str, required=True, help="Path to output COCO dataset")
    parser.add_argument("--copy-only", action="store_true", help="Force copying instead of hardlinks")
    args = parser.parse_args()
    convert_yolo_to_coco(args.root, args.out, use_hardlinks=not args.copy_only)
