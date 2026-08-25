#!/usr/bin/env python
"""
Bottle & Cup Object Detection Training
======================================
Trains a YOLOv8 model on the bottle/cup dataset (Pascal VOC format).
Exports to ONNX and TensorFlow.js formats for web deployment.

Dataset: DATA/bottles/images/images/ (100 images)
Annotations: DATA/bottles/annotation/Bottles and Cups anotated/ (100 XML files)
Classes: bottle, cup
"""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from ultralytics import YOLO

# Paths
HERE = Path(__file__).parent
DATA_ROOT = HERE / "bottles"
IMAGES_DIR = DATA_ROOT / "images" / "images"
ANNOTATIONS_DIR = DATA_ROOT / "annotation" / "Bottles and Cups anotated"
OUTPUT_DIR = HERE / "bottle_cup_model"
YOLO_DATA_DIR = OUTPUT_DIR / "yolo_dataset"

CLASSES = ["bottle", "cup"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def convert_voc_to_yolo():
    """Convert Pascal VOC XML annotations to YOLO format (txt files)."""
    print("Converting VOC annotations to YOLO format...")
    
    # Create YOLO dataset structure
    for split in ["train", "val"]:
        (YOLO_DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"Found {len(image_files)} images")
    
    # Split: 80% train, 20% val
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    def process_split(files, split_name):
        for img_path in files:
            # Copy image
            dst_img = YOLO_DATA_DIR / "images" / split_name / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Convert annotation
            xml_name = img_path.stem + ".xml"
            xml_path = ANNOTATIONS_DIR / xml_name
            
            if not xml_path.exists():
                print(f"  Warning: No annotation for {img_path.name}")
                continue
            
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            size = root.find("size")
            img_w = int(size.find("width").text)
            img_h = int(size.find("height").text)
            
            yolo_labels = []
            for obj in root.findall("object"):
                class_name = obj.find("name").text.lower()
                if class_name not in CLASS_TO_IDX:
                    continue
                
                class_id = CLASS_TO_IDX[class_name]
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                
                # Convert to YOLO format (normalized center x, y, width, height)
                x_center = (xmin + xmax) / 2.0 / img_w
                y_center = (ymin + ymax) / 2.0 / img_h
                width = (xmax - xmin) / img_w
                height = (ymax - ymin) / img_h
                
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Write label file
            label_path = YOLO_DATA_DIR / "labels" / split_name / (img_path.stem + ".txt")
            label_path.write_text("\n".join(yolo_labels))
    
    process_split(train_files, "train")
    process_split(val_files, "val")
    
    # Create data.yaml
    data_yaml = f"""path: {YOLO_DATA_DIR.absolute()}
train: images/train
val: images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""
    (YOLO_DATA_DIR / "data.yaml").write_text(data_yaml)
    print(f"Created {YOLO_DATA_DIR}/data.yaml")
    print(f"Train: {len(train_files)} images, Val: {len(val_files)} images")


def train_model():
    """Train YOLOv8 model."""
    print("\nTraining YOLOv8n model...")
    
    model = YOLO("yolov8n.pt")  # Load nano model for speed
    
    results = model.train(
        data=str(YOLO_DATA_DIR / "data.yaml"),
        epochs=50,
        imgsz=640,
        batch=16,
        device="cpu",  # Change to "0" for GPU
        project=str(OUTPUT_DIR),
        name="train",
        exist_ok=True,
        patience=10,
        save=True,
        plots=True,
    )
    
    return model


def export_models(model):
    """Export to ONNX and TensorFlow.js formats."""
    print("\nExporting models...")
    
    # Export to ONNX
    onnx_path = model.export(format="onnx", imgsz=640, simplify=True)
    print(f"ONNX model: {onnx_path}")
    
    # Export to TensorFlow.js (requires tensorflowjs)
    try:
        tfjs_path = model.export(format="tfjs", imgsz=640)
        print(f"TF.js model: {tfjs_path}")
    except Exception as e:
        print(f"TF.js export failed (optional): {e}")
    
    # Copy best model to assets for Android if needed
    best_pt = OUTPUT_DIR / "train" / "weights" / "best.pt"
    if best_pt.exists():
        android_assets = HERE.parent / "app" / "src" / "main" / "assets"
        android_assets.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, android_assets / "bottle_cup_detector.pt")
        print(f"Copied to Android assets: {android_assets / 'bottle_cup_detector.pt'}")


def main():
    print("=" * 60)
    print("BOTTLE & CUP DETECTOR TRAINING")
    print("=" * 60)
    
    # Check dependencies
    try:
        import ultralytics
        print(f"ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("Installing ultralytics...")
        os.system("pip install ultralytics")
        import ultralytics
    
    # Convert annotations
    convert_voc_to_yolo()
    
    # Train
    model = train_model()
    
    # Export
    export_models(model)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model: {OUTPUT_DIR}/train/weights/best.pt")
    print(f"ONNX model: {OUTPUT_DIR}/train/weights/best.onnx")
    print(f"TF.js model: {OUTPUT_DIR}/train/weights/best_web_model/")
    print("\nTo use in web app:")
    print("1. Copy TF.js model to web/ directory")
    print("2. Update app.js to load custom model instead of COCO-SSD")
    print("3. Or simply add 'bottle' and 'cup' to criticalObjects in app.js (COCO-SSD already detects them)")


if __name__ == "__main__":
    main()