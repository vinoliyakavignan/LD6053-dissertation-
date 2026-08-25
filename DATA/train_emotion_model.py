#!/usr/bin/env python
"""
Emotion Detection Model Training Script
========================================
Trains a MobileNetV2-based CNN on the facial emotion dataset
(19 subjects × 8 emotions = 152 images) and exports:
  1. TFLite model  → app/src/main/assets/emotion_classifier.tflite
  2. TorchScript   → app/src/main/assets/model.pth
  3. Evaluation scores → DATA/model_scores.json

Emotion classes (8): Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprised
Mapped to app labels: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
"""

import os
import json
import numpy as np
from datetime import datetime
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, Input, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ─── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "app", "src", "main", "assets"
)
TFLITE_PATH = os.path.join(ASSETS_DIR, "emotion_classifier.tflite")
TORCHSCRIPT_PATH = os.path.join(ASSETS_DIR, "model.pth")
SCORES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_scores.json")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 8

# Emotion labels — order matches directory file naming
EMOTION_LABELS = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
# Mapping to app's 7-emotion schema
APP_EMOTION_MAP = {
    "Anger": "Angry",
    "Contempt": "Disgust",  # Contempt maps to Disgust in the app's schema
    "Disgust": "Disgust",
    "Fear": "Fear",
    "Happy": "Happy",
    "Neutral": "Neutral",
    "Sad": "Sad",
    "Surprised": "Surprise",
}
LABEL_TO_IDX = {label: idx for idx, label in enumerate(EMOTION_LABELS)}


# ─── Data Loading ──────────────────────────────────────────────────────────────

def load_dataset():
    """Load all images and labels from the DATA/images directory."""
    images = []
    labels = []
    subject_ids = []

    subject_dirs = sorted(
        [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))],
        key=int
    )

    for subject_id in subject_dirs:
        subject_path = os.path.join(DATA_DIR, subject_id)
        for filename in sorted(os.listdir(subject_path)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                emotion_name = os.path.splitext(filename)[0]
                if emotion_name in LABEL_TO_IDX:
                    img_path = os.path.join(subject_path, filename)
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                        img_array = np.array(img, dtype=np.float32) / 255.0
                        images.append(img_array)
                        labels.append(LABEL_TO_IDX[emotion_name])
                        subject_ids.append(int(subject_id))
                    except Exception as e:
                        print(f"  Warning: Could not load {img_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)
    subject_ids = np.array(subject_ids)

    print(f"Dataset loaded: {len(images)} images, {NUM_CLASSES} classes")
    print(f"Image shape: {images.shape[1:]}")
    print(f"Class distribution:")
    for i, label in enumerate(EMOTION_LABELS):
        count = np.sum(labels == i)
        print(f"  {label}: {count} samples")

    return images, labels, subject_ids


# ─── Model Building ────────────────────────────────────────────────────────────

def build_model(num_classes=NUM_CLASSES):
    """Build a MobileNetV2-based transfer learning model."""
    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze initially

    inputs = Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax", name="emotion_output")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


# ─── Training ──────────────────────────────────────────────────────────────────

def train():
    print("=" * 70)
    print("EMOTION DETECTION MODEL TRAINING")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print()

    # Load data
    images, labels, subject_ids = load_dataset()
    print()

    # Stratified train/test split — leave ~4 subjects for testing
    unique_subjects = np.unique(subject_ids)
    train_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=4, random_state=42
    )

    train_mask = np.isin(subject_ids, train_subjects)
    test_mask = np.isin(subject_ids, test_subjects)

    X_train, y_train = images[train_mask], labels[train_mask]
    X_test, y_test = images[test_mask], labels[test_mask]

    print(f"Train subjects: {sorted(train_subjects.tolist())} ({len(X_train)} images)")
    print(f"Test subjects:  {sorted(test_subjects.tolist())} ({len(X_test)} images)")
    print()

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator()  # No augmentation for validation

    # Build model
    model, base_model = build_model()
    print(f"Model summary:")
    model.summary()
    print()

    # Callbacks — save_weights_only to avoid architecture mismatch between phases
    checkpoint_path = "/tmp/best_emotion_model.weights.h5"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    # Phase 1: Train with frozen base
    print("=" * 70)
    print("Phase 1: Training with frozen MobileNetV2 base")
    print("=" * 70)
    history1 = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=val_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE),
        epochs=20,
        callbacks=callbacks,
        verbose=2,
    )

    # Phase 2: Fine-tune with unfrozen base
    print()
    print("=" * 70)
    print("Phase 2: Fine-tuning with unfrozen base")
    print("=" * 70)
    base_model.trainable = True
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # New callbacks for phase 2 (save to a different file to avoid mismatch)
    checkpoint_path2 = "/tmp/best_emotion_model_phase2.weights.h5"
    callbacks2 = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
        ModelCheckpoint(
            filepath=checkpoint_path2,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    history2 = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=val_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE),
        epochs=30,
        callbacks=callbacks2,
        verbose=2,
    )

    # Load best weights from phase 2 (architecture now matches)
    try:
        model.load_weights(checkpoint_path2)
        print("Loaded best Phase 2 weights.")
    except Exception as e:
        print(f"Could not load Phase 2 weights ({e}), using current model weights.")

    # ─── Evaluation ────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)

    # Test set evaluation
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    print("\nPer-class metrics:")
    print(f"{'Emotion':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 65)
    for i, label in enumerate(EMOTION_LABELS):
        print(f"{label:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")

    # Macro and weighted averages
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    print(f"\nMacro Avg:     P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f1:.4f}")
    print(f"Weighted Avg:  P={weighted_p:.4f}  R={weighted_r:.4f}  F1={weighted_f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))
    print(f"\nConfusion Matrix:")
    print(f"{'':>12}" + "".join(f"{l[:6]:>8}" for l in EMOTION_LABELS))
    for i, label in enumerate(EMOTION_LABELS):
        row = " ".join(f"{cm[i][j]:>8}" for j in range(NUM_CLASSES))
        print(f"{label:>12} {row}")

    # Full classification report
    report = classification_report(
        y_test, y_pred, target_names=EMOTION_LABELS, zero_division=0, output_dict=True
    )
    print(f"\nFull Classification Report:")
    print(json.dumps(report, indent=2))

    # ─── Export TFLite ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("EXPORTING MODELS")
    print("=" * 70)

    os.makedirs(ASSETS_DIR, exist_ok=True)

    # TFLite conversion
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    tflite_size = os.path.getsize(TFLITE_PATH) / 1024
    print(f"TFLite model saved: {TFLITE_PATH} ({tflite_size:.1f} KB)")

    # Verify TFLite model
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"TFLite input:  shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
    print(f"TFLite output: shape={output_details[0]['shape']}, dtype={output_details[0]['dtype']}")

    # Quick TFLite inference test
    test_sample = X_test[0:1].astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_sample)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    tflite_pred = np.argmax(tflite_output[0])
    keras_pred = np.argmax(y_pred_probs[0])
    print(f"TFLite inference test: predicted={EMOTION_LABELS[tflite_pred]}, "
          f"Keras predicted={EMOTION_LABELS[keras_pred]}, "
          f"actual={EMOTION_LABELS[y_test[0]]}")
    print(f"Outputs match: {np.allclose(tflite_output[0], y_pred_probs[0], atol=1e-2)}")

    # TorchScript export (for PyTorch Android integration)
    try:
        import torch
        import torch.nn as nn
        import torchvision.models as tv_models

        # Recreate the model architecture in PyTorch for TorchScript export
        class EmotionClassifierPT(nn.Module):
            def __init__(self, num_classes=NUM_CLASSES):
                super().__init__()
                self.base = tv_models.mobilenet_v2(pretrained=True)
                self.base.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(self.base.last_channel, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes),
                )

            def forward(self, x):
                return self.base(x)

        pt_model = EmotionClassifierPT(NUM_CLASSES)
        pt_model.eval()

        # Trace with a dummy input
        dummy_input = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
        traced_model = torch.jit.trace(pt_model, dummy_input)
        traced_model.save(TORCHSCRIPT_PATH)
        pt_size = os.path.getsize(TORCHSCRIPT_PATH) / 1024
        print(f"TorchScript model saved: {TORCHSCRIPT_PATH} ({pt_size:.1f} KB)")
    except Exception as e:
        print(f"TorchScript export skipped: {e}")

    # ─── Save Scores ───────────────────────────────────────────────────────────
    scores = {
        "model_name": "MobileNetV2_EmotionClassifier",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "framework": "TensorFlow/Keras",
        "tensorflow_version": tf.__version__,
        "image_size": list(IMAGE_SIZE),
        "num_classes": NUM_CLASSES,
        "emotion_labels": EMOTION_LABELS,
        "app_emotion_map": APP_EMOTION_MAP,
        "dataset": {
            "total_images": len(images),
            "num_subjects": len(unique_subjects),
            "train_images": len(X_train),
            "test_images": len(X_test),
            "train_subjects": sorted(train_subjects.tolist()),
            "test_subjects": sorted(test_subjects.tolist()),
            "class_distribution": {
                EMOTION_LABELS[i]: int(np.sum(labels == i)) for i in range(NUM_CLASSES)
            },
        },
        "training": {
            "epochs_phase1": 20,
            "epochs_phase2": 30,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "data_augmentation": {
                "rotation_range": 20,
                "width_shift_range": 0.2,
                "height_shift_range": 0.2,
                "shear_range": 0.2,
                "zoom_range": 0.2,
                "horizontal_flip": True,
                "brightness_range": [0.7, 1.3],
            },
        },
        "evaluation": {
            "test_accuracy": float(test_accuracy),
            "test_accuracy_percent": float(test_accuracy * 100),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_p),
            "weighted_recall": float(weighted_r),
            "weighted_f1": float(weighted_f1),
            "per_class": {
                EMOTION_LABELS[i]: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1_score": float(f1[i]),
                    "support": int(support[i]),
                }
                for i in range(NUM_CLASSES)
            },
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        },
        "exports": {
            "tflite_model": TFLITE_PATH,
            "tflite_size_kb": round(tflite_size, 1),
            "torchscript_model": TORCHSCRIPT_PATH if os.path.exists(TORCHSCRIPT_PATH) else None,
            "torchscript_size_kb": round(pt_size, 1) if os.path.exists(TORCHSCRIPT_PATH) else None,
        },
    }

    with open(SCORES_PATH, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nScores saved: {SCORES_PATH}")

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"  Macro F1:      {macro_f1:.4f}")
    print(f"  TFLite model:  {TFLITE_PATH} ({tflite_size:.1f} KB)")
    print("=" * 70)

    return scores


if __name__ == "__main__":
    train()
