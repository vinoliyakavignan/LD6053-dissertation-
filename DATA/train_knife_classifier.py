import os
import datetime
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "knife_dataset")
SAVED_MODEL_DIR = os.path.join(HERE, "saved_model")
TFLITE_MODEL_PATH = os.path.join(HERE, "knife_classifier.tflite")
LABELS_PATH = os.path.join(HERE, "labels.txt")
LOG_DIR = os.path.join(HERE, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
ANDROID_ASSETS_DIR = os.path.normpath(os.path.join(HERE, "..", "app", "src", "main", "assets"))

# Model settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 2
EPOCHS = 25
PATIENCE = 5

# Dataset structure expectation:
# knife_dataset/train/knife/
# knife_dataset/train/non_knife/
# knife_dataset/validation/knife/
# knife_dataset/validation/non_knife/
# knife_dataset/test/knife/
# knife_dataset/test/non_knife/

CLASS_NAMES = ["knife", "non_knife"]


def build_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.20,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "validation"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    test_generator = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    return train_generator, validation_generator, test_generator


def build_model():
    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def export_labels():
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        for label in CLASS_NAMES:
            f.write(label + "\n")


def export_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    tflite_model = converter.convert()

    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to {TFLITE_MODEL_PATH}")


def main():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Expected dataset folder not found: {DATA_DIR}")

    train_gen, val_gen, test_gen = build_data_generators()
    model = build_model()
    model.summary()

    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LABELS_PATH), exist_ok=True)
    os.makedirs(ANDROID_ASSETS_DIR, exist_ok=True)

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(SAVED_MODEL_DIR, "knife_classifier_best.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[checkpoint, early_stopping]
    )

    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

    print("Exporting saved model and TFLite model...")
    tf.saved_model.save(model, SAVED_MODEL_DIR)
    export_tflite(model)
    export_labels()

    shutil.copy(TFLITE_MODEL_PATH, os.path.join(ANDROID_ASSETS_DIR, os.path.basename(TFLITE_MODEL_PATH)))
    shutil.copy(LABELS_PATH, os.path.join(ANDROID_ASSETS_DIR, os.path.basename(LABELS_PATH)))
    print(f"Copied {TFLITE_MODEL_PATH} and {LABELS_PATH} to Android assets at {ANDROID_ASSETS_DIR}")

    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(history.history["accuracy"], label="train_accuracy")
        plt.plot(history.history["val_accuracy"], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(HERE, "accuracy_plot.png"))

        plt.figure()
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(HERE, "loss_plot.png"))
        print("Saved history plots.")
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping training history plots.")

    print("Training complete.")


if __name__ == "__main__":
    main()
