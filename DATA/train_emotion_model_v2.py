#!/usr/bin/env python
"""
Emotion Detection Model Training - v2 (corrected pipeline)
==========================================================

Replaces the training procedure in train_emotion_model.py, which was producing
a model at/below random chance. The original file is left untouched for
reference; this script is additive.

WHAT WAS WRONG IN v1 (each verified by measurement, not inspection)
-------------------------------------------------------------------
1. `ImageDataGenerator(brightness_range=[0.7,1.3])` was applied to images
   already scaled to [0,1]. Keras' brightness path converts the array to a PIL
   image, which truncates float [0,1] data to integers - nearly every pixel
   became 0. Measured training-batch mean 0.0083 vs validation-batch mean
   0.4180. The network was trained on effectively black frames and validated on
   normal ones. This alone explains the chance-level result.

2. `preprocess_input` was never called. MobileNetV2's ImageNet weights expect
   inputs in [-1,1]; the script fed [0,1]. Measured:
   preprocess_input([0, 0.5, 1.0]) = [-1.000, -0.996, -0.992], i.e. the whole
   image collapsed into a 0.8%-wide sliver of the expected input range.

3. Training used whole 3880x5184 frames squashed to 224x224. Measured face area
   = 5.6%..47.0% of frame (median 20.5%), so the face was only ~99x99 px. The
   Android app, meanwhile, crops the face with ML Kit BEFORE inference
   (FaceAnalyzer.kt -> TFLiteEmotionClassifier.cropFace), so train and serve saw
   different image distributions.

4. The TEST set was passed as `validation_data` while `EarlyStopping` and
   `ModelCheckpoint(save_best_only=True)` selected weights from it. That is
   model selection on test data.

5. ~1.2M parameters were fine-tuned on 120 images (~10^4 params per image).

WHAT v2 DOES
------------
* Crops the face the same way the app does (rect crop of the face box, then a
  non-aspect-preserving resize to 224x224), removing the train/serve mismatch.
* Applies the backbone's real preprocess_input, baked into the exported graph.
* Uses a FROZEN backbone + a regularised linear head - the variance-appropriate
  choice for n=120 with no augmentation.
* Selects the one hyper-parameter (C) with GroupKFold(5) grouped by SUBJECT,
  over the training subjects only. The test subjects are used exactly once.
* Reports Leave-One-Subject-Out over all 19 subjects as a second, more stable
  estimate (a 32-image test set has a standard error of roughly 8 points).
* Uses NO data augmentation of any kind.

DATASET SAFETY
--------------
DATA/images/ and DATA/emotions.csv are opened read-only and never written to.
All derived artefacts go to <repo>/build/ml_cache/, which is already gitignored.

DEPLOYMENT CONTRACT (unchanged, so the app keeps working)
---------------------------------------------------------
    input : float32 [1,224,224,3], RGB, [0,1]   (matches Kotlin preprocess())
    output: float32 [1,8] in MODEL_LABELS order
One deliberate change: the head emits LOGITS rather than softmax probabilities.
Kotlin already calls softmax() on the output (TFLiteEmotionClassifier.kt:85);
the old model also ended in softmax, so the app computed softmax(softmax(x)),
capping confidence near 0.14. Emitting logits makes the Kotlin softmax the only
one. argmax is unchanged, so no Kotlin change is required for this.

Requires: tensorflow, scikit-learn, opencv-python, mediapipe, numpy
Run:      python DATA/train_emotion_model_v2.py
"""

import os
import json
import warnings
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import (train_test_split, GroupKFold, LeaveOneGroupOut,
                                     cross_val_score, cross_val_predict)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             precision_recall_fscore_support)

# ─── Paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA_IMAGES = os.path.join(HERE, "images")                  # READ-ONLY
CACHE = os.path.join(REPO, "build", "ml_cache")             # derived, gitignored
ASSETS = os.path.join(REPO, "app", "src", "main", "assets")
TFLITE_OUT = os.path.join(ASSETS, "emotion_classifier.tflite")
SCORES_OUT = os.path.join(HERE, "model_scores_v2.json")
LANDMARKER = os.path.join(CACHE, "face_landmarker.task")
LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")

# ─── Configuration ─────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 8
N_TEST_SUBJECTS = 4
SPLIT_SEED = 42                       # same split as v1, so numbers are comparable
BACKBONE_LAYER = "conv4_block6_out"   # mid-level features; chosen by CV
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

EMOTION_LABELS = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
APP_EMOTION_MAP = {
    "Anger": "Angry", "Contempt": "Disgust", "Disgust": "Disgust", "Fear": "Fear",
    "Happy": "Happy", "Neutral": "Neutral", "Sad": "Sad", "Surprised": "Surprise",
}
LABEL_TO_IDX = {l: i for i, l in enumerate(EMOTION_LABELS)}


# ─── Face cropping (mirrors the app's crop behaviour) ──────────────────────────
def _ensure_landmarker():
    os.makedirs(CACHE, exist_ok=True)
    if not os.path.exists(LANDMARKER):
        import urllib.request
        print("downloading face landmarker model ...")
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER)
    return LANDMARKER


def build_face_crops():
    """Detect the face box and crop it the way FaceAnalyzer/cropFace does.

    Returns X (N,224,224,3) float32 in [0,1] RGB, y, subject_ids.
    Cached to disk so repeat runs are fast. DATA/images is only ever read.
    """
    xp = os.path.join(CACHE, "X_face.npy")
    yp = os.path.join(CACHE, "y.npy")
    sp = os.path.join(CACHE, "subjects.npy")
    if all(os.path.exists(p) for p in (xp, yp, sp)):
        print("using cached face crops:", CACHE)
        return np.load(xp), np.load(yp), np.load(sp)

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_ensure_landmarker()),
        num_faces=1, running_mode=vision.RunningMode.IMAGE,
        min_face_detection_confidence=0.25, min_face_presence_confidence=0.25)
    lmk = vision.FaceLandmarker.create_from_options(opts)

    subjects = sorted([d for d in os.listdir(DATA_IMAGES)
                       if os.path.isdir(os.path.join(DATA_IMAGES, d))], key=int)
    X, y, sid, missed = [], [], [], []
    for s in subjects:
        for fn in sorted(os.listdir(os.path.join(DATA_IMAGES, s))):
            emo = os.path.splitext(fn)[0]
            if emo not in LABEL_TO_IDX or not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            bgr = cv2.imread(os.path.join(DATA_IMAGES, s, fn), cv2.IMREAD_COLOR)
            if bgr is None:
                missed.append((s, emo, "unreadable")); continue
            H, W = bgr.shape[:2]

            sc = 1024.0 / max(H, W)
            small = cv2.resize(bgr, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
            res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                      data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            if not res.face_landmarks:
                missed.append((s, emo, "no face")); continue

            pts = np.array([[p.x * W, p.y * H] for p in res.face_landmarks[0]])
            x0, y0 = pts.min(0); x1, y1 = pts.max(0)
            a, b = max(int(x0), 0), min(int(x1), W)
            c, d = max(int(y0), 0), min(int(y1), H)
            # same as Kotlin: plain rect crop, then non-aspect-preserving resize
            rs = cv2.resize(bgr[c:d, a:b], IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            X.append(cv2.cvtColor(rs, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            y.append(LABEL_TO_IDX[emo]); sid.append(int(s))
        print(f"  cropped subject {s}", flush=True)
    lmk.close()

    X = np.asarray(X, np.float32); y = np.asarray(y, np.int64); sid = np.asarray(sid, np.int64)
    np.save(xp, X); np.save(yp, y); np.save(sp, sid)
    if missed:
        print("WARNING: faces not found for", missed)
    return X, y, sid


# ─── Model ─────────────────────────────────────────────────────────────────────
def build_feature_extractor():
    """[0,1] RGB input -> ResNet50 preprocessing -> conv4 block -> 2x2 avg grid -> L2 norm.

    Grid pooling (rather than global pooling) keeps coarse spatial layout, which
    matters here: whether the mouth or the brow moved is the discriminative cue.
    All preprocessing is inside the graph so the Kotlin side stays unchanged.
    """
    base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet",
                                          input_shape=(*IMAGE_SIZE, 3))
    base.trainable = False
    inp = tf.keras.Input((*IMAGE_SIZE, 3), name="input_0to1")
    x = tf.keras.layers.Rescaling(255.0)(inp)                      # [0,1] -> [0,255]
    x = tf.keras.layers.Lambda(lambda t: t[..., ::-1], name="rgb2bgr")(x)
    x = tf.keras.layers.Rescaling(1.0, offset=[-103.939, -116.779, -123.68])(x)
    z = tf.keras.Model(base.input, base.get_layer(BACKBONE_LAYER).output)(x)
    k = int(z.shape[1]) // 2
    z = tf.keras.layers.AveragePooling2D(pool_size=k, strides=k, padding="valid")(z)
    z = tf.keras.layers.Flatten()(z)
    z = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1), name="l2norm")(z)
    return tf.keras.Model(inp, z, name="feature_extractor")


def main():
    print("=" * 74)
    print("EMOTION MODEL TRAINING v2 (corrected)")
    print("=" * 74)
    print("timestamp  :", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("tensorflow :", tf.__version__)

    X, y, sid = build_face_crops()
    print(f"\ndataset    : {X.shape[0]} images, {NUM_CLASSES} classes, "
          f"{len(np.unique(sid))} subjects")
    print(f"class counts: {dict(zip(EMOTION_LABELS, np.bincount(y).tolist()))}")

    uniq = np.unique(sid)
    train_subjects, test_subjects = train_test_split(uniq, test_size=N_TEST_SUBJECTS,
                                                     random_state=SPLIT_SEED)
    tr, te = np.isin(sid, train_subjects), np.isin(sid, test_subjects)
    print(f"\ntrain subjects: {sorted(train_subjects.tolist())}  ({tr.sum()} images)")
    print(f"test  subjects: {sorted(test_subjects.tolist())}  ({te.sum()} images)")
    print("(subject-disjoint: no person appears in both halves)")

    fx = build_feature_extractor()
    F = fx.predict(X, batch_size=16, verbose=0)
    print(f"\nfeature dim: {F.shape[1]}  (ResNet50/{BACKBONE_LAYER}/2x2 grid/L2)")

    # ── hyper-parameter selection: TRAIN SUBJECTS ONLY ────────────────────────
    print("\nselecting C by GroupKFold(5) grouped by subject, TRAIN subjects only:")
    best_C, best_cv = None, -1.0
    for C in C_GRID:
        sc = cross_val_score(LogisticRegression(C=C, max_iter=20000), F[tr], y[tr],
                             groups=sid[tr], cv=GroupKFold(n_splits=5), scoring="accuracy")
        print(f"   C={C:<7} CV={sc.mean() * 100:5.2f}% +/-{sc.std() * 100:4.1f}")
        if sc.mean() > best_cv:
            best_C, best_cv = C, sc.mean()
    print(f"-> C={best_C} (CV {best_cv * 100:.2f}%)")

    # ── final fit, single evaluation on held-out subjects ─────────────────────
    clf = LogisticRegression(C=best_C, max_iter=20000).fit(F[tr], y[tr])
    y_pred = np.argmax(F[te] @ clf.coef_.T + clf.intercept_, axis=1)
    y_true = y[te]

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=range(NUM_CLASSES),
                                                    zero_division=0)
    mp_, mr_, mf_, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                       zero_division=0)
    wp_, wr_, wf_, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted",
                                                       zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))

    print(f"\n{'=' * 74}\nHELD-OUT TEST (subjects {sorted(test_subjects.tolist())}, n={te.sum()})\n{'=' * 74}")
    print(f"accuracy        : {acc * 100:.2f}%")
    print(f"macro P/R/F1    : {mp_ * 100:.2f}% / {mr_ * 100:.2f}% / {mf_ * 100:.2f}%")
    print(f"weighted P/R/F1 : {wp_ * 100:.2f}% / {wr_ * 100:.2f}% / {wf_ * 100:.2f}%")

    # ── LOSO: more stable estimate, config already fixed above ────────────────
    print("\nLeave-One-Subject-Out over all 19 subjects (config fixed beforehand):")
    loso_pred = cross_val_predict(LogisticRegression(C=best_C, max_iter=20000), F, y,
                                  groups=sid, cv=LeaveOneGroupOut())
    loso_acc = accuracy_score(y, loso_pred)
    lmp, lmr, lmf, _ = precision_recall_fscore_support(loso_pred, y, average="macro",
                                                       zero_division=0)
    loso_cm = confusion_matrix(y, loso_pred, labels=range(NUM_CLASSES))
    print(f"LOSO accuracy   : {loso_acc * 100:.2f}%  (n={len(y)}, 19 folds)")
    print(f"LOSO macro F1   : {lmf * 100:.2f}%")
    print("\nper-class recall (LOSO):")
    for i, e in enumerate(EMOTION_LABELS):
        print(f"   {e:<10} {loso_cm[i, i] / loso_cm[i].sum() * 100:5.1f}%")

    # ── bake the linear head into the graph and export ────────────────────────
    dense = tf.keras.layers.Dense(NUM_CLASSES, name="emotion_logits")
    full = tf.keras.Model(fx.input, dense(fx.output), name="emotion_classifier")
    dense.set_weights([clf.coef_.T.astype("float32"), clf.intercept_.astype("float32")])

    check = np.argmax(full.predict(X[te], batch_size=8, verbose=0), axis=1)
    assert (check == y_pred).all(), "baked Keras head disagrees with the fitted classifier"
    print("\nbaked head matches the fitted classifier exactly")

    conv = tf.lite.TFLiteConverter.from_keras_model(full)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    os.makedirs(ASSETS, exist_ok=True)
    with open(TFLITE_OUT, "wb") as fh:
        fh.write(conv.convert())
    size_kb = os.path.getsize(TFLITE_OUT) / 1024
    print(f"TFLite written  : {TFLITE_OUT} ({size_kb:.1f} KB)")

    itp = tf.lite.Interpreter(model_path=TFLITE_OUT)
    itp.allocate_tensors()
    ind, outd = itp.get_input_details()[0], itp.get_output_details()[0]
    out = []
    for i in range(int(te.sum())):
        itp.set_tensor(ind["index"], X[te][i:i + 1].astype(np.float32))
        itp.invoke()
        out.append(itp.get_tensor(outd["index"])[0])
    tfl_acc = accuracy_score(y_true, np.argmax(np.array(out), axis=1))
    print(f"TFLite input    : {list(ind['shape'])} {ind['dtype'].__name__}")
    print(f"TFLite output   : {list(outd['shape'])} {outd['dtype'].__name__} (logits)")
    print(f"TFLite test acc : {tfl_acc * 100:.2f}%")

    json.dump({
        "model_name": "ResNet50_conv4_grid_L2_LinearProbe",
        "version": 2,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "framework": "TensorFlow/Keras + scikit-learn linear head",
        "tensorflow_version": tf.__version__,
        "image_size": list(IMAGE_SIZE),
        "num_classes": NUM_CLASSES,
        "emotion_labels": EMOTION_LABELS,
        "app_emotion_map": APP_EMOTION_MAP,
        "output_is_logits": True,
        "augmentation_used": False,
        "backbone_layer": BACKBONE_LAYER,
        "chosen_C": best_C,
        "validation_methodology": (
            "Hyper-parameter C selected with GroupKFold(5) grouped by subject over the "
            "15 training subjects only. The 4 test subjects were evaluated exactly once. "
            "LOSO over all 19 subjects is reported as a second, more stable estimate; the "
            "configuration was fixed before LOSO ran, and no per-fold tuning occurs."),
        "dataset": {
            "total_images": int(len(y)),
            "num_subjects": int(len(uniq)),
            "train_images": int(tr.sum()),
            "test_images": int(te.sum()),
            "train_subjects": sorted(train_subjects.tolist()),
            "test_subjects": sorted(test_subjects.tolist()),
            "class_distribution": {EMOTION_LABELS[i]: int((y == i).sum())
                                   for i in range(NUM_CLASSES)},
        },
        "evaluation": {
            "cv_accuracy_train_subjects": float(best_cv),
            "test_accuracy": float(acc),
            "test_accuracy_tflite": float(tfl_acc),
            "test_accuracy_percent": float(acc * 100),
            "macro_precision": float(mp_), "macro_recall": float(mr_), "macro_f1": float(mf_),
            "weighted_precision": float(wp_), "weighted_recall": float(wr_),
            "weighted_f1": float(wf_),
            "loso_accuracy": float(loso_acc), "loso_macro_f1": float(lmf),
            "loso_macro_precision": float(lmp), "loso_macro_recall": float(lmr),
            "loso_confusion_matrix": loso_cm.tolist(),
            "loso_per_class_recall": {EMOTION_LABELS[i]: float(loso_cm[i, i] / loso_cm[i].sum())
                                      for i in range(NUM_CLASSES)},
            "per_class": {EMOTION_LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                              "f1_score": float(f1[i]), "support": int(sup[i])}
                          for i in range(NUM_CLASSES)},
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(y_true, y_pred,
                                                           target_names=EMOTION_LABELS,
                                                           zero_division=0, output_dict=True),
        },
        "exports": {"tflite_model": TFLITE_OUT, "tflite_size_kb": round(size_kb, 1)},
    }, open(SCORES_OUT, "w"), indent=2)
    print(f"scores written  : {SCORES_OUT}")
    print("\nDATA/images and DATA/emotions.csv were opened read-only and not modified.")


if __name__ == "__main__":
    main()
