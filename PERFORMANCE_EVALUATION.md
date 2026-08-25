# Performance Evaluation: Emotion and Object Detection

## 1. Scope

This evaluation covers the two visual AI functions used by the Android safety-monitoring application:

1. **Facial emotion classification** using `emotion_classifier.tflite`.
2. **Object analysis**, which consists of:
   - a general EfficientDet-Lite0 COCO object detector (`efficientdet_lite0.tflite`), and
   - a separate MobileNetV2 binary whole-image knife classifier (`knife_classifier.tflite`).

The distinction is important: the knife model is an **image classifier**, not an object detector. It assigns one knife/non-knife score to the complete frame and does not produce a knife bounding box.

This report uses only evidence stored in the repository. Values labelled **measured** come from saved evaluation output. Values labelled **not measured** must not be presented as experimental results.

## 2. Evaluation Metrics

The following metrics are appropriate for this safety-oriented system:

- **Accuracy:** proportion of all samples classified correctly. This can be misleading with imbalanced data.
- **Precision:** proportion of positive predictions that are correct. Low knife precision produces excessive false alarms.
- **Recall (sensitivity):** proportion of real positive cases detected. Low knife recall is particularly dangerous because threats are missed.
- **F1-score:** harmonic mean of precision and recall.
- **Macro F1:** unweighted mean of the class F1-scores; suitable for the balanced eight-class emotion task.
- **Confusion matrix:** shows which classes are confused.
- **mAP and IoU:** standard localization metrics for an object detector. They cannot be computed for the repository's knife data because it has no bounding-box annotations.
- **Latency, throughput, and stability:** on-device median/P95 inference time, frames per second, memory, and repeated-frame false-alert rate. These have not been benchmarked in the repository.

## 3. Emotion Detection

### 3.1 Model and data

The deployed v2 emotion model is a frozen ResNet50 `conv4_block6_out` feature extractor with an L2-regularized linear classification head. It accepts a cropped RGB face resized to 224 × 224 and predicts eight classes: Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, and Surprised.

The dataset contains 152 images from 19 subjects, with exactly 19 images per class. The saved holdout split uses 120 images from 15 subjects for training and 32 images from four unseen subjects for testing. Hyperparameter selection uses five-fold group cross-validation on the training subjects only. This subject-separated design is preferable to a random image split because it tests generalization to unseen people.

### 3.2 Measured results

| Evaluation | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Four-subject holdout (32 images) | 31.25% | 27.36% | 31.25% | 27.70% |
| Leave-one-subject-out (152 predictions) | 40.79% | 40.79% | 39.57% | 39.71% |
| Training-subject GroupKFold estimate | 37.50% | — | — | — |
| Random-chance baseline (8 balanced classes) | 12.50% | — | 12.50% | — |

The exported TensorFlow Lite model obtained the same holdout accuracy as the source model (31.25%), indicating that export did not change the predicted classes on the saved test set.

The v2 holdout accuracy is 21.88 percentage points higher than the earlier MobileNetV2 model (9.38%), and its macro F1 increased from 9.91% to 27.70%. Nevertheless, the absolute performance remains too low for emotion predictions to be treated as reliable safety evidence.

### 3.3 Per-class results

| Emotion | Holdout precision | Holdout recall | Holdout F1 | LOSO recall |
|---|---:|---:|---:|---:|
| Anger | 50.00% | 25.00% | 33.33% | 47.37% |
| Contempt | 0.00% | 0.00% | 0.00% | 15.79% |
| Disgust | 22.22% | 50.00% | 30.77% | 57.89% |
| Fear | 33.33% | 25.00% | 28.57% | 42.11% |
| Happy | 60.00% | 75.00% | 66.67% | 68.42% |
| Neutral | 20.00% | 25.00% | 22.22% | 42.11% |
| Sad | 0.00% | 0.00% | 0.00% | 10.53% |
| Surprised | 33.33% | 50.00% | 40.00% | 42.11% |

Happy is the strongest class. Contempt and Sad are the weakest: neither was correctly recognized in the 32-image holdout, and their LOSO recalls were only 15.79% and 10.53%. Because the holdout contains only four examples of each class, each individual error changes a class recall by 25 percentage points. The LOSO results are therefore the more stable estimate, although they are still based on a small, constrained dataset.

### 3.4 Emotion conclusion

The model performs above the balanced chance level but does not demonstrate deployment-grade accuracy. It may be used as a low-weight contextual signal, as the current threat engine does, but should not independently trigger emergency action. In particular, the weak recall for Sad, Contempt, Anger, and Fear limits the model's value for distress detection.

## 4. Object and Knife Detection

### 4.1 General COCO detector

The application loads EfficientDet-Lite0 and returns at most five detections with scores of at least 0.30. The repository does not contain a labeled COCO-style evaluation set, saved predictions, mAP, precision/recall results, or device latency measurements for this model. Therefore, **project-specific detection accuracy and mAP are not measured**.

Published benchmark figures for the base model should not be substituted for application results: camera position, lighting, motion blur, object size, model packaging, and the app's 0.30 threshold all affect performance. A project-specific test must use representative camera frames and bounding-box annotations.

### 4.2 Knife classifier dataset

| Split | Knife | Non-knife | Total | Knife proportion |
|---|---:|---:|---:|---:|
| Train | 340 | 8 | 348 | 97.70% |
| Validation | 60 | 8 | 68 | 88.24% |
| Test | 100 | 8 | 108 | 92.59% |

The split is extremely imbalanced and contains only 24 non-knife images across all three subsets. The non-knife examples are drawn from the emotion-image collection rather than from realistic confusing objects and backgrounds. Consequently, they do not adequately represent phones, tools, utensils, hands, reflective objects, clutter, or low-light scenes that could cause false alarms.

An always-knife classifier would achieve **92.59% test accuracy (100/108)** without learning any visual feature. Accuracy is therefore not a valid headline metric for this test set. Balanced accuracy, per-class recall, knife precision/recall, F1, precision-recall AUC, and the full confusion matrix are required.

The deployed TFLite model was subsequently evaluated at the app's fixed 0.50 knife threshold against all 108 repository test images. The exact Android preprocessing contract was reproduced: RGB, 224 × 224, float32 values in [0,1], with knife score calculated as `1 - model_output`.

| Knife metric | Measured result |
|---|---:|
| Precision | 92.59% |
| Recall (sensitivity) | 100.00% |
| Specificity | 0.00% |
| Balanced accuracy | 50.00% |
| F1-score | 96.15% |
| PR-AUC (average precision) | 95.09% |
| Raw accuracy | 92.59% |

Confusion matrix at threshold 0.50 (rows are actual, columns are predicted):

| Actual / predicted | Knife | Non-knife |
|---|---:|---:|
| Knife | 100 (TP) | 0 (FN) |
| Non-knife | 8 (FP) | 0 (TN) |

The model classified **every test image as knife**. Its 92.59% accuracy exactly matches the always-knife baseline. Recall and F1 appear high only because 92.59% of the test images are positive; zero specificity and 50% balanced accuracy reveal that the deployed threshold does not discriminate the two classes on this test. PR-AUC indicates some score-ranking ability, but it is inflated by the 100/8 class imbalance and does not make the 0.50 operating point usable.

Single-thread desktop CPU TFLite inference, after 20 warm-up runs and excluding image decoding/resizing, measured 10.21 ms median, 16.07 ms P90, and 18.01 ms P95 over 108 runs. This is a host benchmark, not Android-device latency. An Android instrumentation benchmark was compiled (20 warm-ups and 200 timed runs), but could not execute because no Android device or emulator was connected.

### 4.3 Functional limitations

- The classifier resizes the complete camera frame to 224 × 224, so a small or distant knife may occupy too few pixels to recognize.
- It cannot localize the knife or verify that a detected object is held by a person.
- COCO does not provide a standard knife class in this app's detector; the separate classifier is the only knife-specific path.
- The general detector and knife classifier use different thresholds (0.30 and 0.50), neither validated against a deployment dataset.
- Frame-level scores are returned directly; no temporal confirmation rule is implemented in `ObjectAnalyzer`, increasing susceptibility to transient false positives.

### 4.4 Object conclusion

The measured knife model is unusable at its current 0.50 threshold because it rejects none of the non-knife test images. Even these results are only preliminary because the test set has just eight unrepresentative negative examples. No defensible project-specific accuracy or mAP claim can currently be made for general object localization because bounding-box ground truth is absent.

## 5. Recommended Evaluation Procedure

1. Collect a frozen, deployment-representative test set that is never used for training or threshold tuning. Include different people, rooms, indoor/outdoor lighting, camera orientations, distances, partial occlusion, and motion blur.
2. For emotion recognition, use at least 50–100 examples per class from unseen subjects and report subject-level bootstrap confidence intervals.
3. For general object detection, annotate class labels and bounding boxes. Report mAP@0.50, mAP@0.50:0.95, per-class average precision, recall, and small/medium/large-object results.
4. For knife recognition, include balanced and realistic negatives such as phones, scissors, cutlery, tools, pens, hands, and empty scenes. Report TP, FP, TN, FN, precision, recall, specificity, balanced accuracy, F1, PR-AUC, and ROC-AUC.
5. Select thresholds on a validation set, then lock them before evaluating the test set. In a safety system, prioritize knife recall while enforcing an operationally acceptable false-alert rate.
6. Evaluate consecutive frames and incidents as well as individual images. Report false alerts per monitoring hour and missed incidents, since those measures reflect actual use better than frame accuracy.
7. Benchmark the release APK on the target phone after warm-up. Report median, P90, and P95 latency for face detection, emotion inference, general detection, knife inference, and the combined pipeline; also report effective FPS, peak memory, CPU load, and battery drain.
8. Repeat testing across demographic groups and environmental conditions. Emotion models in particular require fairness analysis and should never be interpreted as direct evidence of intent.

## 6. Suggested Acceptance Criteria

These are proposed engineering gates, not measured results:

| Component | Suggested minimum before safety pilot |
|---|---|
| Emotion model | Macro F1 ≥ 70% on unseen subjects; no safety-critical class recall below 65% |
| Knife model | Recall ≥ 95%, precision ≥ 90%, and specificity ≥ 95% on a representative locked test set |
| General detector | mAP@0.50 and per-critical-class recall reported; target set from actual camera conditions |
| Runtime | P95 combined inference latency compatible with ≥ 10 processed FPS on the target device |
| Operational stability | False alerts per hour and missed incident rate measured and approved for the intended setting |

The appropriate gates ultimately depend on the deployment environment and the human-response workflow. Emergency notifications should require temporal confirmation and preferably corroborating signals rather than a single-frame prediction.

## 7. Overall Assessment

The emotion pipeline has a sounder subject-separated evaluation than its earlier version and preserves accuracy after TensorFlow Lite export, but its measured macro F1 of 27.70% on the holdout and 39.71% under LOSO is not adequate for standalone safety decisions. The object pipeline is technically integrated but lacks project-specific performance evidence. Its knife dataset is so imbalanced and unrepresentative that even a high raw accuracy could conceal unsafe behavior.

The project should therefore be described as a **functional prototype with preliminary emotion measurements**, not as a validated safety-monitoring system. The next priority is a representative, locked object/knife test set with saved predictions, followed by threshold calibration and on-device latency and incident-level testing.

## 8. Evidence Used

- `DATA/model_scores_v2.json` — v2 emotion dataset, methodology, holdout, LOSO, per-class, and TFLite results.
- `DATA/model_scores.json` — earlier MobileNetV2 emotion results.
- `DATA/train_emotion_model_v2.py` — emotion preprocessing, subject grouping, model, and export procedure.
- `DATA/train_knife_classifier.py` — knife classifier architecture and training code.
- `DATA/evaluate_knife_model.py` — reproducible deployed-TFLite evaluator.
- `DATA/knife_evaluation.json` — measured knife metrics and host latency.
- `DATA/knife_test_predictions.csv` — per-image knife scores, predictions, and timing.
- `DATA/knife_dataset/` — class counts in train, validation, and test splits.
- `app/src/main/java/com/example/saftymonitoringsystem/ai/ObjectAnalyzer.kt` — deployed detector configuration and thresholds.
- `README.md` — project architecture and the classification-versus-detection limitation.
