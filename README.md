# Hand Gesture Recognition — PRODIGY_ML_04

## Problem Statement
Develop a hand gesture recognition model that can accurately identify and classify different hand gestures from image data, enabling intuitive human-computer interaction and gesture-based control systems.

## Dataset
[LeapGestRecog](https://www.kaggle.com/gti-upm/leapgestrecog) — near-infrared images captured via a Leap Motion sensor.
- 10 subjects, each performing 10 distinct gestures
- ~20,000 grayscale images total
- Classes: palm, l, fist, fist_moved, thumb, index, ok, palm_moved, c, down

## Approach

### Preprocessing
- Converted all images to grayscale (already near-IR/grayscale in source)
- Resized to 64×64 for computational efficiency
- Normalized pixel values to [0, 1]
- Stratified train/validation/test split (70/15/15)

### Model Architecture
A CNN built with TensorFlow/Keras:
- 3 convolutional blocks (32 → 64 → 128 filters), each followed by max pooling
- Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(10, softmax)
- Optimizer: Adam, Loss: categorical cross-entropy

### Training
- Trained for up to 20 epochs with early stopping (patience=5) on validation loss
- Model checkpointing to save the best-performing weights

## Results
- Validation accuracy: **~99.8%**
- Training converged within the first 2 epochs, with a stable plateau afterward and no meaningful train/validation gap (see training curves below)

![Accuracy and Loss Curves](accuracy_loss_curves.png)

*(Add your confusion matrix / classification report screenshot here if included in repo)*

## Files
- `gesture_recognition.ipynb` — full notebook: data loading, model training, evaluation
- `models/best_model.keras` — trained model weights
- `confusion_matrix.png` — per-class performance breakdown (if included)

## Limitations & Notes
- Train/val/test split was random (stratified by class), not split by subject — so the reported accuracy reflects performance on unseen *images* of subjects the model has already seen in training, not entirely unseen individuals. A subject-wise split would give a stricter measure of generalization to new users.
- The dataset consists of near-infrared images with a controlled dark background. A model trained on this data will likely need additional preprocessing (background subtraction, lighting normalization, or fine-tuning on real webcam samples) before it generalizes well to raw RGB webcam footage in a real-world gesture-control application.

## How to Run
1. Download the dataset from Kaggle (`gti-upm/leapgestrecog`)
2. Open `gesture_recognition.ipynb` in Google Colab or Jupyter
3. Run all cells top to bottom
