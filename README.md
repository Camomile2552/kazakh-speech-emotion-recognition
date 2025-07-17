# kazakh-speech-emotion-recognition
A deep learning pipeline for recognizing emotions from Kazakh speech using MFCC features and CNN architecture. The model achieves ~87% training accuracy and ~72% validation accuracy across six emotion classes, leveraging audio augmentation and feature extraction techniques for robust classification.


# 🗣️ Kazakh Speech Emotion Recognition (KazSER)

This project builds a deep learning pipeline for recognizing **emotions in Kazakh speech** using MFCC-based audio features and Convolutional Neural Networks (CNN). It includes preprocessing, augmentation, feature extraction, model training, evaluation, and visualization.

---

## Dataset

- Custom dataset containing labeled **Kazakh language** video and audio files.
- Emotion labels include:
  - Neutral
  - Happy
  - Sad
  - Angry
  - Fear
  - Disgust
- Input files: `.mp3` audio files
- Label source: `.csv` file with `FileName` and `Emotion`

---

## Data Preprocessing

- Converted video to audio (if needed)
- Applied **augmentation techniques**:
  - Noise addition
  - Time stretching
  - Pitch shifting
  - Speed adjustment
  - Shifting

- Extracted features:
  - **MFCC (Mel Frequency Cepstral Coefficients)**

---

## Model Architecture

A 1D CNN-based model with multiple convolutional and pooling layers:

- Input shape: `(n_mfcc, 1)`
- Layers:
  - Conv1D + ReLU
  - AveragePooling1D
  - Dropout
  - MaxPooling1D
  - Fully Connected Dense layer
- Final layer uses **softmax** for 6-class classification

---

## ⚙Training

- Optimizer: `Adam`
- Loss function: `categorical_crossentropy`
- Batch size: `16`
- Epochs: `50`
- Learning rate: reduced dynamically using `ReduceLROnPlateau`
- Validation split: `30%` of data, further split into validation and test

---

## Results

- **Training Accuracy**: ~87%
- **Validation Accuracy**: ~72%
- Plotted training vs. validation loss and accuracy

---

## Evaluation

- Visualized **Confusion Matrix** with Seaborn
- Generated **Classification Report** with:
  - Precision
  - Recall
  - F1-Score

---

## Model Export

- Saved trained model architecture in `model.json`
- Saved model weights in `weights.h5`

---

## Visualizations

- Waveplots and Spectrograms of example audio
- Emotion distribution histograms
- Training vs. validation curves
- Confusion matrix

---

## Technologies Used

- Python
- Librosa
- NumPy, Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn, Plotly
