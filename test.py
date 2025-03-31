import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Assuming y_test contains ground truth labels and deep_ensemble_classes are the predicted labels.
# If you have a label encoder, get the class names:
class_names = label_encoder.classes_

# Compute the confusion matrix
cm = confusion_matrix(y_test, deep_ensemble_classes)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Optionally, print the detailed classification report
print(classification_report(y_test, deep_ensemble_classes, target_names=class_names))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Assuming `y_test` contains true labels and `preds_knn` contains KNN predictions
# If you have a label encoder, get the class names:
class_names = label_encoder.classes_

# Compute the confusion matrix
cm_knn = confusion_matrix(y_test, knn_preds)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8,6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('KNN Confusion Matrix')
plt.show()

# Print detailed classification report
print(classification_report(y_test,knn_preds, target_names=class_names))

import numpy as np
from sklearn.metrics import confusion_matrix

# Get confusion matrix
cm = confusion_matrix(y_test, knn_preds)

# Compute class-wise accuracy
class_wise_accuracy = cm.diagonal() / cm.sum(axis=1)

# Print class-wise accuracy
emotion_labels = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print("\nClass-wise Emotion Accuracy:\n")
for i, label in enumerate(emotion_labels):
    print(f"{label.capitalize()}: {class_wise_accuracy[i]:.2%}")

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Example arrays; replace these with your actual true and predicted labels
# y_true = np.array([...])
# y_pred = np.array([...])
# For example:
# y_true = np.array([0, 1, 2, 2, 1, 0, 2, 1, 0])
# y_pred = np.array([0, 1, 2, 1, 1, 0, 2, 0, 0])

# Compute the confusion matrix
cm = confusion_matrix(y_test, knn_preds)

# Get number of classes and assume class names are provided by your label encoder
num_classes = cm.shape[0]
class_names = label_encoder.classes_  # For instance: ['angry', 'calm', 'disgust', ...]

# Initialize dictionaries to store FP and FN for each class
fp_dict = {}
fn_dict = {}

# Loop through each class (i represents the class index)
for i in range(num_classes):
    TP = cm[i, i]
    FP = np.sum(cm[:, i]) - TP  # Sum of the column, excluding TP
    FN = np.sum(cm[i, :]) - TP  # Sum of the row, excluding TP
    fp_dict[class_names[i]] = FP
    fn_dict[class_names[i]] = FN

# Print the results
print("False Positives per class:")
for cls, fp in fp_dict.items():
    print(f"{cls.capitalize()}: {fp}")

print("\nFalse Negatives per class:")
for cls, fn in fn_dict.items():
    print(f"{cls.capitalize()}: {fn}")

import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import joblib  # used for loading the .pkl file

def extract_features(data, sample_rate):
    """
    Extracts a comprehensive set of audio features from a given audio signal.
    The function computes:
      - Zero Crossing Rate (ZCR)
      - Chroma STFT
      - MFCCs (mean over time)
      - Delta MFCCs
      - Delta-Delta MFCCs
      - Root Mean Square Energy (RMS)
      - Mel-scaled spectrogram (mean over time)
      - Spectral Contrast
      - Tonnetz (from harmonic component)

    Returns:
      A 1D numpy array containing the concatenated features.
    """
    result = np.array([])

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # Chroma STFT
    stft = np.abs(librosa.stft(data))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma))

    # MFCCs (using 40 coefficients)
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    result = np.hstack((result, mfcc_mean))

    # Delta MFCCs
    delta_mfcc = np.mean(librosa.feature.delta(mfcc).T, axis=0)
    result = np.hstack((result, delta_mfcc))

    # Delta-Delta MFCCs
    delta2_mfcc = np.mean(librosa.feature.delta(mfcc, order=2).T, axis=0)
    result = np.hstack((result, delta2_mfcc))

    # Root Mean Square Energy
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # Mel Spectrogram (log-scaled)
    mel = librosa.feature.melspectrogram(y=data, sr=sample_rate)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_mean = np.mean(mel_db.T, axis=0)
    result = np.hstack((result, mel_mean))

    # Spectral Contrast
    spec_contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, spec_contrast))

    # Tonnetz (from the harmonic component)
    y_harmonic = librosa.effects.harmonic(data)
    tonnetz = np.mean(librosa.feature.tonnetz(y=y_harmonic, sr=sample_rate).T, axis=0)
    result = np.hstack((result, tonnetz))

    return result

def get_features(path):
    """
    Loads an audio file with a fixed duration and offset, then extracts the feature vector.
    """
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    features = extract_features(data, sample_rate)
    return features

audio_file_path = "/kaggle/input/cremad/AudioWAV/1001_IEO_ANG_HI.wav"

features = get_features(audio_file_path)
target_features = 25 * 11  # 275
num_original_features = features.shape[0]
if num_original_features < target_features:
    features = np.pad(features, (0, target_features - num_original_features), mode='constant')
elif num_original_features > target_features:
    features = features[:target_features]

# Reshape into (1, 25, 11) to match the CNN-LSTM input.
X_test_single = features.reshape(1, 25, 11)

X_test_single = features.reshape(1, 25, 11)

# -------------------------------
# 4. Load Saved Models and KNN Classifier
# -------------------------------
custom_objects = {'AttentionLayer': AttentionLayer}

# Load the saved deep models.
model1 = load_model('final_model1.keras', custom_objects=custom_objects, compile=False)
model2 = load_model('final_model2.keras', custom_objects=custom_objects, compile=False)

# Load the saved KNN classifier from a .pkl file.
knn_weighted = joblib.load('knn_model.pkl')

# -------------------------------
# 5. Generate Predictions from the Deep Ensemble
# -------------------------------
preds1 = model1.predict(X_test_single)
preds2 = model2.predict(X_test_single)
deep_ensemble_preds = (preds1 + preds2) / 2.0

# -------------------------------
# 6. Extract Embeddings for KNN Prediction
# -------------------------------
# Get the intermediate "dense_layer" outputs for the KNN classifier.
embedding_model_dense1 = Model(inputs=model1.input, outputs=model1.get_layer("dense_layer").output)
embedding_model_dense2 = Model(inputs=model2.input, outputs=model2.get_layer("dense_layer_2").output)

embeddings_test1 = embedding_model_dense1.predict(X_test_single)
embeddings_test2 = embedding_model_dense2.predict(X_test_single)
embeddings_test = (embeddings_test1 + embeddings_test2) / 2.0

# Flatten embeddings if necessary.
if len(embeddings_test.shape) > 2:
    embeddings_test = embeddings_test.reshape(embeddings_test.shape[0], -1)

# Load the saved LabelEncoder
label_encoder = joblib.load('label_encoder.pkl')

# Convert predicted class index to the actual emotion label
predicted_emotion = label_encoder.inverse_transform([final_preds[0]])[0]

print("Predicted Emotion:", predicted_emotion)

from sklearn.metrics.pairwise import cosine_distances
import numpy as np

# Get stored training embeddings from the KNN model
knn_training_embeddings = knn_weighted._fit_X  # Extract stored embeddings
knn_training_labels = knn_weighted._y          # Corresponding emotion labels

# Compute the cosine distances between test embedding and all stored embeddings
cosine_distances_all = cosine_distances(embeddings_test, knn_training_embeddings)[0]

# Decode one-hot encoded labels back to emotion names
emotion_labels = label_encoder.inverse_transform(knn_training_labels)

# Map distances to emotions
emotion_distances = {}
for label, distance in zip(emotion_labels, cosine_distances_all):
    if label not in emotion_distances:
        emotion_distances[label] = []
    emotion_distances[label].append(distance)

# Compute the average distance for each emotion
avg_emotion_distances = {emotion: np.mean(distances) for emotion, distances in emotion_distances.items()}

# Display results
print("\nCosine Distances from Each Emotion:")
for emotion, distance in avg_emotion_distances.items():
    print(f"Emotion {emotion}: Distance = {distance:.4f}")
