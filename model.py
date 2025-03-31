import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, Dropout,
                                     Add, Bidirectional, LSTM, Dense, MultiHeadAttention,
                                     LayerNormalization, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Enable mixed-precision training (uses float16 on supported hardware)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# 1. Data Preprocessing
# -------------------------------
data_df = pd.read_csv("/kaggle/input/features7/features12.csv")

X = data_df.drop(columns=['labels']).values
y = data_df["labels"].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))

target_features = 25 * 11  # 275
num_original_features = X.shape[1]
if num_original_features < target_features:
    X = np.pad(X, ((0, 0), (0, target_features - num_original_features)), mode='constant')
elif num_original_features > target_features:
    X = X[:, :target_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

X_train = X_train.reshape(-1, 25, 11)
X_test = X_test.reshape(-1, 25, 11)

# -------------------------------
# 2. Define the Advanced Attention-based CNN-LSTM Model
# -------------------------------
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], 1),
                                 initializer="normal",
                                 trainable=True)
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = inputs * a
        return tf.keras.backend.sum(output, axis=1)

def build_advanced_model():
    inputs = Input(shape=(25, 11))

    # --- CNN Branch with Residual Connection ---
    x = Conv1D(filters=352, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization(dtype='float32')(x)
    x = Dropout(0.18)(x)  # increased dropout by 20% (0.15 -> 0.18)

    cnn_branch = Conv1D(filters=512, kernel_size=5, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    cnn_branch = BatchNormalization(dtype='float32')(cnn_branch)
    cnn_branch = Dropout(0.18)(cnn_branch)  # increased dropout by 20%

    # Project x to 512 channels if necessary.
    if x.shape[-1] != cnn_branch.shape[-1]:
        x_proj = Conv1D(filters=512, kernel_size=1, padding='same')(x)
    else:
        x_proj = x
    cnn_out = Add()([x_proj, cnn_branch])
    cnn_out = LayerNormalization()(cnn_out)

    # --- Updated Multi-Head Self-Attention over CNN features ---
    # Increase number of heads from 2 to 4 and boost key_dim by 40% (32 -> ~45)
    mha = MultiHeadAttention(num_heads=8, key_dim=128)(cnn_out, cnn_out)
    cnn_out = Add()([cnn_out, mha])
    cnn_out = LayerNormalization(name="mha_out")(cnn_out)

    # --- LSTM Branch ---
    lstm_out = Bidirectional(LSTM(512, return_sequences=True, dropout=0.12, recurrent_activation="sigmoid"))(cnn_out)
    # LSTM dropout increased by 20% (0.1 -> 0.12)

    # --- Custom Attention over LSTM outputs ---
    attn_out = AttentionLayer()(lstm_out)

    # --- Dense Layers for Feature Extraction ---
    dense_out = Dense(384, activation='relu', name='dense_layer', kernel_initializer='he_normal')(attn_out)
    dense_out = Dropout(0.54)(dense_out)  # increased dropout by 20% (0.45 -> 0.54)

    outputs = Dense(num_classes, activation='softmax', dtype='float32')(dense_out)

    optimizer = Adam(learning_rate=0.00020675)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model1 = build_advanced_model()
model1.summary()

model1_checkpoint = ModelCheckpoint('final_model1.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
history1 = model1.fit(X_train, y_train, epochs=90, batch_size=32, validation_data=(X_test, y_test), callbacks=[model1_checkpoint], verbose=1)
preds1 = model1.predict(X_test)

def build_advanced_model_1():
    inputs = Input(shape=(25, 11))

    # --- CNN Branch with Residual Connection ---
    # Original dropout: 0.15; New dropout: 0.15 + 0.05 = 0.20
    x = Conv1D(filters=352, kernel_size=3, activation='relu', padding='same',
               kernel_initializer='he_normal')(inputs)
    x = BatchNormalization(dtype='float32')(x)
    x = Dropout(0.20)(x)

    # Second convolution: original dropout: 0.15; New dropout: 0.15 + 0.10 = 0.25
    cnn_branch = Conv1D(filters=512, kernel_size=5, activation='relu', padding='same',
                        kernel_initializer='he_normal')(x)
    cnn_branch = BatchNormalization(dtype='float32')(cnn_branch)
    cnn_branch = Dropout(0.25)(cnn_branch)

    # Project x to 512 channels if necessary.
    if x.shape[-1] != cnn_branch.shape[-1]:
        x_proj = Conv1D(filters=512, kernel_size=1, padding='same')(x)
    else:
        x_proj = x
    cnn_out = Add()([x_proj, cnn_branch])
    cnn_out = LayerNormalization()(cnn_out)

    # --- Multi-Head Self-Attention over CNN features ---
    # Increase heads to 8 and key_dim from 32 to 128 (scaling accordingly)
    mha = MultiHeadAttention(num_heads=16, key_dim=256)(cnn_out, cnn_out)
    cnn_out = Add()([cnn_out, mha])
    cnn_out = LayerNormalization(name="mha_out")(cnn_out)

    # --- LSTM Branch ---
    # Original LSTM dropout: 0.10; New dropout: 0.10 + 0.15 = 0.25
    lstm_out = Bidirectional(LSTM(512, return_sequences=True, dropout=0.25,
                                  recurrent_activation="sigmoid"))(cnn_out)

    # --- Custom Attention over LSTM outputs ---
    attn_out = AttentionLayer()(lstm_out)

    # --- Stacked Dense Layers for Classification ---
    # First Dense Block:
    # Use the original dense dropout (0.45) plus an increment of 0.20 = 0.65
    dense1 = Dense(512, activation='relu', name='dense_layer_1', kernel_initializer='he_normal')(attn_out)
    dense1 = Dropout(0.50)(dense1)

    # Second Dense Block:
    # Use the original dense dropout (0.45) plus an increment of 0.25 = 0.70
    dense2 = Dense(384, activation='relu', name='dense_layer_2', kernel_initializer='he_normal')(dense1)
    dense2 = Dropout(0.60)(dense2)

    # Final output layer for classification remains the same.
    outputs = Dense(num_classes, activation='softmax', dtype='float32', name='output_layer')(dense2)

    optimizer = Adam(learning_rate=0.00020675)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train model instance 2
tf.random.set_seed(4567)
model2 = build_advanced_model_1()
model2_checkpoint = ModelCheckpoint('final_model2.keras', monitor='val_accuracy',
                                    save_best_only=True, verbose=1)
history2 = model2.fit(X_train, y_train, epochs=60, batch_size=16,
                      validation_data=(X_test, y_test), callbacks=[model2_checkpoint], verbose=1)
preds2 = model2.predict(X_test)

deep_ensemble_preds = (preds1 + preds2) / 2.0
deep_ensemble_classes = np.argmax(deep_ensemble_preds, axis=1)
deep_ensemble_acc = accuracy_score(y_test, deep_ensemble_classes)
print(f"Deep Ensemble Accuracy: {deep_ensemble_acc * 100:.2f}%")

embedding_model_dense1 = Model(inputs=model1.input, outputs=model1.get_layer("dense_layer").output)
embedding_model_dense2 = Model(inputs=model2.input, outputs=model2.get_layer("dense_layer_2").output)
embeddings_dense_train1 = embedding_model_dense1.predict(X_train)
embeddings_dense_train2 = embedding_model_dense2.predict(X_train)
embeddings_dense_test1  = embedding_model_dense1.predict(X_test)
embeddings_dense_test2  = embedding_model_dense2.predict(X_test)
embeddings_dense_train = (embeddings_dense_train1 + embeddings_dense_train2) / 2.0
embeddings_dense_test  = (embeddings_dense_test1 + embeddings_dense_test2) / 2.0

if len(embeddings_dense_train.shape) > 2:
    embeddings_dense_train = embeddings_dense_train.reshape(embeddings_dense_train.shape[0], -1)
    embeddings_dense_test = embeddings_dense_test.reshape(embeddings_dense_test.shape[0], -1)

if len(embeddings_dense_train.shape) > 2:
    embeddings_dense_train = embeddings_dense_train.reshape(embeddings_dense_train.shape[0], -1)
    embeddings_dense_test = embeddings_dense_test.reshape(embeddings_dense_test.shape[0], -1)

from sklearn.neighbors import KNeighborsClassifier

best_k = 1  # Using k=1 as per your experiments
knn_weighted = KNeighborsClassifier(n_neighbors=best_k, metric='cosine', weights='distance')
knn_weighted.fit(embeddings_dense_train, y_train)
knn_preds = knn_weighted.predict(embeddings_dense_test)
knn_acc = accuracy_score(y_test, knn_preds)
print(f"KNN Accuracy (Weighted, k={best_k}, Cosine): {knn_acc * 100:.2f}%")

knn_weight = 1.0
deep_weight = 0.0
final_probs = (deep_weight * deep_ensemble_preds) + (knn_weight * knn_weighted.predict_proba(embeddings_dense_test))
final_preds = np.argmax(final_probs, axis=1)
final_accuracy = accuracy_score(y_test, final_preds)
print(f"Final Ensemble (Deep {deep_weight*100:.0f}%, KNN {knn_weight*100:.0f}%) Accuracy: {final_accuracy * 100:.2f}%")

# -------------------------------
# 4. Save the Trained KNN Classifier & Label Encoder
# -------------------------------

# Save the KNN classifier to a .pkl file for later use.
joblib.dump(knn_weighted, 'knn_model.pkl')
print("KNN classifier saved to knn_model.pkl")

# Save the label encoder for decoding predicted labels
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Label encoder saved to label_encoder.pkl")

print(f"Type of KNN model: {type(knn_weighted)}")
print(f"Type of Label Encoder: {type(label_encoder)}")
