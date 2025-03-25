import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
features_pkl = "features_v2.pkl"
df = pd.read_pickle(features_pkl)  # Read the .pkl file directly as a DataFrame

# Debug: Inspect the loaded DataFrame
print(f"Loaded DataFrame shape: {df.shape}")
print(f"DataFrame columns: {df.columns}")

# Prepare data
X = df.iloc[:, :-1].values  # All columns except the last one (features)
y = df.iloc[:, -1].values   # The last column (labels)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define MLP model without Dropout
model = keras.Sequential([
    keras.layers.Input(shape=(63,)),  # Input shape matches 21 landmarks * 3 (x, y, z)
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),  # Batch normalization for stability
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation='softmax')  # Output layer for 10 classes (0-9)
])

# Compile model with a learning rate scheduler
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Add early stopping and learning rate reduction
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=100,  # Increased epochs for better convergence
    batch_size=32,  # Larger batch size for stability
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_scheduler]
)

# Evaluate model
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[str(cls) for cls in label_encoder.classes_]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Convert model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TensorFlow Lite model
with open("asl_number_classifier_v2.tflite", "wb") as f:
    f.write(tflite_model)

print("Model training complete. TensorFlow Lite model saved as 'asl_number_classifier_v2.tflite'")
