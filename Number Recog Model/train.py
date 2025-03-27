import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
features_pkl = "features.pkl"
df = pd.read_pickle(features_pkl)  # Read the .pkl file directly as a DataFrame

# Debug: Inspect the loaded DataFrame
print(f"Loaded DataFrame shape: {df.shape}")
print(f"DataFrame columns: {df.columns}")

# Prepare data
X = df.iloc[:, :-1].values  # All columns except the last one (features)
y = df.iloc[:, -1].values   # The last column (labels)

# Normalize the features (x, y, z coordinates)
X_min = np.min(X, axis=0)  # Minimum value for each feature
X_max = np.max(X, axis=0)  # Maximum value for each feature
X = (X - X_min) / (X_max - X_min + 1e-6)  # Normalize to range [0, 1]

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Reshape input data for CNN
X = X.reshape(-1, 21, 3, 1)  # Reshape to (samples, 21 landmarks, 3 coordinates, 1 channel)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define a CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=(21, 3, 1)),  # Smaller kernel size
    keras.layers.MaxPooling2D((2, 1)),  # Adjust pooling size to avoid reducing dimensions too much
    keras.layers.Conv2D(64, (2, 2), activation='relu'),  # Second convolutional layer
    keras.layers.Flatten(),  # Flatten the output
    keras.layers.Dense(128, activation='relu'),  # Fully connected layer
    keras.layers.Dense(10, activation='softmax')  # Output layer for 10 classes (0-9)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,  # Reduced epochs for simplicity
    batch_size=32,  # Batch size
    validation_data=(X_test, y_test)
)

# Evaluate the model
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
with open("asl_number_classifier.tflite", "wb") as f:
    f.write(tflite_model)

print("Model training complete. TensorFlow Lite model saved as 'asl_number_classifier.tflite'")
