import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Path to the .pkl file
FEATURES_PATH = os.path.join('features.pkl')

# Load features and labels
with open(FEATURES_PATH, 'rb') as f:  # Open the file in binary read mode
    data = pickle.load(f)  # Pass the file object to pickle.load()

features = np.array(data['features'])
labels = np.array(data['labels'])

# One-hot encode the labels
num_classes = len(set(labels))  # Number of unique classes
labels = to_categorical(labels, num_classes=num_classes)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define the TensorFlow model
model = Sequential([
    Dense(128, activation='relu', input_shape=(features.shape[1],)),  # Input layer
    Dropout(0.3),  # Dropout for regularization
    Dense(64, activation='relu'),  # Hidden layer
    Dropout(0.3),  # Dropout for regularization
    Dense(num_classes, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model to a file
MODEL_PATH = 'tensorflow_model.h5'
model.save(MODEL_PATH)
print(f"Trained model saved to {MODEL_PATH}")

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
TFLITE_MODEL_PATH = 'tensorflow_model.tflite'
with open(TFLITE_MODEL_PATH, 'wb') as tflite_file:
    tflite_file.write(tflite_model)
print(f"TensorFlow Lite model saved to {TFLITE_MODEL_PATH}")

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=['hello', 'thanks', 'iloveyou', 'sorry']))
print(f"Accuracy: {accuracy_score(y_test_classes, y_pred_classes) * 100:.2f}%")