import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
X_train = X_train / np.max(X_train)
X_test = X_test / np.max(X_test)

# Define CNN model
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')  # 10 classes (0-9)
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Evaluate model
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Convert model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TensorFlow Lite model
with open("asl_number_classifier_v2.tflite", "wb") as f:
    f.write(tflite_model)

print("Model training complete. TensorFlow Lite model saved as 'asl_number_classifier_v2.tflite'")
