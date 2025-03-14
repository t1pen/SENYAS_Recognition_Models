import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load extracted features from Pickle file
FEATURES_DIR = "gesture_features_pkl"
pkl_path = f"{FEATURES_DIR}/gesture_features.pkl"

with open(pkl_path, "rb") as f:
    gesture_data = pickle.load(f)

# Prepare dataset
X = []  # Feature sequences
y = []  # Labels

GESTURES = list(gesture_data.keys())  # Get gesture names
num_classes = len(GESTURES)  # Number of gestures

for gesture_index, gesture in enumerate(GESTURES):
    for sequence in gesture_data[gesture]:
        X.append(sequence)
        y.append(gesture_index)  # Label the data

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# One-hot encode labels
y = to_categorical(y, num_classes=num_classes)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# Build MLP Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("gesture_recognition_mlp.h5")

print("Training completed and model saved as 'gesture_recognition_mlp.h5'.")
