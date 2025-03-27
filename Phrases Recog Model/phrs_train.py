import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define actions and parameters
ACTIONS = ["hello", "thanks", "iloveyou", "sorry"]
DATA_DIR = "action_sequences"
SEQUENCE_LENGTH = 20  # Ensure all sequences have the same length

X, y = [], []

# Load and preprocess data
for action_index, action in enumerate(ACTIONS):
    action_path = os.path.join(DATA_DIR, action)
    for npy_file in os.listdir(action_path):
        if not npy_file.endswith(".npy"):
            continue

        # Load the sequence
        sequence = np.load(os.path.join(action_path, npy_file), allow_pickle=True)

        # Extract pose landmarks from each frame
        sequence = np.array([frame["pose"] for frame in sequence])  # Extract pose landmarks

        # Normalize keypoints (optional)
        sequence[:, :, :2] = sequence[:, :, :2] / np.array([640, 480])  # Normalize x, y (assuming 640x480 resolution)

        # Pad or truncate the sequence
        if len(sequence) < SEQUENCE_LENGTH:
            padding = np.zeros((SEQUENCE_LENGTH - len(sequence), sequence.shape[1], sequence.shape[2]))
            sequence = np.vstack((sequence, padding))
        elif len(sequence) > SEQUENCE_LENGTH:
            sequence = sequence[:SEQUENCE_LENGTH]

        X.append(sequence)
        y.append(action_index)

X = np.array(X)
y = np.array(y)

# One-hot encode labels
y = to_categorical(y, num_classes=len(ACTIONS))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify the shape of X before reshaping
print(f"Shape of X before reshaping: {X_train.shape}")  # (num_samples, SEQUENCE_LENGTH, landmarks, coordinates)
print(f"Number of features used during training: {X_train.shape[2] * X_train.shape[3]}")  # landmarks * coordinates

# Reshape X_train and X_test to flatten the landmarks and coordinates
X_train = X_train.reshape(X_train.shape[0], SEQUENCE_LENGTH, -1)
X_test = X_test.reshape(X_test.shape[0], SEQUENCE_LENGTH, -1)

# Verify the reshaped input
print(f"Shape of X_train after reshaping: {X_train.shape}")  # (num_samples, SEQUENCE_LENGTH, features)

# Define the LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, X_train.shape[2])),
    Dropout(0.2),
    LSTM(128, return_sequences=False, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(ACTIONS), activation='softmax')  # Output layer for classification
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# Save the trained model
model.save("action_recognition_model.h5")
print("Model saved as action_recognition_model.h5")

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")