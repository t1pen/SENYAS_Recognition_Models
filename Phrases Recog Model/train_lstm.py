import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define the directory containing the .npy files
DATA_DIR = "action_sequences"
ACTIONS = ["hello", "thanks", "iloveyou", "sorry"]  # Define your actions
SEQUENCE_LENGTH = 30  # Number of frames per sequence

# Load and preprocess data
X, y = [], []

for action_index, action in enumerate(ACTIONS):
    action_path = os.path.join(DATA_DIR, action)
    for npy_file in os.listdir(action_path):
        if not npy_file.endswith(".npy"):
            continue  # Skip non-npy files

        npy_path = os.path.join(action_path, npy_file)
        sequence = np.load(npy_path, allow_pickle=True)  # Load the .npy file

        # Pad or truncate the sequence to SEQUENCE_LENGTH
        if len(sequence) < SEQUENCE_LENGTH:
            # Pad with zeros if the sequence is too short
            padding = np.zeros((SEQUENCE_LENGTH - len(sequence), sequence[0].shape[0]))
            sequence = np.vstack((sequence, padding))
        elif len(sequence) > SEQUENCE_LENGTH:
            # Truncate if the sequence is too long
            sequence = sequence[:SEQUENCE_LENGTH]

        # Append the sequence and its label
        X.append(sequence)
        y.append(action_index)

# Ensure X is not empty
if len(X) == 0:
    raise ValueError("No valid sequences found in the dataset. Check your data directory and preprocessing steps.")

# Convert X and y to numpy arrays
X = np.array(X)
y = to_categorical(y, num_classes=len(ACTIONS))  # One-hot encode labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, X.shape[2])),
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
model.save("lstm_action_recognition_model.h5")
print("Model saved as lstm_action_recognition_model.h5")

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")