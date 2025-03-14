import cv2
import mediapipe as mp
import numpy as np
import os
import pickle

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Define dataset paths
IMAGE_DATA_DIR = "gesture_images"  # Folder containing images
FEATURES_DIR = "gesture_features_pkl"  # Folder to save extracted features

# Ensure the features directory exists
os.makedirs(FEATURES_DIR, exist_ok=True)

# Dictionary to store extracted features
gesture_data = {}

# Process each gesture folder
for gesture in os.listdir(IMAGE_DATA_DIR):
    gesture_path = os.path.join(IMAGE_DATA_DIR, gesture)
    gesture_data[gesture] = []  # Initialize list for each gesture

    for img_name in os.listdir(gesture_path):
        img_path = os.path.join(gesture_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue  # Skip unreadable images

        # Convert to RGB for MediaPipe processing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_image)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Extract (x, y, z) coordinates
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                
                # Append features to the list
                gesture_data[gesture].append(landmarks)

                print(f"Extracted XYZ features for {img_name}")

# Save extracted features to a Pickle file
pkl_path = os.path.join(FEATURES_DIR, "gesture_features.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(gesture_data, f)

print(f"Feature extraction completed. Data saved to {pkl_path}")
