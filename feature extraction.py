import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd

dataset_path = "dataset/"
output_csv = "features.csv"
labels = [str(i) for i in range(10)]  # Labels from 0-9

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def extract_features(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return landmarks, True  # Return landmarks and True if hands detected
    return None, False  # Return None and False if no hands detected

features = []
no_hand_detected = []
for label in labels:
    label_path = os.path.join(dataset_path, label)
    if not os.path.exists(label_path):
        continue
    
    for filename in os.listdir(label_path):
        image_path = os.path.join(label_path, filename)
        landmarks, hand_detected = extract_features(image_path)
        if hand_detected:
            features.append([label] + landmarks)
        else:
            no_hand_detected.append(image_path)  # Store images with no hands detected

# Convert to DataFrame and save
columns = ["label"] + [f"x{i//3}_y{i//3}_z{i//3}" for i in range(63)]
df = pd.DataFrame(features, columns=columns)
df.to_csv(output_csv, index=False)

print(f"Feature extraction complete. Data saved to {output_csv}")
if no_hand_detected:
    print("Warning: No hands detected in the following images:")
    for img in no_hand_detected:
        print(img)