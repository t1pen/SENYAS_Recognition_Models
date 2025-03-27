import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dataset directory (expects subfolders named 0-9 with images inside)
DATA_DIR = './dataset'

data = []
labels = []
sample_counts = {}  # Dictionary to track the number of samples per label

# Iterate through dataset folders
for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        continue
    
    print(f"Processing folder: {label}")  # Feedback for folder processing
    sample_counts[label] = 0  # Initialize sample count for this label
    
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        
        # Read and process image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        data_aux = [0] * (21 * 3)  # Initialize with zeros for 21 landmarks (x, y, z)
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if idx > 0:  # Process only the first hand
                    break
                
                # Extract (x, y, z) coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    z = hand_landmarks.landmark[i].z
                    data_aux[i * 3] = x  # Set x-coordinate
                    data_aux[i * 3 + 1] = y  # Set y-coordinate
                    data_aux[i * 3 + 2] = z  # Set z-coordinate
        
        # Add the feature vector and label
        data.append(data_aux)
        labels.append(int(label))  # Convert label to integer
        sample_counts[label] += 1  # Increment sample count for this label
    
    print(f"Completed folder: {label}")  # Feedback for folder completion

# Display the number of samples processed for each label
print("\nSample counts per label:")
for label, count in sample_counts.items():
    print(f"Label {label}: {count} samples")

# Convert to DataFrame
df = pd.DataFrame(data)
df['label'] = labels

# Save extracted features as a DataFrame
with open('features.pkl', 'wb') as f:
    pickle.dump(df, f)

print("\nFeature extraction complete. Data saved as features.pkl")
