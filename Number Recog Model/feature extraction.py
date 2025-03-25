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
DATA_DIR = './dataset_2'

data = []
labels = []

# Iterate through dataset folders
for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        continue
    
    print(f"Processing folder: {label}")  # Feedback for folder processing
    
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        
        # Read and process image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []
                
                # Extract (x, y) coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                
                # Normalize coordinates (shift based on min values)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x - min(x_)
                    y = hand_landmarks.landmark[i].y - min(y_)
                    data_aux.append(x)
                    data_aux.append(y)
                
                data.append(data_aux)
                labels.append(int(label))  # Convert label to integer
    
    print(f"Completed folder: {label}")  # Feedback for folder completion

# Convert to DataFrame
df = pd.DataFrame(data)
df['label'] = labels

# Save extracted features as a DataFrame
with open('features_v2.pkl', 'wb') as f:
    pickle.dump(df, f)

print("Feature extraction complete. Data saved as features_v2.pkl")
