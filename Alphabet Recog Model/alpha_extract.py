import cv2
import mediapipe as mp
import os
import pandas as pd
import pickle

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Define dataset directory
dataset_dir = "Alphabet_data_2"
output_pkl = "landmarks_dataset_v2.pkl"

# Prepare dataset storage
data = []

# Dictionary to store sample count per letter
letter_counts = {}

# Iterate through each alphabet folder
for letter in sorted(os.listdir(dataset_dir)):
    letter_dir = os.path.join(dataset_dir, letter)
    if not os.path.isdir(letter_dir):
        continue

    print(f"Processing letter: {letter}")
    letter_counts[letter] = 0  # Initialize count
    
    # Process each image in the letter directory
    for img_name in sorted(os.listdir(letter_dir)):
        img_path = os.path.join(letter_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        # Extract landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [letter]  # Start with label
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                data.append(landmarks)
                letter_counts[letter] += 1  # Increment count
                break  # Only process the first detected hand

# Convert to DataFrame
columns = ["Label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
df = pd.DataFrame(data, columns=columns)

# Save dataset as a Pickle file
with open(output_pkl, "wb") as file:
    pickle.dump(df, file)

# Display dataset counts
print("\nLandmark extraction completed.")
print("Sample count per letter:")
for letter, count in letter_counts.items():
    print(f"{letter}: {count}")

print(f"Data saved to {output_pkl}")
