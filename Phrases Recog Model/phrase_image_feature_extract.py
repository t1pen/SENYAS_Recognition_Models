import os
import cv2
import mediapipe as mp
import numpy as np
import pickle

# Path for the dataset
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou', 'sorry'])

# Path to save the extracted features
FEATURES_PATH = os.path.join('features.pkl')

mp_hands = mp.solutions.hands  # Hands model

def extract_hand_landmarks(image, model):
    """Extract and normalize hand landmarks from an image using MediaPipe."""
    image_height, image_width, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image.flags.writeable = False  # Make image non-writeable for efficiency
    results = model.process(image)  # Process the image
    image.flags.writeable = True  # Make image writeable again

    # Extract landmarks for the first detected hand
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = np.array([[lm.x * image_width, lm.y * image_height, lm.z] for lm in hand_landmarks.landmark])

        # Normalize landmarks by centering around the wrist (landmark 0)
        wrist = landmarks[0]
        landmarks -= wrist  # Center the landmarks around the wrist

        # Normalize x and y by image width and height
        landmarks[:, 0] /= image_width  # Normalize x
        landmarks[:, 1] /= image_height  # Normalize y

        return landmarks.flatten()
    else:
        return np.zeros(21 * 3)  # Return zero-filled array if no hand is detected

# Initialize MediaPipe Hands
with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
    features = []
    labels = []

    for action_index, action in enumerate(actions):
        action_path = os.path.join(DATA_PATH, action)
        for image_name in os.listdir(action_path):
            image_path = os.path.join(action_path, image_name)
            image = cv2.imread(image_path)

            # Extract and normalize hand landmarks
            landmarks = extract_hand_landmarks(image, hands)

            # Append features and labels
            features.append(landmarks)
            labels.append(action_index)

    # Save features and labels as a .pkl file
    with open(FEATURES_PATH, 'wb') as f:
        pickle.dump({'features': np.array(features), 'labels': np.array(labels)}, f)

print(f"Features and labels saved to {FEATURES_PATH}")