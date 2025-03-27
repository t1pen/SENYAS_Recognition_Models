import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("action_recognition_model.h5")
ACTIONS = ["hello", "thanks", "iloveyou", "sorry"]
SEQUENCE_LENGTH = 20  # Same as used during training

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5)

# Indices for relevant pose landmarks
POSE_LANDMARKS_RELEVANT = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 15  # Nose, eyes, ears, left shoulder, elbow, wrist
]

# Initialize variables
sequence = []  # Stores the sequence of frames
predicted_action = ""  # Stores the predicted action

# Start video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Press 'Q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    hand_result = hands.process(rgb_frame)
    pose_result = pose.process(rgb_frame)

    # Initialize keypoints with zero-filled placeholders
    keypoints = np.zeros((12, 3))  # 12 landmarks, 3 coordinates (x, y, z)

    # Extract relevant pose landmarks
    if pose_result.pose_landmarks:
        keypoints[:len(POSE_LANDMARKS_RELEVANT)] = np.array([
            (pose_result.pose_landmarks.landmark[i].x,
             pose_result.pose_landmarks.landmark[i].y,
             pose_result.pose_landmarks.landmark[i].z)
            for i in POSE_LANDMARKS_RELEVANT
        ])

    # Add the frame's keypoints to the sequence
    sequence.append(keypoints.flatten())  # Flatten to match the training input shape

    # Ensure the sequence length is consistent
    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    # Perform prediction if the sequence is full
    if len(sequence) == SEQUENCE_LENGTH:
        input_data = np.array(sequence).reshape(1, SEQUENCE_LENGTH, -1)  # Reshape for the model
        prediction = model.predict(input_data)
        predicted_action = ACTIONS[np.argmax(prediction)]

    # Display the predicted action
    cv2.putText(frame, f"Action: {predicted_action}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Real-Time Action Recognition", frame)

    # Exit on 'Q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()