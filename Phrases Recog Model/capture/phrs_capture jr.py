import cv2
import mediapipe as mp
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np

# Initialize MediaPipe Solutions
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5)

# Define actions
ACTIONS = ["hello", "thanks", "iloveyou", "sorry"]
DATA_DIR = "action_sequences"
max_sequences_per_action = 20  # Maximum number of sequences per action
frames_per_sequence = 30  # Number of frames per sequence

# Create directories for actions
for action in ACTIONS:
    os.makedirs(os.path.join(DATA_DIR, action), exist_ok=True)

# Video capture
cap = cv2.VideoCapture(0)

# Control variables
action_index = 0  # Tracks the current action
recording = False  # Is capturing active?
sequence_count = 0  # Tracks sequences collected for the current action
frame_sequence = []  # Stores keypoints for each sequence

print("Press 'SPACE' to Start/Stop Capturing, 'B' for Previous Action, 'N' for Next Action, 'Q' to Quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    hand_result = hands.process(rgb_frame)
    pose_result = pose.process(rgb_frame)

    # Define total landmark counts
    HAND_LANDMARKS = 21  # MediaPipe Hands has 21 landmarks
    POSE_LANDMARKS = 33  # MediaPipe Pose has 33 landmarks

    # Initialize keypoints with zero-filled placeholders
    keypoints = {
        "hands": np.zeros((HAND_LANDMARKS, 3)),
        "pose": np.zeros((POSE_LANDMARKS, 3))
    }

    # Extract hand landmarks
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            keypoints["hands"] = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Extract pose landmarks
    if pose_result.pose_landmarks:
        keypoints["pose"] = np.array([(lm.x, lm.y, lm.z) for lm in pose_result.pose_landmarks.landmark])
        mp.solutions.drawing_utils.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Start recording regardless of missing keypoints
    if recording:
        frame_sequence.append(keypoints)

        print(f"Captured {len(frame_sequence)}/{frames_per_sequence} frames for {ACTIONS[action_index]}")

        # Stop recording after reaching max frames per sequence
        if len(frame_sequence) >= frames_per_sequence:
            sequence_path = os.path.join(DATA_DIR, ACTIONS[action_index], f"sequence_{20+sequence_count}.npy")
            np.save(sequence_path, np.array(frame_sequence, dtype=object))

            print(f"Saved sequence {sequence_count} for {ACTIONS[action_index]}")
            sequence_count += 1
            frame_sequence = []  # Reset sequence storage
            recording = False

    # Show current action and instructions
    overlay = frame.copy()
    cv2.putText(overlay, f"Action: {ACTIONS[action_index]} | Sequences: {sequence_count}/{max_sequences_per_action}", 
                (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, "[SPACE] Start/Stop  [B] Prev  [N] Next  [Q] Quit", 
                (20, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    cv2.imshow("Action Recognition Data Collection", overlay)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Quit
        break
    elif key == ord(' '):  # Start/Stop recording
        if sequence_count < max_sequences_per_action:
            recording = not recording
            frame_sequence = []  # Reset sequence storage on new recording
            print(f"{'Started' if recording else 'Stopped'} recording for {ACTIONS[action_index]}")
    elif key == ord('n'):  # Next action
        action_index = (action_index + 1) % len(ACTIONS)
        sequence_count = 0  # Reset sequence count for the new action
        print(f"Switched to action: {ACTIONS[action_index]}")
    elif key == ord('b'):  # Previous action
        action_index = (action_index - 1) % len(ACTIONS)
        sequence_count = 0  # Reset sequence count for the new action
        print(f"Switched to action: {ACTIONS[action_index]}")

cap.release()
cv2.destroyAllWindows()
