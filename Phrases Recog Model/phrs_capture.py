import cv2
import mediapipe as mp
import os
import json

# Initialize MediaPipe Solutions
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5)

# Define actions
ACTIONS = ["hello", "thanks", "iloveyou", "goodbye", "sorry"]
DATA_DIR = "action_sequences"
max_sequences_per_action = 200  # Maximum number of sequences per action

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
    face_result = face_mesh.process(rgb_frame)
    pose_result = pose.process(rgb_frame)

    keypoints = {
        "hands": [],
        "face": [],
        "pose": []
    }

    # Extract hand landmarks
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            keypoints["hands"].append([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

    # Extract face landmarks
    if face_result.multi_face_landmarks:
        for face_landmarks in face_result.multi_face_landmarks:
            keypoints["face"] = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]

    # Extract pose landmarks
    if pose_result.pose_landmarks:
        keypoints["pose"] = [(lm.x, lm.y, lm.z) for lm in pose_result.pose_landmarks.landmark]

    # Start recording only if a hand is detected
    if recording and keypoints["hands"]:
        frame_sequence.append(keypoints)

        print(f"Captured {len(frame_sequence)}/{max_sequences_per_action} frames for {ACTIONS[action_index]}")

        # Stop recording after reaching max frames per sequence
        if len(frame_sequence) >= max_sequences_per_action:
            sequence_path = os.path.join(DATA_DIR, ACTIONS[action_index], f"sequence_{sequence_count}.json")
            with open(sequence_path, "w") as f:
                json.dump(frame_sequence, f)

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
