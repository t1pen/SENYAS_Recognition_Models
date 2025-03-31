import cv2
import numpy as np
import os
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# Path for exported data, numpy array
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'iloveyou', 'sorry'])
no_sequences = 30
sequence_length = 30

# Create directories for each action
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Set up the MediaPipe Holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    action_idx = 0  # Track current action
    sequence_idx = 0  # Track current sequence
    frame_idx = 0  # Track current frame
    capturing = False  # Flag to control capturing state

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        if results is not None:
            # Draw landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Display status
        action = actions[action_idx]
        status = "CAPTURING" if capturing else "WAITING"
        cv2.putText(image, f'Action: {action} | Sequence: {sequence_idx} | Frame: {frame_idx}', (15, 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f'Status: {status}', (15, 32),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, 'CONTROLS:', (15, 52),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, 'SPACE - Start/Stop Capture | N - Next Action | R - Reset Sequence | Q - Quit', 
                   (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', image)

        # Handle keyboard controls
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord(' '):  # Toggle capturing
            capturing = not capturing
            if capturing:
                print(f"Started capturing {action}")
            else:
                print(f"Stopped capturing {action}")
        elif key == ord('n'):  # Next action
            action_idx = (action_idx + 1) % len(actions)
            sequence_idx = 0
            frame_idx = 0
            capturing = False
            print(f"Switched to action: {actions[action_idx]}")
        elif key == ord('r'):  # Reset current sequence
            frame_idx = 0
            capturing = False
            print(f"Reset sequence for {action}")

        # Save frame if capturing
        if capturing and results is not None:
            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, str(sequence_idx), str(frame_idx))
            np.save(npy_path, keypoints)
            frame_idx += 1

            # Move to next sequence if frames complete
            if frame_idx == sequence_length:
                sequence_idx += 1
                frame_idx = 0
                capturing = False
                print(f"Completed sequence {sequence_idx-1} for {action}")

                # Move to next action if sequences complete
                if sequence_idx == no_sequences:
                    action_idx = (action_idx + 1) % len(actions)
                    sequence_idx = 0
                    print(f"Completed all sequences for {action}")
                    if action_idx == 0:
                        print("Completed all actions!")
                        break

    cap.release()
    cv2.destroyAllWindows()