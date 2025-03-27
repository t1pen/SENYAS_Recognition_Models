import cv2
import numpy as np
import os
import mediapipe as mp

# Path for exported data, image files
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou', 'sorry'])

# Number of images per action
no_images = 200

# Create directories for each action
for action in actions:
    try:
        os.makedirs(os.path.join(DATA_PATH, action))
    except FileExistsError:
        pass

mp_hands = mp.solutions.hands  # Hands model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_hand_landmarks(image, results):
    # Draw right hand connections
    if results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use 0 for the default camera
# Set mediapipe model
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    # Initialize control variables
    capturing = False  # Toggle capturing with SPACE key
    action_index = 0  # Start with the first action
    image_count = 0  # Track the number of images captured for the current action

    # Define the fixed bounding box coordinates
    box_x_min, box_y_min = 200, 100  # Top-left corner of the box
    box_x_max, box_y_max = 400, 300  # Bottom-right corner of the box

    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Draw the fixed bounding box
        cv2.rectangle(frame, (box_x_min, box_y_min), (box_x_max, box_y_max), (0, 255, 0), 2)

        # Make detections
        image, results = mediapipe_detection(frame, hands)

        # Draw landmarks
        draw_hand_landmarks(image, results)

        # Display instructions and capturing status
        cv2.putText(image, f"Action: {actions[action_index]} | Capturing: {'ON' if capturing else 'OFF'}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, "[SPACE] Start/Stop  [N] Next Action  [Q] Quit", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Show capturing progress
        if capturing:
            cv2.putText(image, f"Captured: {image_count}/{no_images}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Handle key presses
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord(' '):  # Start/Stop capturing
            capturing = not capturing
            if capturing:
                image_count = 0  # Reset the image count
                print(f"Started capturing for action: {actions[action_index]}")
            else:
                print(f"Stopped capturing for action: {actions[action_index]}")
        elif key == ord('n'):  # Next action
            action_index = (action_index + 1) % len(actions)
            print(f"Switched to action: {actions[action_index]}")
            capturing = False  # Stop capturing when switching actions

        # Capture images if capturing is ON
        if capturing and image_count < no_images:
            # Ensure a hand is detected
            if results.multi_hand_landmarks:
                # Check if the hand is inside the fixed bounding box
                hand_landmarks = results.multi_hand_landmarks[0]
                h, w, _ = frame.shape
                hand_x_min = min([int(lm.x * w) for lm in hand_landmarks.landmark])
                hand_y_min = min([int(lm.y * h) for lm in hand_landmarks.landmark])
                hand_x_max = max([int(lm.x * w) for lm in hand_landmarks.landmark])
                hand_y_max = max([int(lm.y * h) for lm in hand_landmarks.landmark])

                # Check if the hand is fully inside the fixed bounding box
                if (hand_x_min >= box_x_min and hand_y_min >= box_y_min and
                        hand_x_max <= box_x_max and hand_y_max <= box_y_max):
                    # Save the current frame as an image
                    cropped_hand = frame[box_y_min:box_y_max, box_x_min:box_x_max]  # Crop the fixed box region
                    image_path = os.path.join(DATA_PATH, actions[action_index], f"{image_count}.jpg")
                    cv2.imwrite(image_path, cropped_hand)
                    print(f"Saved image {image_count} for action: {actions[action_index]}")
                    image_count += 1

            # Stop capturing if the required number of images is reached
            if image_count >= no_images:
                print(f"Completed capturing for action: {actions[action_index]}")
                capturing = False

    cap.release()
    cv2.destroyAllWindows()