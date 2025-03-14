import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)

# Define gestures
GESTURES = ["hello", "thanks", "iloveyou"]
DATA_DIR = "gesture_images"
max_images_per_gesture = 200  # Maximum number of images per gesture

# Create directories for gestures
for gesture in GESTURES:
    os.makedirs(os.path.join(DATA_DIR, gesture), exist_ok=True)

# Video capture
cap = cv2.VideoCapture(0)

# Control variables
gesture_index = 0  # Tracks the current gesture
recording = False  # Is capturing active?
image_count = 0  # Tracks images collected for the current gesture

print("Press 'SPACE' to Start/Stop Capturing, 'B' for Previous Gesture, 'N' for Next Gesture, 'Q' to Quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Start recording only if a hand is detected
    if recording and result.multi_hand_landmarks:
        img_path = os.path.join(DATA_DIR, GESTURES[gesture_index], f"{image_count}.jpg")
        cv2.imwrite(img_path, frame)  # Save the raw frame without overlay
        image_count += 1

        print(f"Captured {image_count}/{max_images_per_gesture} images for {GESTURES[gesture_index]}")

        # Stop recording after reaching 200 images
        if image_count >= max_images_per_gesture:
            print(f"Captured {max_images_per_gesture} images for {GESTURES[gesture_index]}")
            recording = False

    # Show current gesture and instructions (only in the OpenCV window, not in saved images)
    overlay = frame.copy()
    cv2.putText(overlay, f"Gesture: {GESTURES[gesture_index]} | Images: {image_count}/{max_images_per_gesture}", 
                (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, "[SPACE] Start/Stop  [B] Prev  [N] Next  [Q] Quit", 
                (20, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    cv2.imshow("ASL Image Collection", overlay)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Quit
        break
    elif key == ord(' '):  # Start/Stop recording
        if image_count < max_images_per_gesture:
            recording = not recording
            print(f"{'Started' if recording else 'Stopped'} recording for {GESTURES[gesture_index]}")
    elif key == ord('n'):  # Next gesture
        gesture_index = (gesture_index + 1) % len(GESTURES)
        image_count = 0  # Reset image count for the new gesture
        print(f"Switched to gesture: {GESTURES[gesture_index]}")
    elif key == ord('b'):  # Previous gesture
        gesture_index = (gesture_index - 1) % len(GESTURES)
        image_count = 0  # Reset image count for the new gesture
        print(f"Switched to gesture: {GESTURES[gesture_index]}")

cap.release()
cv2.destroyAllWindows()
