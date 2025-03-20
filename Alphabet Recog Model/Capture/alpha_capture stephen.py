import cv2
import os
import string

# Define the main dataset folder
dataset_dir = "Alphabet_data_2"
os.makedirs(dataset_dir, exist_ok=True)

# Define alphabet list
alphabet = list(string.ascii_uppercase)
current_letter_index = 0
capture = False
frame_count = 0
frames_per_letter = 200

# Initialize camera
cap = cv2.VideoCapture(0)

# Define ROI (Region of Interest) - Adjust based on your needs
roi_x, roi_y, roi_w, roi_h = 250, 100, 300, 300  # Example values

print("Press Space to Start/Stop Capturing, N for Next Letter, B for Previous Letter, Q to Quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    # Flip frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Extract ROI
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    
    # Display current letter
    current_letter = alphabet[current_letter_index]
    cv2.putText(frame, f"Capturing: {current_letter} ({frame_count}/{frames_per_letter})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Draw ROI rectangle
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 0), 2)
    
    # Show full frame with ROI highlighted
    cv2.imshow("ASL Alphabet Dataset Collection", frame)
    
    # Save frames if capturing
    if capture and frame_count < frames_per_letter:
        letter_dir = os.path.join(dataset_dir, current_letter)
        os.makedirs(letter_dir, exist_ok=True)
        file_path = os.path.join(letter_dir, f"{frame_count:03d}.jpg")
        cv2.imwrite(file_path, roi)
        frame_count += 1
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord(' '):  # Start/Stop capturing
        capture = not capture
        print("Capturing" if capture else "Paused")
    elif key == ord('n') and current_letter_index < len(alphabet) - 1:  # Next letter
        current_letter_index += 1
        frame_count = 0
        capture = False
        print(f"Switched to {alphabet[current_letter_index]}")
    elif key == ord('b') and current_letter_index > 0:  # Previous letter
        current_letter_index -= 1
        frame_count = 0
        capture = False
        print(f"Switched to {alphabet[current_letter_index]}")

# Release resources
cap.release()
cv2.destroyAllWindows()
