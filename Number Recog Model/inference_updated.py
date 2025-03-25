import cv2
import numpy as np
import tensorflow.lite as tflite
import mediapipe as mp
from collections import deque
from statistics import mode
import time

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="asl_number_classifier_v2.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Mediapipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Label map
labels = [str(i) for i in range(10)]

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Prediction smoothing
prediction_history = deque(maxlen=10)

# FPS calculation
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip and process frame
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    roi_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(roi_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Detect closed fist based on landmarks
            distances = []
            for i in range(1, len(hand_landmarks.landmark)):
                x1, y1 = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y  # Wrist
                x2, y2 = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y
                distances.append(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
            
            if all(d < 0.1 for d in distances):  # Adjust threshold as needed
                cv2.putText(frame, "Closed Fist Detected", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                continue  # Skip further processing for this frame
            
            # Normalize landmarks
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            z_coords = [lm.z for lm in hand_landmarks.landmark]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            min_z, max_z = min(z_coords), max(z_coords)
            epsilon = 1e-6  # Avoid division by zero

            landmarks = []
            for lm in hand_landmarks.landmark:
                x = (lm.x - min_x) / (max_x - min_x + epsilon)
                y = (lm.y - min_y) / (max_y - min_y + epsilon)
                z = (lm.z - min_z) / (max_z - min_z + epsilon)
                landmarks.extend([x, y, z])
            
            # Ensure the input data matches the expected shape
            input_data = np.array(landmarks, dtype=np.float32).reshape(1, -1)
            if input_data.shape[1] != input_details[0]['shape'][1]:
                continue
            
            # Perform inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction = np.argmax(output_data)
            confidence = output_data[0][prediction]

            # Add prediction to history
            prediction_history.append(prediction)
            smoothed_prediction = mode(prediction_history)

            # Display prediction
            confidence_threshold = 0.99
            if confidence >= confidence_threshold:
                cv2.putText(frame, f"Prediction: {labels[smoothed_prediction]} ({confidence:.2f})", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "No Gesture Detected", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Show frame
    cv2.imshow("ASL Number Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
