import cv2
import numpy as np
import tensorflow.lite as tflite
import mediapipe as mp
from collections import deque
from statistics import mode
import time

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="asl_number_classifier.tflite")
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
            # Use raw landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])  # Collect raw x, y, z coordinates
            
            # Convert landmarks to a NumPy array
            landmarks = np.array(landmarks, dtype=np.float32)

            # Normalize the landmarks (to range [0, 1])
            min_vals = np.min(landmarks, axis=0)  # Minimum values for x, y, z
            max_vals = np.max(landmarks, axis=0)  # Maximum values for x, y, z
            landmarks = (landmarks - min_vals) / (max_vals - min_vals + 1e-6)  # Normalize

            # Reshape to match the model's input shape
            input_data = landmarks.reshape(1, 21, 3, 1)  # Shape: (1, 21, 3, 1)
            
            # Ensure the input data matches the expected shape
            if input_data.shape[1:] != tuple(input_details[0]['shape'][1:]):
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

            # Display prediction only if confidence is 1.00
            if confidence >= 0.70:
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
