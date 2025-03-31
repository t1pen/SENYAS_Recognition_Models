import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import os

# Configuration
CONFIG = {
    'model_path': 'gesture_model.tflite',  # Changed to use final model
    'actions': ['hello', 'thanks', 'iloveyou', 'sorry'],
    'sequence_length': 30,
    'threshold': 0.85,
    'min_consecutive_predictions': 5,
    'prediction_history_size': 15,
    'confidence_threshold': 0.90,
    'colors': [(245,117,16), (117,245,16), (16,117,245), (245,245,16)]
}

class SignLanguageDetector:
    def __init__(self, config):
        self.config = config
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.sequence = []
        self.predictions = []
        self.sentence = []
        self.setup_model()

    def setup_model(self):
        """Initialize TFLite model"""
        try:
            if not os.path.exists(self.config['model_path']):
                raise FileNotFoundError(f"Model file not found: {self.config['model_path']}")
                
            self.interpreter = tf.lite.Interpreter(model_path=self.config['model_path'])
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print("\nModel Details:")
            print(f"Model path: {self.config['model_path']}")
            print(f"Input shape: {self.input_details[0]['shape']}")
            print(f"Input type: {self.input_details[0]['dtype']}")
            print(f"Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def extract_keypoints(self, results):
        """Extract keypoints from MediaPipe results"""
        pose = np.array([[res.x, res.y, res.z, res.visibility] 
                        for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        lh = np.array([[res.x, res.y, res.z] 
                      for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] 
                      for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, lh, rh])

    def process_frame(self, frame, holistic):
        """Process a single frame"""
        # Make detections
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract keypoints and process
        if results.left_hand_landmarks is not None or results.right_hand_landmarks is not None:
            keypoints = self.extract_keypoints(results)
            if np.mean(np.abs(keypoints[-126:])) > 0.01:  # Motion detection
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-self.config['sequence_length']:]
        else:
            self.sequence = []
            self.predictions = []

        # Draw landmarks
        self.draw_landmarks(image, results)
        
        # Make prediction if sequence is complete
        if len(self.sequence) == self.config['sequence_length']:
            self.make_prediction(image)

        return image

    def make_prediction(self, image):
        """Make prediction from sequence"""
        try:
            # Prepare input data
            input_data = np.array(self.sequence, dtype=np.float32)
            input_data = (input_data - np.mean(input_data)) / (np.std(input_data) + 1e-6)
            input_data = np.expand_dims(input_data, axis=0)

            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            res = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

            # Process prediction
            if res[np.argmax(res)] > self.config['threshold']:
                prediction = self.config['actions'][np.argmax(res)]
                self.predictions.append(prediction)
                self.predictions = self.predictions[-self.config['prediction_history_size']:]

                # Update sentence if predictions are consistent
                if (len(self.predictions) >= self.config['min_consecutive_predictions'] and
                    len(set(self.predictions[-self.config['min_consecutive_predictions']:])) == 1):
                    if len(self.sentence) == 0 or prediction != self.sentence[-1]:
                        self.sentence.append(prediction)
                        self.sentence = self.sentence[-5:]  # Keep last 5 words

            # Visualize probabilities
            self.draw_probabilities(image, res)

        except Exception as e:
            print(f"Prediction error: {e}")

    def draw_landmarks(self, image, results):
        """Draw MediaPipe landmarks"""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            )
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

    def draw_probabilities(self, image, res):
        """Draw probability bars"""
        for num, prob in enumerate(res):
            cv2.rectangle(image, (0,60+num*40), (int(prob*100), 90+num*40), 
                         self.config['colors'][num], -1)
            cv2.putText(image, f'{self.config["actions"][num]} ({prob:.2f})', 
                       (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def main():
    detector = SignLanguageDetector(CONFIG)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    prev_time = 0

    with detector.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            frame = cv2.flip(frame, 1)
            image = detector.process_frame(frame, holistic)

            # Draw sentence and FPS
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(detector.sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if prev_time > 0 else 0.0
            prev_time = current_time
            cv2.putText(image, f'FPS: {int(fps)}', (550, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            cv2.imshow('Sign Language Recognition', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()