import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf

# Configuration
CONFIG = {
    'data_path': 'MP_Data',
    'actions': ['hello', 'thanks', 'iloveyou', 'sorry'],
    'sequence_length': 30,
    'no_sequences': 30,
    'test_size': 0.05,
    'random_seed': 42,
    'model_name': 'gesture_model'
}

def load_and_preprocess_data(config):
    """Load and preprocess data with error handling"""
    sequences = []
    labels = []
    label_map = {label: num for num, label in enumerate(config['actions'])}

    for action in config['actions']:
        action_path = os.path.join(config['data_path'], action)
        try:
            sequence_dirs = sorted([int(f) for f in os.listdir(action_path) if f.isdigit()])
            for sequence in sequence_dirs:
                window = []
                for frame_num in range(config['sequence_length']):
                    npy_path = os.path.join(action_path, str(sequence), f"{frame_num}.npy")
                    if os.path.exists(npy_path):
                        window.append(np.load(npy_path))
                
                if len(window) == config['sequence_length']:
                    sequences.append(window)
                    labels.append(label_map[action])
                else:
                    print(f"Warning: Incomplete sequence {sequence} for {action}")
        except Exception as e:
            print(f"Error processing {action}: {str(e)}")
            continue

    return np.array(sequences), to_categorical(labels).astype(int)

def create_model(input_shape, num_classes):
    """Create and compile the LSTM model"""
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        LSTM(128, return_sequences=True, activation='relu'),
        Dropout(0.2),
        LSTM(64, return_sequences=False, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return model

def save_metrics(metrics_data, filename='model_metrics.txt'):
    """Save metrics to file"""
    with open(filename, 'w') as f:
        for section, content in metrics_data.items():
            f.write(f"\n{section}\n")
            f.write("="* len(section) + "\n")
            f.write(str(content) + "\n")

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data(CONFIG)
    
    # Normalize data
    X = (X - np.mean(X)) / (np.std(X) + 1e-8)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_seed'],
        stratify=y
    )

    # Create and train model
    model = create_model((CONFIG['sequence_length'], 258), len(CONFIG['actions']))
    
    # Callbacks
    callbacks = [
        TensorBoard(log_dir='Logs'),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint(
            f"{CONFIG['model_name']}_best.h5",
            monitor='val_categorical_accuracy',
            save_best_only=True
        )
    ]

    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=200,
            batch_size=32,
            callbacks=callbacks
        )

        # Evaluate model
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # Save metrics
        metrics_data = {
            "Classification Report": classification_report(
                y_true_classes, 
                y_pred_classes,
                target_names=CONFIG['actions'],
                labels=range(len(CONFIG['actions'])),
                zero_division=0
            ),
            "Confusion Matrix": confusion_matrix(y_true_classes, y_pred_classes)
        }
        save_metrics(metrics_data)

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.experimental_enable_resource_variables = True
        tflite_model = converter.convert()

        with open(f"{CONFIG['model_name']}.tflite", 'wb') as f:
            f.write(tflite_model)

        print("Training and conversion completed successfully!")

    except Exception as e:
        print(f"Error during training or conversion: {str(e)}")
        raise

if __name__ == "__main__":
    main()