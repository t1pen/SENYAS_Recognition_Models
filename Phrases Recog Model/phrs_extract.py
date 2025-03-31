import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 30

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

print(np.array(sequences).shape)

print(np.array(labels).shape)

X = np.array(sequences)

y = to_categorical(labels).astype(int)

# Preprocess the data
print("Original shapes:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Normalize the data
X = X.astype('float32')
X = (X - np.mean(X)) / np.std(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

print("\nTraining shapes:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define the model with correct input shape
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 258)),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(actions.shape[0], activation='softmax')
])

# Compile the model
model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

# Print model summary
model.summary()

# Add early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, 
    y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[tb_callback, early_stopping]
)

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calculate and print metrics
print("\nModel Evaluation Metrics:")
print("------------------------")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=actions))

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=actions,
            yticklabels=actions)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.close()

# Create a detailed confusion matrix visualization
plt.figure(figsize=(12, 8))
confusion_matrix_display = sns.heatmap(cm, 
                                     annot=True,
                                     cmap='Blues',
                                     fmt='g',
                                     xticklabels=actions,
                                     yticklabels=actions)

# Customize the plot
plt.title('Confusion Matrix for Sign Language Recognition', pad=20, fontsize=16)
plt.xlabel('Predicted Labels', labelpad=10, fontsize=12)
plt.ylabel('True Labels', labelpad=10, fontsize=12)

# Add color bar
cbar = confusion_matrix_display.collections[0].colorbar
cbar.set_label('Number of Predictions', rotation=270, labelpad=25)

# Rotate x-labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Add grid to make it easier to read
plt.grid(False)

# Tight layout to prevent label cutoff
plt.tight_layout()

# Save the plot with high DPI for better quality
plt.savefig('confusion_matrix_detailed.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate and display additional metrics
accuracy = np.trace(cm) / np.sum(cm)
misclass = 1 - accuracy

print("\nConfusion Matrix Metrics:")
print("-----------------------")
print(f"Accuracy: {accuracy:.2%}")
print(f"Misclassification Rate: {misclass:.2%}")

# Add these metrics to the metrics file
with open('model_metrics.txt', 'a') as f:
    f.write("\n\nConfusion Matrix Metrics:\n")
    f.write("-----------------------\n")
    f.write(f"Accuracy: {accuracy:.2%}\n")
    f.write(f"Misclassification Rate: {misclass:.2%}\n")

# Training History Plot
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# Print example predictions
print("\nExample Predictions:")
print("-------------------")
for i in range(5):
    true_class = actions[y_true_classes[i]]
    pred_class = actions[y_pred_classes[i]]
    confidence = y_pred[i][y_pred_classes[i]] * 100
    print(f"Sample {i+1}: True: {true_class}, Predicted: {pred_class}, Confidence: {confidence:.2f}%")

# Save model metrics to a text file
with open('model_metrics.txt', 'w') as f:
    f.write("Model Evaluation Metrics\n")
    f.write("=======================\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_true_classes, y_pred_classes, target_names=actions))
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

# After model training and evaluation, add optimized TFLite conversion
print("\nConverting model to TFLite with optimizations...")

# Save the Keras model first
model_path = 'gesture_model.h5'
model.save(model_path)
print(f"Saved Keras model to {model_path}")

# Configure TFLite conversion with optimizations
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Add configurations for LSTM support
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False
converter.experimental_enable_resource_variables = True

# Basic optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset for optimization
def representative_dataset():
    for i in range(min(100, len(X_train))):
        yield [np.expand_dims(X_train[i], axis=0).astype(np.float32)]

converter.representative_dataset = representative_dataset

# Convert the model
try:
    print("Converting model to TFLite...")
    tflite_model = converter.convert()
    
    # Save the TFLite model
    tflite_path = 'gesture_model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to {tflite_path}")
    
    # Compare model sizes
    tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
    keras_size = os.path.getsize(model_path) / (1024 * 1024)    # MB
    
    print(f"\nModel size comparison:")
    print(f"Keras model: {keras_size:.2f} MB")
    print(f"TFLite model: {tflite_size:.2f} MB")
    print(f"Size reduction: {((keras_size - tflite_size) / keras_size) * 100:.1f}%")
    
except Exception as e:
    print(f"Error during conversion: {e}")
    raise

# Add optimization details to metrics file
with open('model_metrics.txt', 'a') as f:
    f.write("\n\nTFLite Model Metrics:\n")
    f.write("-------------------\n")
    f.write(f"Keras model size: {keras_size:.2f} MB\n")
    f.write(f"TFLite model size: {tflite_size:.2f} MB\n")
    f.write(f"Size reduction: {((keras_size - tflite_size) / keras_size) * 100:.1f}%\n")