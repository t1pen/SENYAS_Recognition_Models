import tensorflow as tf
from tensorflow import keras
from keras import layers
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset from corrected pickle file
pickle_file = "landmarks_dataset_v2.pkl"
with open(pickle_file, "rb") as f:
    data = pickle.load(f)

print(type(data))  # Check if it's a dictionary or something else
print(data.keys())  # Print the available keys

# Extract landmark columns (all columns except 'Label')
landmark_columns = [col for col in data.columns if col != "Label"]
X = data[landmark_columns].to_numpy()  # Convert to NumPy array

# Extract labels
y = np.array(data["Label"])  # Assuming 'Label' is the correct column name

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalize data
X = X.astype('float32') / np.max(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define an improved MLP model
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("asl_mlp_model_v2.h5")

# Save label encoder to pickle
with open("label_encoder_v2.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Model training complete and saved as asl_mlp_model.h5")
print("Label encoder saved as label_encoder.pkl")

