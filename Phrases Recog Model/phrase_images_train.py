import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Path to the .pkl file
FEATURES_PATH = os.path.join('features.pkl')

# Load features and labels
with open(FEATURES_PATH, 'rb') as f:  # Open the file in binary read mode
    data = pickle.load(f)  # Pass the file object to pickle.load()

features = data['features']
labels = data['labels']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define and train the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Save the trained model to a file
MODEL_PATH = 'mlp_model.pkl'
with open(MODEL_PATH, 'wb') as model_file:
    pickle.dump(mlp, model_file)
print(f"Trained model saved to {MODEL_PATH}")

# Evaluate the model
y_pred = mlp.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['hello', 'thanks', 'iloveyou', 'sorry']))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")