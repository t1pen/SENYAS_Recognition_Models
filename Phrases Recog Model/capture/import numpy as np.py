import numpy as np
import os

# Load a sample .npy file
file_path = os.path.join('MP_Data', 'hello', '0.npy')  # Replace with the actual path
data = np.load(file_path)

print(f"Shape of data: {data.shape}")  # Should be (sequence_length, 63)
print(f"First frame keypoints: {data[0]}")  # Keypoints for the first frame