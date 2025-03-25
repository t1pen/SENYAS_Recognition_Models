import os
import numpy as np

# Define the directory containing the .npy files
DATA_DIR = "action_sequences"

# Dictionary to store extracted data
extracted_data = {}

# Process each action folder
for action in os.listdir(DATA_DIR):
    action_path = os.path.join(DATA_DIR, action)
    if not os.path.isdir(action_path):
        continue  # Skip non-directory files

    extracted_data[action] = []  # Initialize list for each action

    # Process each .npy file in the action folder
    for npy_file in os.listdir(action_path):
        if not npy_file.endswith(".npy"):
            continue  # Skip non-npy files

        npy_path = os.path.join(action_path, npy_file)
        sequence = np.load(npy_path, allow_pickle=True)  # Load the .npy file

        # Append the sequence to the action's data
        extracted_data[action].append(sequence)

        print(f"Loaded sequence from {npy_file} for action '{action}'")

# Example: Accessing data for a specific action
for action, sequences in extracted_data.items():
    print(f"Action: {action}, Number of Sequences: {len(sequences)}")
    for seq_idx, sequence in enumerate(sequences):
        print(f"  Sequence {seq_idx + 1}: {len(sequence)} frames")