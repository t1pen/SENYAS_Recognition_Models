import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("gesture_recognition_mlp.h5")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable resource variables
converter.experimental_enable_resource_variables = True

# Allow TensorFlow Select Ops (to support TensorArray operations)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Default TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS     # Allow TF operations that are not natively supported in TFLite
]

# Disable lowering tensor list ops
converter._experimental_lower_tensor_list_ops = False

# Convert the model
tflite_model = converter.convert()

# Save the model
with open("gesture_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully converted to TFLite!")
