import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("asl_mlp_model.h5")

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("asl_mlp_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully converted to TFLite and saved as 'asl_mlp_model_v2.tflite'")
