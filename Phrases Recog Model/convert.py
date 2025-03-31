import tensorflow as tf
import os

def convert_to_tflite(model_path, output_path):
    """Convert H5 model to TFLite format with optimizations"""
    try:
        # Load the best model checkpoint
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        
        # Configure TFLite conversion
        print("Configuring TFLite converter...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Enable resource variables
        converter.experimental_enable_resource_variables = True
        
        # Configure supported operations
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        # Disable tensor list ops lowering
        converter._experimental_lower_tensor_list_ops = False
        
        # Convert model
        print("Converting model to TFLite format...")
        tflite_model = converter.convert()
        
        # Save the converted model
        print(f"Saving TFLite model to {output_path}...")
        with open(output_path, "wb") as f:
            f.write(tflite_model)
            
        # Get file sizes for comparison
        h5_size = os.path.getsize(model_path) / (1024 * 1024)
        tflite_size = os.path.getsize(output_path) / (1024 * 1024)
        
        print("\nConversion completed successfully!")
        print(f"Original model size: {h5_size:.2f} MB")
        print(f"TFLite model size: {tflite_size:.2f} MB")
        print(f"Size reduction: {((h5_size - tflite_size) / h5_size * 100):.1f}%")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        raise

if __name__ == "__main__":
    model_path = "gesture_model_best.h5"
    output_path = "gesture_model_final.tflite"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    convert_to_tflite(model_path, output_path)
