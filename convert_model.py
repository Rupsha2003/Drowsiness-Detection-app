# convert_model.py
import tensorflow as tf

# Load your existing Keras model
model = tf.keras.models.load_model('models/drowsiness_cnn_model.h5')

# Create a TFLite converter object
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Tell the converter to use certain optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the new, smaller TFLite model
with open('models/model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model successfully converted to model.tflite!")
