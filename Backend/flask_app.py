from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import os
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=r"C:\Users\Chathuni\Desktop\New Model\pepper_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['Healthy',
 'Lace Bug Infection',
 'Vine borer Infection',
 'Yellow Mottle Infection']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Preprocess image
    img = Image.open(file_path).resize((244, 244))  # adjust if needed
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = class_names[np.argmax(output_data)]

    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
