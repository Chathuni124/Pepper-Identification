from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model(r"C:\Users\Chathuni\Desktop\New Model\trained_model.keras")

class_names = ['Healthy', 'Lace Bug Infection', 'Vine Borer Infection', 'Yellow Mottle Infection']

# Route to handle image prediction
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        # Preprocess the image
        img_path = os.path.join("uploads", file.filename)
        file.save(img_path)

        img = image.load_img(img_path, target_size=(244, 244))
        img_array = image.img_to_array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)  

        # Predict
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        return jsonify({'prediction': predicted_class})

if __name__ == "__main__":
    app.run(debug=True)
