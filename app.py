from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from io import BytesIO

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'  # Kept for potential future use, but not used
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created {UPLOAD_FOLDER}")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

try:
    model = tf.keras.models.load_model('digit_model.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def predict_digit(file_object):
    print("Attempting to process image from file object")
    # Read image from file object in memory
    img_bytes = file_object.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to decode image")
        return "Error: Could not decode image."
    print("Image decoded, resizing to 28x28")
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    print("Predicting digit...")
    try:
        prediction = model.predict(img)
        digit = np.argmax(prediction) + 1
        print(f"Predicted digit: {digit}")
        return digit
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error: Prediction failed."

@app.route('/')
def home():
    return render_template('digitreco.html')

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Use .png, .jpg, or .jpeg'}), 400
    if file:
        print(f"Processing file: {file.filename}")
        digit = predict_digit(file)
        if isinstance(digit, str) and digit.startswith("Error"):
            return jsonify({'error': digit}), 500
        return jsonify({'digit': str(digit)})

if __name__ == '__main__':
    app.run(debug=True)