from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
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

def predict_digit(image_path):
    print(f"Attempting to load image from {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image from {image_path}")
        return "Error: Could not load image."
    print("Image loaded, resizing to 28x28")
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
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        print(f"Saving file to {file_path}")
        file.save(file_path)
        digit = predict_digit(file_path)
        if isinstance(digit, str) and digit.startswith("Error"):
            return jsonify({'error': digit}), 500
        return jsonify({'digit': str(digit)})

if __name__ == '__main__':
    app.run(debug=True)