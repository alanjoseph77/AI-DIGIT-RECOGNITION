from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
import requests
from time import sleep

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def download_model(url, destination, max_retries=3):
    session = requests.Session()
    for attempt in range(max_retries):
        try:
            response = session.get(url, stream=True, timeout=30)
            token = None
            for key, value in response.cookies.items():
                if 'download_warning' in value:
                    token = value
                    break
            if token:
                params = {'confirm': token}
                response = session.get(url, params=params, stream=True, timeout=30)
            total_size = int(response.headers.get('content-length', -1))
            downloaded_size = 0
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        print(f"Downloaded {downloaded_size}/{total_size if total_size > 0 else 'unknown'} bytes")
            if total_size > 0 and downloaded_size != total_size:
                raise ValueError(f"Download incomplete: expected {total_size} bytes, got {downloaded_size} bytes")
            print(f"Model downloaded, size: {os.path.getsize(destination)} bytes")
            return
        except Exception as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                sleep(2 ** attempt)  # Exponential backoff
            else:
                raise

def load_model():
    model_path = 'digit_model.h5'
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        download_model('https://drive.google.com/uc?export=download&id=1_OJId1A-UxYT4laacotfHA9g5U5KWMXa', model_path)
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

model = load_model()

def predict_digit(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Error: Could not load image."
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    prediction = model.predict(img)
    digit = np.argmax(prediction) + 1
    return digit

@app.route('/')
def home():
    return render_template('digitreco.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        digit = predict_digit(file_path)
        return jsonify({'digit': str(digit)})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)