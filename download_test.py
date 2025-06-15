import requests
import os
from time import sleep

model_path = 'test_downloaded_model.h5'
url = 'https://drive.google.com/uc?export=download&id=1_OJId1A-UxYT4laacotfHA9g5U5KWMXa'
session = requests.Session()
response = session.get(url, stream=True)
token = None
for key, value in response.cookies.items():
    if 'download_warning' in value:
        token = value
        break
if token:
    params = {'confirm': token}
    response = session.get(url, params=params, stream=True)
with open(model_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=32768):
        if chunk:
            f.write(chunk)

import tensorflow as tf
model = tf.keras.models.load_model(model_path)
print("Model loaded from downloaded file!")