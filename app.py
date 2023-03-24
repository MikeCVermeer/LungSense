import PIL
from flask import Flask, render_template, request, jsonify
import base64
import io
import cv2
import numpy as np
from keras.models import load_model
from keras.backend import clear_session
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model_path = 'pneumonia_cnn.h5'
model = load_model(model_path)

app = Flask(__name__, static_folder='static', template_folder='templates')

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the POST request
    img_data = request.form['img_data']
    img_data = img_data[22:]  # Remove the 'data:image/png;base64,' part
    img_data = base64.b64decode(img_data)
    img = io.BytesIO(img_data)

    # Convert the image to a NumPy array
    img = np.asarray(PIL.Image.open(img))

    # Preprocess the image
    img = preprocess_image(img)

    # Make the prediction
    prediction = model.predict(img)
    clear_session()

    # Prepare the response
    result = {
        'normal': float(prediction[0][0]),
        'pneumonia': float(prediction[0][1]),
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)