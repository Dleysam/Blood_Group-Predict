from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import zipfile

app = Flask(__name__)

# 👉 Check if the SavedModel folder exists, else unzip it
if not os.path.exists('bloodgroup_savedmodel'):
    if os.path.exists('bloodgroup_savedmodel.zip'):
        print("Unzipping SavedModel...")
        with zipfile.ZipFile('bloodgroup_savedmodel.zip', 'r') as zip_ref:
            zip_ref.extractall('bloodgroup_savedmodel')
        print("Unzipping done.")
    else:
        raise FileNotFoundError("SavedModel folder and ZIP not found!")

# ✅ Load the SavedModel after ensuring it’s there
model = load_model('bloodgroup_savedmodel')

# Your classes
class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    img = Image.open(file).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]

    return jsonify({'predicted_blood_group': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
