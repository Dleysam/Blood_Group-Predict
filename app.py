from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# 1. Define the path to your model folder
# Ensure your folder in GitHub is named 'bloodgroup_savedmodel'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'bloodgroup_savedmodel')

# 2. Load the model using the CPU-only method (better for Vercel)
try:
    # We use tf.keras.models.load_model for SavedModel folders
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded successfully from {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server'})
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
        
    file = request.files['file']
    
    try:
        # Preprocessing
        img = Image.open(file).convert('RGB').resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        
        return jsonify({'predicted_blood_group': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)})

# No app.run() needed for Vercel
print(f"✅ Using SavedModel folder: {saved_model_dir}")

model = load_model(saved_model_dir)

class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    img = Image.open(file).convert('RGB').resize((128, 128))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    return jsonify({'predicted_blood_group': predicted_class})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
