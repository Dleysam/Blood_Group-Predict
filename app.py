from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# ✅ Vercel-friendly Pathing
# This looks for the folder 'bloodgroup_savedmodel' in your main directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'bloodgroup_savedmodel')

# ✅ Global model variable
model = None

def load_my_model():
    global model
    if model is None:
        try:
            model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Model load failed: {e}")

# Class names for your specific predictor
class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    load_my_model() # Ensure model is loaded before predicting
    
    if model is None:
        return jsonify({'error': 'Model could not be initialized on the server.'})
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
        
    file = request.files['file']
    
    try:
        # 1. Open and resize to the 128x128 your model expects
        img = Image.open(file).convert('RGB').resize((128, 128))
        
        # 2. Convert to numpy array and normalize (0 to 1)
        img_array = np.array(img) / 255.0
        
        # 3. Add batch dimension (1, 128, 128, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 4. Run prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        
        return jsonify({
            'predicted_blood_group': predicted_class,
            'confidence': float(np.max(predictions[0]))
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# No app.run() needed for Vercel deployment
