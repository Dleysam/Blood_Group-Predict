from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import zipfile

app = Flask(__name__)

# ✅ Unzip & auto-detect SavedModel
if not any('saved_model.pb' in files for _, _, files in os.walk(".")):
    if os.path.exists('bloodgroup_savedmodel.zip'):
        print("Unzipping SavedModel...")
        with zipfile.ZipFile('bloodgroup_savedmodel.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        print("Unzipping done.")
    else:
        raise FileNotFoundError("SavedModel ZIP not found!")

print("\n===> After Unzip, folders and files:")
for root, dirs, files in os.walk(".", topdown=True):
    print("ROOT:", root)
    for name in dirs:
        print("   DIR:", name)
    for name in files:
        print("   FILE:", name)

saved_model_dir = None
for root, dirs, files in os.walk("."):
    if 'saved_model.pb' in files:
        saved_model_dir = root
        break

if saved_model_dir is None:
    raise FileNotFoundError("❌ Could not find any folder containing saved_model.pb!")

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
    img = Image.open(file).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]

    return jsonify({'predicted_blood_group': predicted_class})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
