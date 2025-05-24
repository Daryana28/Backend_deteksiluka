from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
from model_loader import load_skin_model

app = Flask(__name__)
CORS(app)

# Load model dan label
model = load_skin_model()
labels = ['Abrasions', 'Bruises', 'Burns', 'Cut', 'Ingrown_nails', 'Laceration', 'Stab_wound']

# Endpoint deteksi luka
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = image.resize((224, 224))

    arr = img_to_array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)
    idx = np.argmax(pred)
    confidence = float(np.max(pred))

    result = {
        "kelas_luka": labels[idx],
        "confidence": round(confidence, 4)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5050)