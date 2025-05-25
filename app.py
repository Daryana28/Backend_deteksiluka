from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
from model_loader import load_tflite_interpreter

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({"status": "LukaLens backend siap!"})

# Load model TFLite dan label
interpreter = load_tflite_interpreter()
labels = ['Abrasions', 'Bruises', 'Burns', 'Cut', 'Ingrown_nails', 'Laceration', 'Stab_wound']

def predict_image_tflite(image_array, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.expand_dims(image_array, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Endpoint deteksi luka
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = image.resize((224, 224))

    arr = img_to_array(image) / 255.0

    pred = predict_image_tflite(arr, interpreter)
    idx = np.argmax(pred)
    confidence = float(np.max(pred))

    result = {
        "kelas_luka": labels[idx],
        "confidence": round(confidence, 4)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5050)