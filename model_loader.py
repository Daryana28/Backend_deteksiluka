import os
import gdown
import tensorflow as tf

def download_tflite_model():
    file_id = "1qwdvYQxKyEJIrJ6HdFjgnQM9bcmlR6Gv"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "model_deteksi_luka.tflite"
    if not os.path.exists(output):
        print("Mengunduh model .tflite dari Google Drive...")
        gdown.download(url, output, quiet=False)
        print("Model berhasil diunduh.")
    else:
        print("Model sudah tersedia secara lokal.")

def load_tflite_interpreter():
    download_tflite_model()
    interpreter = tf.lite.Interpreter(model_path="model_deteksi_luka.tflite")
    interpreter.allocate_tensors()
    return interpreter