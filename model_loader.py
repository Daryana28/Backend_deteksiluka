import os
import requests
from tensorflow.keras.models import load_model

def download_model_from_drive():
    file_id = '103z0h6J-iAKJhk-tJ-r_04-PeD8PqPTR'
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    print("Mengunduh model dari Google Drive...")
    response = requests.get(url)
    with open("model_deteksi_luka.h5", "wb") as f:
        f.write(response.content)
    print("Model berhasil diunduh.")

def load_skin_model():
    if not os.path.exists("model_deteksi_luka.h5"):
        download_model_from_drive()
    model = load_model("model_deteksi_luka.h5")
    return model