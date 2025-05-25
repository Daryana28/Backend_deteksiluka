import os
import gdown
from tensorflow.keras.models import load_model

def download_model_from_drive():
    file_id = '103z0h6J-iAKJhk-tJ-r_04-PeD8PqPTR'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = "model_deteksi_luka.h5"

    if not os.path.exists(output):
        print("Mengunduh model dari Google Drive...")
        gdown.download(url, output, quiet=False)
        print("Model berhasil diunduh.")
    else:
        print("Model sudah tersedia secara lokal.")

def load_skin_model():
    download_model_from_drive()
    return load_model("model_deteksi_luka.h5")