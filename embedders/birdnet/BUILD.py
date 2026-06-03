"""Download BirdNET v2.4 FP32 TFLite model from Zenodo."""
import os
import shutil
import tempfile
import urllib.request
import zipfile

dir_embedder = os.path.dirname(os.path.realpath(__file__))
model_filename = 'audio-model.tflite'
model_path = os.path.join(dir_embedder, model_filename)

if os.path.exists(model_path):
    print(f'{model_filename} already exists; skipping download')
else:
    url = 'https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite.zip'
    print(f'Downloading from {url} ...')
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, 'birdnet.zip')
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extract(model_filename, tmpdir)
            shutil.move(os.path.join(tmpdir, model_filename), model_path)
    print(f'Model saved to {model_path}')
