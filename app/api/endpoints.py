from fastapi import APIRouter, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

api_routes = APIRouter()

detector = tf.saved_model.load('./export/saved_model')

def detect_objects(path:str, model) -> dict:
  """Fungsi mengekstrak gambar dari file, menambahkan sumbu baru
     dan meneruskan gambar melalui model deteksi objek.
     : jalur param: Jalur file
     :param model: Model deteksi objek
     : kembali: Kamus keluaran model
     """
  image_tensor = tf.image.decode_jpeg(tf.io.read_file(path), channels= 3)[tf.newaxis, ...]
  return model(image_tensor)


def count_persons(path: str, model, threshold=0) -> int:
  """Fungsi menghitung jumlah orang dalam sebuah gambar
     memproses output "detection_classes" dari model
     dan dengan mempertimbangkan ambang kepercayaan.
     : jalur param: Jalur file
     :param model: Model deteksi objek
     : ambang batas param: Ambang untuk skor kepercayaan
     :return: Jumlah orang untuk satu gambar
     """
  results = detect_objects(path, model)
  # class 1D = 'person'
  return (results['detection_classes'].numpy()[0] == 1)[
      np.where(results['detection_scores'].numpy()[0] > threshold)].sum()


@api_routes.get('/hello')
async def ping():
    return "hello"


@api_routes.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    contents = await file.read()
    with open(file.filename, "wb") as f:
        f.write(contents)
    x = count_persons(file.filename, detector, threshold=0.4)
    return {"estimate": x.tolist()}


