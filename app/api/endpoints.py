from fastapi import APIRouter, File, UploadFile
import numpy as np
import tensorflow as tf

api_routes = APIRouter()

detector = tf.saved_model.load('./export/saved_model')

def detect_objects(img, model) -> dict:
  """Fungsi mengekstrak gambar dari file, menambahkan sumbu baru
     dan meneruskan gambar melalui model deteksi objek.
     : jalur param: Jalur file
     :param model: Model deteksi objek
     : kembali: Kamus keluaran model
     """
  image_tensor = tf.image.decode_jpeg(img, channels= 3)[tf.newaxis, ...]
  return model(image_tensor)


def count_persons(path, model, threshold=0) -> int:
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


@api_routes.get('/ping')
async def ping():
    return "hello"


@api_routes.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    contents = await file.read()
    x = count_persons(contents, detector, threshold=0.4)
    return {"estimate": x.tolist()}


