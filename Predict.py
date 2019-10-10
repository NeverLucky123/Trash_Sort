import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x, verbose=1)
    return preds[0]


MODEL_FILE = 'C:/Users/henry/PycharmProjects/TrashCan/venv/Include/filename.model'
img = image.load_img('C:/Users/henry/Desktop/MiniDataSet/Recycle/IMG_6185.jpg', target_size=(299, 299))
preds = predict(load_model(MODEL_FILE), img)
print(preds)


