import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO

MODEL_PATH = "chest_classifier.keras"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["COVID", "Pneumonia", "TB", "LungCancer", "Normal"]

def preprocess_img(file, target_size=(224, 224)):
    """
    Preprocess uploaded image file for prediction
    """
    if hasattr(file, "read"):
        file_bytes = BytesIO(file.read())
    else:
        file_bytes = file

    img = image.load_img(file_bytes, target_size=target_size, color_mode="rgb")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def predict_disease(file):
    """
    Predict disease from uploaded image
    """
    img_array = preprocess_img(file)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds)) * 100 
    return CLASS_NAMES[class_idx], confidence
