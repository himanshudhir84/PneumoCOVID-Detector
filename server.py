from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Load model
model = tf.keras.models.load_model("chest_classifier.keras")

CLASS_NAMES = ["COVID", "Pneumonia", "TB", "LungCancer", "Normal"]

def preprocess_img(file_bytes):
    img = image.load_img(BytesIO(file_bytes), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

@app.route("/", methods=["GET"])
def home():
    return "Server is running"

@app.route("/scan-report", methods=["POST"])
def scan_report():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file_bytes = request.files["file"].read()
    img_array = preprocess_img(file_bytes)

    preds = model.predict(img_array)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100

    return jsonify({
        "prediction": CLASS_NAMES[idx],
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
