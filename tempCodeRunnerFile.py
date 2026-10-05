# app.py
import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from gradcam_utils import preprocess_img, make_gradcam_heatmap, overlay_heatmap_on_img
import numpy as np
import cv2

# Initialize Flask app
app = Flask(__name__, template_folder='.', static_folder='.')

# Load your trained model
MODEL_PATH = "chest_classifier.keras"
model = load_model(MODEL_PATH)

# Define last conv layer of EfficientNetB0
LAST_CONV_LAYER = "top_conv"

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    heatmap_img = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", prediction="No file selected")

        # Save uploaded file
        file_path = os.path.join("uploaded_image.jpg")
        file.save(file_path)

        # Preprocess and predict
        img_array, original_img = preprocess_img(file_path)
        preds = model.predict(img_array)
        class_idx = np.argmax(preds[0])
        class_names = ['COVID', 'LungCancer', 'NORMAL', 'PNEUMONIA', 'TB']
        prediction = class_names[class_idx]

        # Generate Grad-CAM
        heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER, pred_index=class_idx)
        overlay = overlay_heatmap_on_img(heatmap, original_img, alpha=0.4)
        heatmap_path = "static_heatmap.jpg"
        cv2.imwrite(heatmap_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        heatmap_img = heatmap_path

    return render_template("index.html", prediction=prediction, heatmap_img=heatmap_img)


if __name__ == "__main__":
    app.run(debug=True)
