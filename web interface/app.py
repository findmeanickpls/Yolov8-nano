from flask import Flask, request, render_template, send_file  
from ultralytics import YOLO  
import cv2  
import numpy as np  
from pathlib import Path  
import os  
from werkzeug.utils import secure_filename  
  
app = Flask(__name__)  
  
BASE_DIR = Path(__file__).resolve().parent  
UPLOAD_FOLDER = BASE_DIR / "uploads"  
RESULT_FOLDER = BASE_DIR / "results"  
MODEL_PATH = BASE_DIR / "runs/detect/head_count_model_v3/weights/best.pt"  
  
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  
os.makedirs(RESULT_FOLDER, exist_ok=True)  
  
model = YOLO(MODEL_PATH)  
  
@app.route('/', methods=['GET', 'POST'])  
def index():  
    head_count = None  
    result_img = None  
    debug_info = []  
    if request.method == 'POST':  
        file = request.files['image']  
        filename = secure_filename(file.filename)  
        upload_path = UPLOAD_FOLDER / filename  
        file.save(str(upload_path))  
        debug_info.append("Saved uploaded image to: " + str(upload_path))  
  
        # Read image as numpy array (same as Jupyter)  
        img = cv2.imread(str(upload_path))
        if img is None:
                debug_info.append("Failed to load image with cv2.imread")
                return render_template('index.html', head_count=None, result_img=None, debug_info=debug_info)  
        debug_info.append("cv2.imread shape: " + str(img.shape) + ", dtype: " + str(img.dtype))  
  
        # Run detection with same parameters as Jupyter  
        results = model(img, imgsz=640, conf=0.1, iou=0.5)  
        result = results[0]  
        head_count = len(result.boxes)  
        debug_info.append("Detections found: " + str(head_count))
          
  
        # Draw boxes  
        annotated_img = result.plot()  
        result_filename = "result_" + filename  
        result_path = RESULT_FOLDER / result_filename  
        cv2.imwrite(str(result_path), annotated_img)  
        debug_info.append("Saved result image to: " + str(result_path))  
        result_img = result_filename  
  
    return render_template('index.html', head_count=head_count, result_img=result_img, debug_info=debug_info)  
  
@app.route('/result/<filename>')  
def result_file(filename):  
    return send_file(RESULT_FOLDER / filename, mimetype='image/jpeg')  
  
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)

