from flask import Flask, render_template, request, redirect, url_for
import pytesseract
from PIL import Image, ImageOps
import cv2
import numpy as np
import os

app = Flask(__name__)

# Specify the path to Tesseract executable for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Apply image preprocessing
        processed_image = preprocess_image(file_path)

        # Use pytesseract to extract text
        extracted_text = pytesseract.image_to_string(processed_image)
        
        return f"<h1>Extracted Text</h1><p>{extracted_text}</p>"

def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image (optional, adjust as needed)
    scale_percent = 150  # 150% of the original size
    width = int(gray_image.shape[1] * scale_percent / 100)
    height = int(gray_image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(gray_image, (width, height))

    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # Apply thresholding to make text stand out
    _, threshold_image = cv2.threshold(blurred_image, 150, 255, cv2.THRESH_BINARY)

    # Convert back to PIL Image for Tesseract
    processed_pil_image = Image.fromarray(threshold_image)

    return processed_pil_image

if __name__ == '__main__':
    app.run(debug=True)
