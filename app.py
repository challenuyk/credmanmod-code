#pyinstaller app.spec

from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
import cv2
import numpy as np
import os
import re

app = Flask(__name__)

# Specify the path to Tesseract executable for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Apply image preprocessing
        processed_image = softer_preprocess_image(file_path)

        # Use pytesseract to extract text
        extracted_text = pytesseract.image_to_string(processed_image)

        # Classify document based on the extracted text
        document_type = classify_document(extracted_text)

        # Return JSON with the document type
        return jsonify({"document_type": document_type})

def softer_preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image (less aggressive)
    scale_percent = 120  # Increase size by 120%
    width = int(gray_image.shape[1] * scale_percent / 100)
    height = int(gray_image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(gray_image, (width, height))

    # Apply slight sharpening (reduce intensity)
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    sharpened_image = cv2.filter2D(resized_image, -1, kernel)

    # Apply basic thresholding
    _, threshold_image = cv2.threshold(sharpened_image, 130, 255, cv2.THRESH_BINARY)

    # Convert back to PIL Image for Tesseract
    processed_pil_image = Image.fromarray(threshold_image)

    return processed_pil_image

def classify_document(text):
    # Convert text to lowercase for easier comparison
    text_lower = text.lower()

    # Define patterns for contextual matching
    patterns = {
        "Birth Certificate": [r"\bbirth certificate\b", r"\bcertificate of live birth\b"],
        "Enrollment Form": [r"\benrollment form\b", r"\bbasic education\b"],
        "Report Card": [r"\bform-137\b", r"\bpermanent record\b", r"\bschool\b"]
    }

    # Dictionary to store count of matches for each type
    match_counts = {
        "Birth Certificate": 0,
        "Enrollment Form": 0,
        "Report Card": 0
    }

    # Check for patterns in the text
    for document_type, regex_list in patterns.items():
        for regex in regex_list:
            if re.search(regex, text_lower):
                match_counts[document_type] += 1

    # Determine the document type based on the highest count of keyword matches
    detected_type = max(match_counts, key=match_counts.get)

    # Additional check if no keywords were detected (handling 2x2 ID Picture)
    if sum(match_counts.values()) == 0 and len(text.strip()) == 0:
        return "2x2 ID Picture"
    elif sum(match_counts.values()) == 0:
        return "Unknown Document"
    else:
        return detected_type

if __name__ == '__main__':
    app.run(debug=True)
