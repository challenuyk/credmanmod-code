#pyinstaller app.spec

from flask import Flask, request, jsonify, send_file
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

        # Apply improved preprocessing
        processed_image_path, processed_image_no_boxes_path = improved_preprocess_image(file_path)

        # Use pytesseract to extract text from the image without bounding boxes
        processed_image_no_boxes = Image.open(processed_image_no_boxes_path)
        extracted_text = pytesseract.image_to_string(processed_image_no_boxes)

        # Classify document based on the extracted text
        document_type = classify_document(extracted_text)

        # Return JSON with the document type and extracted text
        return jsonify({"document_type": document_type, "extracted_text": extracted_text})

def improved_preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter (reduce noise while preserving edges)
    bilateral_filtered_image = cv2.bilateralFilter(gray_image, 9, 75, 75)

    # Try Otsu's thresholding
    _, threshold_image = cv2.threshold(bilateral_filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to remove small noise and emphasize text
    kernel = np.ones((2, 2), np.uint8)
    morph_image = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel)

    # Save the processed image without bounding boxes for text extraction
    processed_image_no_boxes_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image_no_boxes.png')
    cv2.imwrite(processed_image_no_boxes_path, morph_image)

    # Detect words and draw bounding boxes
    d = pytesseract.image_to_data(morph_image, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(morph_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the processed image with bounding boxes for visualization
    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image_with_boxes.png')
    cv2.imwrite(processed_image_path, morph_image)

    return processed_image_path, processed_image_no_boxes_path

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
        return "1x1 ID Picture"
    elif sum(match_counts.values()) == 0:
        return "Unknown Document"
    else:
        return detected_type

if __name__ == '__main__':
    app.run(debug=True)