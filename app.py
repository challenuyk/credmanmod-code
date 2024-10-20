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

    # Denoise the image
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(denoised_image, 50, 150)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        # Find the intersection points of the lines to get the corners of the document
        points = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                points.append((x1, y1))
                points.append((x2, y2))

        # Use the convex hull to get the outermost points
        points = np.array(points)
        hull = cv2.convexHull(points)

        # Get the bounding box of the convex hull
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.int32(box)  # Use np.int32 instead of np.int0

        # Sort the points to get the top-left, top-right, bottom-right, and bottom-left points
        rect = np.zeros((4, 2), dtype="float32")
        s = box.sum(axis=1)
        rect[0] = box[np.argmin(s)]
        rect[2] = box[np.argmax(s)]

        diff = np.diff(box, axis=1)
        rect[1] = box[np.argmin(diff)]
        rect[3] = box[np.argmax(diff)]

        (tl, tr, br, bl) = rect

        # Compute the width and height of the new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Define the destination points for the perspective transform
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # Apply the perspective transform
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # Convert the warped image to grayscale
        gray_image = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    else:
        # If no lines are detected, use the original image
        gray_image = denoised_image

    # Sharpen the image
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened_image = cv2.filter2D(gray_image, -1, kernel_sharpen)

    # Apply Otsu's thresholding after denoising
    _, otsu_thresh_image = cv2.threshold(sharpened_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use adaptive thresholding for fine detail extraction
    adaptive_thresh_image = cv2.adaptiveThreshold(otsu_thresh_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 11, 2)

    # Apply morphological operations to clean up noise and isolate text
    kernel = np.ones((2, 2), np.uint8)
    morph_image = cv2.morphologyEx(adaptive_thresh_image, cv2.MORPH_CLOSE, kernel)

    # Save the processed image without bounding boxes for text extraction
    processed_image_no_boxes_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image_no_boxes.png')
    cv2.imwrite(processed_image_no_boxes_path, morph_image)

    # Convert the image to BGR format to draw colored bounding boxes
    morph_image_bgr = cv2.cvtColor(morph_image, cv2.COLOR_GRAY2BGR)

    # Detect words and draw bounding boxes
    d = pytesseract.image_to_data(morph_image, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(morph_image_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red color (BGR format)

    # Save the processed image with bounding boxes for visualization
    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image_with_boxes.png')
    cv2.imwrite(processed_image_path, morph_image_bgr)

    return processed_image_path, processed_image_no_boxes_path

def classify_document(text):
    # Convert text to lowercase for easier comparison
    text_lower = text.lower()

    # Define patterns for contextual matching
    patterns = {
        "Birth Certificate": [r"\bbirth certificate\b", r"\bcertificate of live birth\b"],
        "Enrollment Form": [r"\benrollment form\b", r"\bbasic education\b"],
        "Report Card": [r"\bform-137\b", r"\bpermanent record\b", r"\breport card\b", 
        r"\blearning areas\b", r"\bdepartment of education\b", r"\blearning progress\b", r"\breport\b"]
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