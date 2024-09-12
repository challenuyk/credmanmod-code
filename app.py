from flask import Flask, render_template, request, redirect, url_for
import pytesseract
from PIL import Image
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
        
        # Open image using PIL
        image = Image.open(file_path)
        
        # Use pytesseract to extract text
        extracted_text = pytesseract.image_to_string(image)
        
        return f"<h1>Extracted Text</h1><p>{extracted_text}</p>"

if __name__ == '__main__':
    app.run(debug=True)
