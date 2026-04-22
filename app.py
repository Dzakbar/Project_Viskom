"""
Flask Web App untuk Beverage Health Classifier
Simple UI untuk upload gambar dan lihat hasil klasifikasi
"""
from flask import Flask, render_template, request, jsonify
import os
import sys
import joblib
import numpy as np
import cv2
import easyocr
import re
from werkzeug.utils import secure_filename

# Import improved parsing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from improved_parsing import extract_nutrition_improved

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and scaler
LABEL_MAP = {0: "sehat", 1: "kurang sehat", 2: "tidak sehat"}
LABEL_REVERSE = {v: k for k, v in LABEL_MAP.items()}

model = joblib.load('models/health_model.pkl')
scaler = joblib.load('models/health_scaler.pkl')

# Initialize OCR reader (will be loaded on first use)
ocr_reader = None

_NUTRITION_PATTERNS = {
    "calories": [
        r"(?:kalori|calories?|energi|energy)[^\d]*(\d+(?:[.,]\d+)?)\s*(?:kkal|kcal|cal)?",
        r"(\d+(?:[.,]\d+)?)\s*(?:kkal|kcal)\b",
    ],
    "sugar": [
        r"(?:gula|sugars?)[^\d]*(\d+(?:[.,]\d+)?)\s*g",
        r"(?:total sugars?)[^\d]*(\d+(?:[.,]\d+)?)\s*g",
    ],
    "fat": [
        r"(?:lemak total|total fat|lemak)[^\d]*(\d+(?:[.,]\d+)?)\s*g",
        r"(?:fat)[^\d]*(\d+(?:[.,]\d+)?)\s*g",
    ],
    "sodium": [
        r"(?:natrium|sodium)[^\d]*(\d+(?:[.,]\d+)?)\s*mg",
        r"(?:na)[^\d]*(\d+(?:[.,]\d+)?)\s*mg",
    ],
}


def get_ocr_reader():
    """Lazy load OCR reader"""
    global ocr_reader
    if ocr_reader is None:
        ocr_reader = easyocr.Reader(['en', 'id'], gpu=False, verbose=False)
    return ocr_reader


def extract_text_ocr(image_path: str) -> str:
    """Extract text from image using OCR"""
    reader = get_ocr_reader()
    results = reader.readtext(image_path, detail=0, paragraph=True)
    full_text = "\n".join(results)
    return full_text


def parse_nutrition(ocr_text: str) -> dict:
    """Parse nutrition values from text using improved parsing"""
    return extract_nutrition_improved(ocr_text)


def predict_health(nutrition: dict) -> dict:
    """Predict beverage health category"""
    features = np.array([[
        nutrition.get("calories", 0.0),
        nutrition.get("sugar", 0.0),
        nutrition.get("fat", 0.0),
        nutrition.get("sodium", 0.0),
    ]])
    
    features_scaled = scaler.transform(features)
    pred_class = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]

    label = LABEL_MAP[pred_class]
    confidence = proba[pred_class] * 100

    probabilities = {LABEL_MAP[i]: round(p * 100, 2) for i, p in enumerate(proba)}

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "probabilities": probabilities,
    }


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Handle image upload and classification"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'File tidak dipilih'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Format file tidak didukung. Gunakan: PNG, JPG, JPEG, GIF, BMP'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        ocr_text = extract_text_ocr(filepath)
        nutrition = parse_nutrition(ocr_text)
        result = predict_health(nutrition)
        
        # Get icon based on label
        icons = {
            "sehat": "✅",
            "kurang sehat": "⚠️",
            "tidak sehat": "❌"
        }
        
        return jsonify({
            'success': True,
            'image': filename,
            'ocr_text': ocr_text,
            'nutrition': nutrition,
            'result': result,
            'icon': icons.get(result['label'], '❓'),
        })
    
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500


@app.route('/api/demo', methods=['GET'])
def demo():
    """Demo dengan nilai nutrisi manual"""
    demo_items = [
        {"nama": "Air Mineral", "kalori": 0, "gula": 0, "lemak": 0, "natrium": 5},
        {"nama": "Teh Botol", "kalori": 80, "gula": 20, "lemak": 0, "natrium": 0},
        {"nama": "Coca Cola", "kalori": 140, "gula": 39, "lemak": 0, "natrium": 45},
        {"nama": "Jus Apel", "kalori": 70, "gula": 16, "lemak": 0, "natrium": 8},
        {"nama": "Energi Drink", "kalori": 160, "gula": 37, "lemak": 0, "natrium": 350},
    ]
    
    results = []
    for item in demo_items:
        nutrition = {
            "calories": item["kalori"],
            "sugar": item["gula"],
            "fat": item["lemak"],
            "sodium": item["natrium"],
        }
        result = predict_health(nutrition)
        icons = {
            "sehat": "✅",
            "kurang sehat": "⚠️",
            "tidak sehat": "❌"
        }
        results.append({
            "nama": item["nama"],
            "nutrition": nutrition,
            "result": result,
            "icon": icons.get(result['label'], '❓'),
        })
    
    return jsonify({'demo_items': results})


if __name__ == '__main__':
    print("Starting Beverage Health Classifier Web App...")
    print("Open: http://localhost:5000")
    app.run(debug=True, host='localhost', port=5000)
