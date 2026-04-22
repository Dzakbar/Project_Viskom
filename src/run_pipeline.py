"""
Pipeline lengkap OCR + ML untuk testing
Versi simplified dengan better error handling
"""
import sys
import os

# Tambah path
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import pandas as pd
import easyocr
import re
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


LABEL_MAP = {0: "sehat", 1: "kurang sehat", 2: "tidak sehat"}
LABEL_REVERSE = {v: k for k, v in LABEL_MAP.items()}

# Nutrition patterns
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


def extract_text_ocr(image_path: str) -> str:
    """Extract text using EasyOCR"""
    print(f"[OCR] Loading EasyOCR reader (EN, ID)...")
    reader = easyocr.Reader(['en', 'id'], gpu=False, verbose=False)
    
    print(f"[OCR] Processing: {image_path}")
    results = reader.readtext(image_path, detail=0, paragraph=True)
    full_text = "\n".join(results)
    
    print(f"[OCR] Text extracted ({len(full_text)} chars):")
    print("─" * 50)
    print(full_text)
    print("─" * 50)
    return full_text


def parse_nutrition(ocr_text: str) -> dict:
    """Parse nutrition values from OCR text"""
    text_lower = ocr_text.lower()
    nutrition = {key: 0.0 for key in _NUTRITION_PATTERNS}

    for nutrient, patterns in _NUTRITION_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                value_str = match.group(1).replace(",", ".")
                try:
                    nutrition[nutrient] = float(value_str)
                    break
                except ValueError:
                    continue

    print(f"\n[Parsing] Extracted nutrition values:")
    for k, v in nutrition.items():
        unit = "mg" if k == "sodium" else ("kkal" if k == "calories" else "g")
        print(f"  {k:10s}: {v} {unit}")

    return nutrition


def nutrition_to_feature_vector(nutrition: dict) -> np.ndarray:
    """Convert nutrition dict to feature vector"""
    features = np.array([[
        nutrition.get("calories", 0.0),
        nutrition.get("sugar", 0.0),
        nutrition.get("fat", 0.0),
        nutrition.get("sodium", 0.0),
    ]])
    return features


def predict_health(nutrition: dict, model, scaler) -> dict:
    """Predict beverage health category"""
    features = nutrition_to_feature_vector(nutrition)
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


def print_result(result: dict, nutrition: dict):
    """Print classification result"""
    icons = {"sehat": "✅", "kurang sehat": "⚠️", "tidak sehat": "❌"}
    icon = icons.get(result["label"], "❓")

    print("\n" + "=" * 60)
    print("  🏥 HASIL KLASIFIKASI MINUMAN")
    print("=" * 60)
    print(f"  Status       : {icon}  {result['label'].upper()}")
    print(f"  Kepercayaan  : {result['confidence']:.1f}%")
    print("\n  Nilai Nutrisi:")
    units = {"calories": "kkal", "sugar": "g", "fat": "g", "sodium": "mg"}
    for k, v in nutrition.items():
        print(f"    {k:10s}: {v} {units.get(k, '')}")
    print("\n  Probabilitas per Kategori:")
    for lbl, prob in result["probabilities"].items():
        bar = "█" * int(prob / 5)
        print(f"    {lbl:15s}: {prob:5.1f}%  {bar}")
    print("=" * 60)


def main(image_path: str):
    """Main pipeline: Image -> OCR -> Parse -> Predict"""
    
    print("\n" + "█" * 65)
    print("  🥤  FULL PIPELINE: OCR + CLASSIFICATION")
    print("█" * 65)
    
    # Check image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image tidak ditemukan: {image_path}")
        return
    
    # Load model
    model_path = "models/health_model.pkl"
    scaler_path = "models/health_scaler.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model tidak ditemukan. Jalankan demo.py terlebih dahulu.")
        return
    
    print(f"\n[Model] Loading: {model_path}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Model loaded")
    
    # OCR
    print(f"\n[Pipeline] Step 1: OCR Extraction")
    ocr_text = extract_text_ocr(image_path)
    
    # Parse
    print(f"\n[Pipeline] Step 2: Parsing Nutrition")
    nutrition = parse_nutrition(ocr_text)
    
    # Predict
    print(f"\n[Pipeline] Step 3: Classification")
    result = predict_health(nutrition, model, scaler)
    print_result(result, nutrition)
    
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Simpan hasil prediksi
    output_file = os.path.join(results_dir, "prediction_result.txt")
    with open(output_file, "w") as f:
        f.write("HASIL PREDIKSI KLASIFIKASI MINUMAN\n")
        f.write("=" * 60 + "\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Status: {result['label']}\n")
        f.write(f"Confidence: {result['confidence']}%\n")
        f.write(f"\nNutrisi:\n")
        for k, v in nutrition.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nProbabilities:\n")
        for lbl, prob in result["probabilities"].items():
            f.write(f"  {lbl}: {prob}%\n")
    
    print(f"\n✅ Hasil disimpan ke: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    main(image_path)
