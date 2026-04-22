"""
Demo dengan enhanced dataset dan improved parsing
"""

import sys
import os

# Tambah parent directory ke path agar bisa import modul
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Import improved parsing
from improved_parsing import extract_nutrition_improved


# Label encoding
LABEL_MAP = {0: "sehat", 1: "kurang sehat", 2: "tidak sehat"}
LABEL_REVERSE = {v: k for k, v in LABEL_MAP.items()}


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Muat dataset dari CSV"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"\n✅ Dataset dimuat: {csv_path}")
    print(f"   Total minuman: {len(df)}")
    print(f"\n   Distribusi kategori:")
    print(df["kategori"].value_counts().to_string())
    return df


def train_model(df: pd.DataFrame):
    """Training model Logistic Regression"""
    feature_cols = ["kalori", "gula", "lemak", "natrium"]
    
    # Normalize kategori column
    df["kategori"] = df["kategori"].str.replace("_", " ")
    
    # Rename kolom jika berbeda
    df = df.rename(columns={
        "kalori": "kalori",
        "gula": "gula",
        "lemak": "lemak",
        "natrium": "natrium",
        "kategori": "kategori"
    })
    
    X = df[feature_cols].values
    y = df["kategori"].map(LABEL_REVERSE).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluasi
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("  📊 EVALUASI MODEL TRAINING")
    print("=" * 60)
    print(f"  Akurasi: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
    ))
    print("=" * 60)

    return model, scaler


def predict_health(nutrition: dict, model, scaler) -> dict:
    """Prediksi dari nilai nutrisi"""
    features = np.array([[
        nutrition["kalori"],
        nutrition["gula"],
        nutrition["lemak"],
        nutrition["natrium"],
    ]])
    
    features_scaled = scaler.transform(features)
    pred_class = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]

    label = LABEL_MAP[pred_class]
    confidence = proba[pred_class] * 100

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "probabilities": {LABEL_MAP[i]: round(p * 100, 2) for i, p in enumerate(proba)}
    }


def main():
    print("\n" + "█" * 65)
    print("  BEVERAGE HEALTH CLASSIFIER - ENHANCED TRAINING")
    print("█" * 65)

    # Path
    csv_path = "data/training_data_enhanced.csv"
    model_path = "models/health_model.pkl"
    scaler_path = "models/health_scaler.pkl"

    # Pastikan folder models ada
    os.makedirs("models", exist_ok=True)

    # Load dataset
    df = load_dataset(csv_path)

    # Train model
    print("\nTraining model dengan dataset enhanced...")
    model, scaler = train_model(df)

    # Save model
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nModel disimpan ke: {model_path}")

    # Demo prediksi
    print("\n" + "=" * 65)
    print("  TEST PREDIKSI DENGAN CONTOH MINUMAN")
    print("=" * 65)

    contoh_minuman = [
        {"nama": "Air Mineral",           "kalori": 0,    "gula": 0,    "lemak": 0,   "natrium": 5},
        {"nama": "Teh Botol",             "kalori": 80,   "gula": 20,   "lemak": 0,   "natrium": 0},
        {"nama": "Coca Cola",             "kalori": 140,  "gula": 39,   "lemak": 0,   "natrium": 45},
        {"nama": "Jus Apel Natural",      "kalori": 70,   "gula": 16,   "lemak": 0,   "natrium": 8},
        {"nama": "Energy Drink",          "kalori": 160,  "gula": 37,   "lemak": 0,   "natrium": 350},
        {"nama": "Pocari Sweat",          "kalori": 120,  "gula": 28,   "lemak": 0,   "natrium": 300},
        {"nama": "Susu Coklat",           "kalori": 150,  "gula": 20,   "lemak": 3.5, "natrium": 100},
    ]

    icons = {"sehat": "OK", "kurang sehat": "WARN", "tidak sehat": "BAD"}

    print("\n{:<20} {:>7} {:>6} {:>6} {:>7}  {}".format(
        "Minuman", "Kalori", "Gula", "Lemak", "Natrium", "Hasil"
    ))
    print("-" * 75)

    for item in contoh_minuman:
        nama = item.pop("nama")
        result = predict_health(item, model, scaler)
        icon = icons[result["label"]]
        print(
            f"{nama:<20} {item['kalori']:>6.0f} {item['gula']:>6.1f} "
            f"{item['lemak']:>6.1f} {item['natrium']:>6.0f}  "
            f"{icon:15} ({result['confidence']:.0f}%)"
        )

    print("-" * 75)
    print("\nOK = SEHAT, WARN = KURANG SEHAT, BAD = TIDAK SEHAT")
    print("Training selesai!")


if __name__ == "__main__":
    main()
