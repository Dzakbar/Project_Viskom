"""
======================================================
  Sistem Klasifikasi Kesehatan Minuman Kemasan
  Menggunakan OCR + Machine Learning
======================================================
Alur:
  1. Input gambar label nutrisi
  2. Preprocessing gambar (grayscale, thresholding)
  3. EasyOCR untuk ekstraksi teks
  4. Parsing nilai nutrisi (calories, sugar, fat, sodium)
  5. Feature vector dari hasil parsing
  6. Training model Logistic Regression dari CSV dataset
  7. Evaluasi akurasi model
  8. Prediksi klasifikasi dari OCR
  9. Output: sehat / kurang sehat / tidak sehat
"""

import cv2
import numpy as np
import pandas as pd
import easyocr
import re
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ─────────────────────────────────────────────
#  1. PREPROCESSING GAMBAR
# ─────────────────────────────────────────────

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Baca dan lakukan preprocessing gambar label nutrisi.

    Langkah:
      - Baca gambar asli
      - Konversi ke grayscale
      - Terapkan Gaussian blur untuk mengurangi noise
      - Adaptive thresholding untuk binarisasi
      - Morphological opening untuk membersihkan artefak kecil

    Returns:
        processed (np.ndarray): Gambar hasil preprocessing (grayscale binary)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur untuk mengurangi noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive thresholding agar teks lebih jelas di berbagai kondisi cahaya
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    # Morphological opening: hilangkan noise kecil
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    print(f"[Preprocessing] Selesai. Ukuran gambar: {processed.shape}")
    return processed


def enhance_for_ocr(image_path: str) -> np.ndarray:
    """
    Enhancement tambahan untuk meningkatkan akurasi OCR.
    Mengembalikan gambar yang di-upscale dan di-sharpen.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")

    # Upscale 2x agar teks lebih mudah dibaca OCR
    h, w = img.shape[:2]
    upscaled = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # Sharpening kernel
    sharpen_kernel = np.array([
        [0, -1,  0],
        [-1, 5, -1],
        [0, -1,  0]
    ])
    sharpened = cv2.filter2D(upscaled, -1, sharpen_kernel)
    return sharpened


# ─────────────────────────────────────────────
#  2. EKSTRAKSI TEKS DENGAN EASYOCR
# ─────────────────────────────────────────────

def extract_text_ocr(image_path: str, languages: list = None) -> str:
    """
    Gunakan EasyOCR untuk mengekstrak seluruh teks dari gambar.

    Args:
        image_path : path ke file gambar
        languages  : daftar bahasa (default: ['en', 'id'])

    Returns:
        full_text (str): Teks lengkap hasil OCR, digabung dengan newline
    """
    if languages is None:
        languages = ['en', 'id']

    print(f"[OCR] Memuat EasyOCR reader (bahasa: {languages})...")
    reader = easyocr.Reader(languages, gpu=False, verbose=False)

    # Gunakan gambar asli untuk OCR (EasyOCR bekerja lebih baik dengan RGB)
    print(f"[OCR] Memproses gambar: {image_path}")
    results = reader.readtext(image_path, detail=0, paragraph=True)

    full_text = "\n".join(results)
    print(f"[OCR] Teks berhasil diekstrak ({len(full_text)} karakter):\n")
    print("─" * 50)
    print(full_text)
    print("─" * 50)
    return full_text


# ─────────────────────────────────────────────
#  3. PARSING NILAI NUTRISI
# ─────────────────────────────────────────────

# Pola regex untuk masing-masing nutrien
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


def parse_nutrition(ocr_text: str) -> dict:
    """
    Parse teks OCR untuk mengambil nilai nutrisi utama.

    Args:
        ocr_text (str): Teks hasil OCR

    Returns:
        nutrition (dict): {"calories": float, "sugar": float,
                           "fat": float, "sodium": float}
                          Nilai 0.0 jika tidak ditemukan.
    """
    text_lower = ocr_text.lower()
    nutrition = {key: 0.0 for key in _NUTRITION_PATTERNS}

    for nutrient, patterns in _NUTRITION_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                # Ganti koma desimal dengan titik
                value_str = match.group(1).replace(",", ".")
                try:
                    nutrition[nutrient] = float(value_str)
                    break  # hentikan setelah pola pertama cocok
                except ValueError:
                    continue

    print(f"\n[Parsing] Nilai nutrisi hasil OCR:")
    for k, v in nutrition.items():
        unit = "mg" if k == "sodium" else ("kkal" if k == "calories" else "g")
        print(f"  {k:10s}: {v} {unit}")

    return nutrition


def nutrition_to_feature_vector(nutrition: dict) -> np.ndarray:
    """
    Ubah dict nutrisi menjadi feature vector numpy array.

    Urutan fitur: [calories, sugar, fat, sodium]

    Returns:
        features (np.ndarray): shape (1, 4)
    """
    features = np.array([[
        nutrition.get("calories", 0.0),
        nutrition.get("sugar", 0.0),
        nutrition.get("fat", 0.0),
        nutrition.get("sodium", 0.0),
    ]])
    return features


# ─────────────────────────────────────────────
#  4. DATASET & TRAINING MODEL
# ─────────────────────────────────────────────

# Label encoding
LABEL_MAP = {0: "sehat", 1: "kurang sehat", 2: "tidak sehat"}
LABEL_REVERSE = {v: k for k, v in LABEL_MAP.items()}


def generate_sample_dataset(csv_path: str = "nutrition_dataset.csv",
                             n_samples: int = 500) -> pd.DataFrame:
    """
    Generate dataset nutrisi sintetis jika file CSV belum ada.

    Kategori berdasarkan threshold per 250 ml serving:
      - sehat       : calories<80, sugar<5,  fat<2,  sodium<150
      - kurang sehat: calories<150, sugar<15, fat<7,  sodium<400
      - tidak sehat : lebih dari threshold kurang sehat

    Args:
        csv_path  : path untuk menyimpan CSV
        n_samples : jumlah sampel yang di-generate

    Returns:
        df (pd.DataFrame): Dataset dengan kolom nutrisi + label
    """
    np.random.seed(42)
    rng = np.random.default_rng(42)

    records = []
    per_class = n_samples // 3

    # Kelas 0 – sehat
    for _ in range(per_class):
        records.append({
            "calories": rng.uniform(10, 79),
            "sugar":    rng.uniform(0, 4.9),
            "fat":      rng.uniform(0, 1.9),
            "sodium":   rng.uniform(0, 149),
            "label":    "sehat",
        })

    # Kelas 1 – kurang sehat
    for _ in range(per_class):
        records.append({
            "calories": rng.uniform(80, 149),
            "sugar":    rng.uniform(5, 14.9),
            "fat":      rng.uniform(2, 6.9),
            "sodium":   rng.uniform(150, 399),
            "label":    "kurang sehat",
        })

    # Kelas 2 – tidak sehat
    for _ in range(n_samples - 2 * per_class):
        records.append({
            "calories": rng.uniform(150, 350),
            "sugar":    rng.uniform(15, 60),
            "fat":      rng.uniform(7, 25),
            "sodium":   rng.uniform(400, 900),
            "label":    "tidak sehat",
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"[Dataset] Dataset sintetis dibuat: {csv_path} ({len(df)} baris)")
    return df


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Muat dataset dari CSV.
    Kolom yang diperlukan: calories, sugar, fat, sodium, label

    Returns:
        df (pd.DataFrame)
    """
    if not os.path.exists(csv_path):
        print(f"[Dataset] File '{csv_path}' tidak ditemukan. Membuat dataset sintetis...")
        return generate_sample_dataset(csv_path)

    df = pd.read_csv(csv_path)
    required = {"calories", "sugar", "fat", "sodium", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom tidak lengkap di CSV: {missing}")

    print(f"[Dataset] Dataset dimuat: {csv_path} ({len(df)} baris)")
    print(df["label"].value_counts().to_string())
    return df


def train_model(df: pd.DataFrame):
    """
    Training model Logistic Regression dari DataFrame nutrisi.

    Langkah:
      - Encode label ke integer
      - Split train/test (80/20)
      - StandardScaler normalisasi fitur
      - Fit Logistic Regression
      - Evaluasi akurasi dan classification report

    Returns:
        model   (LogisticRegression): Model terlatih
        scaler  (StandardScaler)    : Scaler yang sudah di-fit
    """
    feature_cols = ["calories", "sugar", "fat", "sodium"]
    X = df[feature_cols].values
    y = df["label"].map(LABEL_REVERSE).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=500,
        multi_class="auto",
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # ── Evaluasi ──────────────────────────────
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("  EVALUASI MODEL")
    print("=" * 50)
    print(f"  Akurasi        : {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
        zero_division=0,
    ))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Aktual: {LABEL_MAP[i]}" for i in range(3)],
        columns=[f"Pred: {LABEL_MAP[i]}" for i in range(3)],
    )
    print(cm_df.to_string())
    print("=" * 50)

    return model, scaler


def save_model(model, scaler, model_path: str = "health_model.pkl",
               scaler_path: str = "health_scaler.pkl"):
    """Simpan model dan scaler ke disk menggunakan joblib."""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n[Model] Disimpan → {model_path}, {scaler_path}")


def load_model(model_path: str = "health_model.pkl",
               scaler_path: str = "health_scaler.pkl"):
    """Muat model dan scaler dari disk."""
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("File model/scaler tidak ditemukan. Jalankan training terlebih dahulu.")
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"[Model] Dimuat dari {model_path}")
    return model, scaler


# ─────────────────────────────────────────────
#  5. PREDIKSI
# ─────────────────────────────────────────────

def predict_health(nutrition: dict, model, scaler) -> dict:
    """
    Prediksi klasifikasi kesehatan berdasarkan nilai nutrisi.

    Args:
        nutrition (dict): {"calories", "sugar", "fat", "sodium"}
        model           : LogisticRegression terlatih
        scaler          : StandardScaler yang sudah di-fit

    Returns:
        result (dict): {
            "label"       : str,   # "sehat" / "kurang sehat" / "tidak sehat"
            "confidence"  : float, # probabilitas kelas prediksi (%)
            "probabilities": dict  # probabilitas semua kelas
        }
    """
    features = nutrition_to_feature_vector(nutrition)
    features_scaled = scaler.transform(features)

    pred_class = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]

    label = LABEL_MAP[pred_class]
    confidence = proba[pred_class] * 100

    probabilities = {LABEL_MAP[i]: round(p * 100, 2) for i, p in enumerate(proba)}

    return {
        "label":         label,
        "confidence":    round(confidence, 2),
        "probabilities": probabilities,
    }


def print_result(result: dict, nutrition: dict):
    """Cetak hasil klasifikasi secara terformat."""
    icons = {"sehat": "✅", "kurang sehat": "⚠️", "tidak sehat": "❌"}
    icon = icons.get(result["label"], "❓")

    print("\n" + "=" * 50)
    print("  HASIL KLASIFIKASI MINUMAN")
    print("=" * 50)
    print(f"  Status    : {icon}  {result['label'].upper()}")
    print(f"  Kepercayaan: {result['confidence']:.1f}%")
    print("\n  Nilai Nutrisi:")
    units = {"calories": "kkal", "sugar": "g", "fat": "g", "sodium": "mg"}
    for k, v in nutrition.items():
        print(f"    {k:10s}: {v} {units.get(k, '')}")
    print("\n  Probabilitas per Kelas:")
    for lbl, prob in result["probabilities"].items():
        bar = "█" * int(prob / 5)
        print(f"    {lbl:15s}: {prob:5.1f}%  {bar}")
    print("=" * 50)


# ─────────────────────────────────────────────
#  6. PIPELINE UTAMA
# ─────────────────────────────────────────────

def run_full_pipeline(image_path: str,
                      csv_path: str = "data/training_data.csv",
                      model_path: str = "models/health_model.pkl",
                      scaler_path: str = "models/health_scaler.pkl",
                      retrain: bool = False) -> dict:
    """
    Jalankan pipeline lengkap end-to-end:
      Gambar → Preprocessing → OCR → Parsing → Prediksi → Output

    Args:
        image_path  : Path ke gambar label nutrisi
        csv_path    : Path ke dataset CSV (dibuat otomatis jika tidak ada)
        model_path  : Path untuk menyimpan/memuat model
        scaler_path : Path untuk menyimpan/memuat scaler
        retrain     : Paksa retrain meski model sudah ada

    Returns:
        result (dict): Hasil klasifikasi
    """
    print("\n" + "█" * 55)
    print("  SISTEM KLASIFIKASI KESEHATAN MINUMAN KEMASAN")
    print("█" * 55)

    # ── Training / Load Model ────────────────
    if retrain or not os.path.exists(model_path):
        print("\n[Pipeline] Memulai training model...")
        df = load_dataset(csv_path)
        model, scaler = train_model(df)
        save_model(model, scaler, model_path, scaler_path)
    else:
        print("\n[Pipeline] Memuat model yang sudah ada...")
        model, scaler = load_model(model_path, scaler_path)

    # ── Preprocessing Gambar ─────────────────
    print(f"\n[Pipeline] Preprocessing gambar: {image_path}")
    _ = preprocess_image(image_path)   # buat gambar processed (opsional disimpan)

    # ── OCR ──────────────────────────────────
    print("\n[Pipeline] Menjalankan OCR...")
    ocr_text = extract_text_ocr(image_path)

    # ── Parsing Nutrisi ───────────────────────
    print("\n[Pipeline] Parsing nilai nutrisi...")
    nutrition = parse_nutrition(ocr_text)

    # ── Prediksi ─────────────────────────────
    print("\n[Pipeline] Melakukan prediksi klasifikasi...")
    result = predict_health(nutrition, model, scaler)
    print_result(result, nutrition)

    return result


# ─────────────────────────────────────────────
#  7. DEMO TANPA GAMBAR (manual input)
# ─────────────────────────────────────────────

def demo_manual_input():
    """
    Demo: klasifikasi dari nilai nutrisi yang diinput manual
    (berguna untuk testing tanpa kamera/gambar).
    """
    print("\n" + "─" * 55)
    print("  DEMO: Klasifikasi dari Input Manual")
    print("─" * 55)

    # Load / train model
    csv_path    = "data/training_data.csv"
    model_path  = "models/health_model.pkl"
    scaler_path = "models/health_scaler.pkl"

    df = load_dataset(csv_path)
    model, scaler = train_model(df)
    save_model(model, scaler, model_path, scaler_path)

    # Contoh minuman untuk diklasifikasikan
    contoh_minuman = [
        {"nama": "Air Mineral",      "calories": 0,   "sugar": 0,   "fat": 0,   "sodium": 5},
        {"nama": "Teh Manis Kemasan","calories": 120,  "sugar": 28,  "fat": 0,   "sodium": 25},
        {"nama": "Minuman Energi",   "calories": 220,  "sugar": 52,  "fat": 0.5, "sodium": 450},
        {"nama": "Jus Buah 100%",    "calories": 70,   "sugar": 14,  "fat": 0.2, "sodium": 10},
        {"nama": "Soda Manis",       "calories": 180,  "sugar": 44,  "fat": 0,   "sodium": 75},
    ]

    print("\n" + "=" * 65)
    print(f"  {'Minuman':<22} {'Kalori':>7} {'Gula':>6} {'Lemak':>6} {'Na':>6}  {'Hasil'}")
    print("─" * 65)

    for minuman in contoh_minuman:
        nama = minuman.pop("nama")
        result = predict_health(minuman, model, scaler)
        icons = {"sehat": "✅", "kurang sehat": "⚠️", "tidak sehat": "❌"}
        print(
            f"  {nama:<22} "
            f"{minuman['calories']:>6.0f} "
            f"{minuman['sugar']:>6.1f} "
            f"{minuman['fat']:>6.1f} "
            f"{minuman['sodium']:>6.0f}  "
            f"{icons[result['label']]} {result['label']} ({result['confidence']:.0f}%)"
        )

    print("=" * 65)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Mode pipeline penuh dengan gambar
        image_file = sys.argv[1]
        retrain_flag = "--retrain" in sys.argv
        run_full_pipeline(image_file, retrain=retrain_flag)
    else:
        # Mode demo tanpa gambar
        print("[INFO] Tidak ada argumen gambar. Menjalankan demo manual...\n")
        print("Penggunaan dengan gambar:")
        print("  python beverage_health_classifier.py label_gambar.jpg")
        print("  python beverage_health_classifier.py label_gambar.jpg --retrain\n")
        demo_manual_input()
