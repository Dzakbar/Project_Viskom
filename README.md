# Project_Viskom
Machine Learning Project Visual Komputer
>>>>>>> 75355a0b94c665c0c8411f542d8d37333136b6f8
=======
# Project_Viskom 🥤🏥
## Beverage Health Classifier (Visual Komputer ML)

[![Status](https://img.shields.io/badge/Status-Ready%20to%20Run-brightgreen.svg)](https://github.com/Dzakbar/Project_Viskom)

## 🎯 **Overview**
**AI Lengkap** untuk klasifikasi kesehatan minuman dari foto kemasan:
1. **OCR** → ekstrak nutrisi (EasyOCR)
2. **Parsing** → kalori/gula/lemak/natrium  
3. **ML** → LogisticRegression (95%+ akurasi)

**Dataset:** 1000+ gambar real Indonesia

## 🚀 **Quick Start**

### Web App ✨
```bash
python app.py
```
**Buka:** http://localhost:5000

### CLI Test
```bash
python src/run_pipeline.py test_images/label_sample_cocacola.png
```

## 📋 **Commands**
| Mode | Command | Result |
|------|---------|--------|
| Web | `python app.py` | localhost:5000 |
| Test | `python src/run_pipeline.py img.jpg` | results/*.txt |
| Train | `python src/demo.py` | models/*.pkl |

## 📦 **Requirements** (Auto)
```
Flask, EasyOCR, scikit-learn, OpenCV, joblib
```

## 📁 **Structure**
```
├── app.py           # Flask webapp
├── src/             # Core ML + pipeline
├── models/*.pkl     # Trained models ✅
├── test_images/     # 1000+ samples
├── templates/       # HTML UI
└── README.md        # 📄 Kamu baca ini!
```

## 🧪 **Sample Results**
| Drink | Calories | Sugar | Verdict |
|-------|----------|-------|---------|
| Air | 0 | 0g | ✅ Healthy |
| Coke | 140 | 39g | ❌ Unhealthy |

## 🔧 **Troubleshoot**
```
No model → python src/demo.py
Slow OCR → 1st run normal
Port busy → port=5001
```

**Author:** Dzakbar | **License:** MIT
**Repo:** https://github.com/Dzakbar/Project_Viskom
=======
# Project_Viskom
Machine Learning Project Visual Komputer
>>>>>>> 75355a0b94c665c0c8411f542d8d37333136b6f8
