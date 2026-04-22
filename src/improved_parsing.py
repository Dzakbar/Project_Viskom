"""
Improved parsing logic untuk OCR text
Lebih robust dan flexible terhadap berbagai format
"""
import re


# Improved nutrition patterns - lebih lenient
NUTRITION_PATTERNS = {
    "calories": [
        # Format: "Energi 140 kkal" atau "Calories 140"
        r"(?:energi|calories?|kalori)[:\s]+(\d+(?:[.,]\d+)?)\s*(?:kkal|kcal|cal)?",
        # Format dengan spasi di antara angka
        r"energi(?:\s+total)?\s+(\d+)\s*(?:kkal|kcal)",
        # Hanya angka dengan kkal
        r"(\d+)\s*(?:kkal|kcal)\b",
    ],
    "sugar": [
        # Format: "Gula 39g" atau "Sugar 39" atau "Total Sugars 39"
        r"(?:gula|total\s*sugars?|sugar)[:\s]+(\d+(?:[.,]\d+)?)\s*g?",
        # Format dengan kata jenis gula
        r"(?:total\s)?gula[:\s]+(\d+(?:[.,]\d+)?)",
        # Karbohydrat total sebagai fallback
        r"(?:karbohidrat|carbohydrate)[:\s]+(\d+(?:[.,]\d+)?)",
    ],
    "fat": [
        # Format: "Lemak 0g" atau "Total Fat 0"
        r"(?:lemak\s*(?:total)?|total\s*fat|fat)[:\s]+(\d+(?:[.,]\d+)?)\s*g?",
        # Lemak jenuh
        r"(?:total\s)?lemak[:\s]+(\d+(?:[.,]\d+)?)",
    ],
    "sodium": [
        # Format: "Natrium 45mg" atau "Sodium 45"
        r"(?:natrium|sodium|na)[:\s]+(\d+(?:[.,]\d+)?)\s*(?:mg|m9)?",
        # Format dengan kata lain
        r"(?:garam|salt)[:\s]+(\d+(?:[.,]\d+)?)",
    ],
}


def clean_ocr_text(text: str) -> str:
    """
    Clean dan normalize OCR text
    - Replace common OCR errors
    - Fix spacing issues
    """
    # Replace common OCR mistakes
    replacements = {
        '0O': '00',  # Zero confused with O
        'O': '0',    # Only in number context
        'l': '1',    # Lowercase L as 1
        'S': '5',    # S as 5
        'm9': 'mg',  # OCR error for mg
        '  ': ' ',   # Multiple spaces
    }
    
    # Smart replacement for O->0 only in number context
    text = re.sub(r'\bO(?=\d)', '0', text, flags=re.IGNORECASE)
    
    # Fix common spacing issues
    text = re.sub(r'(\d)\s+([a-z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    
    return text.strip()


def extract_nutrition_improved(text: str) -> dict:
    """
    Extract nutrition values dengan improved logic
    Lebih robust terhadap format yang berbeda
    """
    # Clean text first
    text_clean = clean_ocr_text(text)
    text_lower = text_clean.lower()
    
    nutrition = {
        "calories": 0.0,
        "sugar": 0.0,
        "fat": 0.0,
        "sodium": 0.0,
    }
    
    for nutrient, patterns in NUTRITION_PATTERNS.items():
        for pattern in patterns:
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            if matches:
                # Take the first match
                match = matches[0]
                try:
                    value_str = match.group(1).replace(",", ".")
                    value = float(value_str)
                    
                    # Sanity check - jangan ambil value yang terlalu besar atau kecil
                    if nutrient == "calories" and 0 <= value <= 500:
                        nutrition[nutrient] = value
                        break
                    elif nutrient == "sugar" and 0 <= value <= 100:
                        nutrition[nutrient] = value
                        break
                    elif nutrient == "fat" and 0 <= value <= 50:
                        nutrition[nutrient] = value
                        break
                    elif nutrient == "sodium" and 0 <= value <= 2000:
                        nutrition[nutrient] = value
                        break
                except (ValueError, AttributeError):
                    continue
    
    return nutrition


# Test function
if __name__ == "__main__":
    test_texts = [
        """INFORMASI NILAI Gizi JUMLAH PER KEMASAN (76ml) Energi Total 5 kkal
        Lemak Total % AKG* Lemak Jenuh 09 0% Protein 09 0 9 Karbohidrat Total
        19 1% Gula 19 09 Garam (Natrium) 19 1 35 m9""",
        
        "Calories: 140 kCal\nTotal Sugars: 39g\nTotal Fat: 0g\nSodium: 45mg",
        
        "Energi 140 kkal\nGula 39g\nLemak 0g\nNatrium 45mg",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n=== Test {i} ===")
        print(f"Input:\n{text[:100]}...")
        result = extract_nutrition_improved(text)
        print(f"Result: {result}")
