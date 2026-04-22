"""
Buat sample image label nutrisi untuk testing OCR
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_nutrition_label_image(output_path: str):
    """Buat gambar label nutrisi minuman"""
    
    # Buat image putih
    img = Image.new('RGB', (600, 800), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use Arial, fallback ke default jika tidak ada
    try:
        title_font = ImageFont.truetype("arial.ttf", 40)
        text_font = ImageFont.truetype("arial.ttf", 28)
        label_font = ImageFont.truetype("arial.ttf", 24)
    except:
        # Fallback ke default font
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # Title
    draw.text((50, 30), "NUTRITION FACTS", fill='black', font=title_font)
    
    # Garis pemisah
    draw.line([(50, 80), (550, 80)], fill='black', width=3)
    
    # Content
    y_pos = 110
    line_height = 50
    
    # Serving size
    draw.text((50, y_pos), "Serving Size: 250ml", fill='black', font=label_font)
    y_pos += line_height
    
    # Calories
    draw.line([(50, y_pos-10), (550, y_pos-10)], fill='black', width=1)
    draw.text((50, y_pos), "Calories: 140 kCal", fill='black', font=text_font)
    y_pos += line_height
    
    # Sugar
    draw.line([(50, y_pos-10), (550, y_pos-10)], fill='black', width=1)
    draw.text((50, y_pos), "Total Sugars: 39g", fill='black', font=text_font)
    y_pos += line_height
    
    # Fat
    draw.line([(50, y_pos-10), (550, y_pos-10)], fill='black', width=1)
    draw.text((50, y_pos), "Total Fat: 0g", fill='black', font=text_font)
    y_pos += line_height
    
    # Sodium
    draw.line([(50, y_pos-10), (550, y_pos-10)], fill='black', width=1)
    draw.text((50, y_pos), "Sodium: 45mg", fill='black', font=text_font)
    y_pos += line_height
    
    # Bottom line
    draw.line([(50, y_pos-10), (550, y_pos-10)], fill='black', width=3)
    
    # Save
    img.save(output_path)
    print(f"✅ Sample image dibuat: {output_path}")


if __name__ == "__main__":
    import os
    os.makedirs("test_images", exist_ok=True)
    create_nutrition_label_image("test_images/label_sample_cocacola.png")
    print("Gambar siap untuk OCR testing!")
