"""
Enhanced dataset dengan lebih banyak sampel minuman Indonesia
"""
import pandas as pd

data = {
    'nama_minuman': [
        # Minuman Ringan - SEHAT (Kalori rendah, gula rendah)
        'Air Mineral 600ml', 'Air Putih 1000ml', 'Aqua 600ml',
        'Air Minum Biasa', 'Aquades 500ml',
        
        # Juice & Smoothie Sehat - SEHAT/KURANG SEHAT
        'Jus Jeruk Segar 250ml', 'Jus Apel Natural 250ml', 'Jus Nanas Segar 250ml',
        'Smoothie Pisang 300ml', 'Jus Buah Campur 250ml',
        'Jus Semangka Segar 300ml', 'Jus Mangga Segar 250ml',
        'Jus Timun Segar 250ml',
        
        # Teh & Kopi - SEHAT/KURANG SEHAT
        'Teh Botol Sosro 350ml', 'Teh Pucuk 350ml', 'Teh Gelas 250ml',
        'Teh Tarik 300ml', 'Teh Hangat Biasa 250ml',
        'Kopi Hitam Tubruk 200ml', 'Kopi Espresso 30ml', 'Kopi Americano 250ml',
        'Kopi dengan Susu 250ml', 'Kopi Instan 200ml',
        
        # Minuman Bersusu - KURANG SEHAT/TIDAK SEHAT
        'Susu Putih 200ml', 'Susu Coklat 200ml', 'Susu Strawberry 200ml',
        'Susu Kental Manis Cair 200ml', 'Ultramilk 200ml',
        'Greenfield Yogurt 150ml', 'ABC Drinking Yogurt 200ml',
        'Milkshake Vanilla 300ml', 'Milkshake Coklat 300ml',
        'Frappucino 250ml',
        
        # Minuman Ringan Bergas - TIDAK SEHAT
        'Coca Cola 330ml', 'Coca Cola 250ml', 'Sprite 330ml', 'Sprite 250ml',
        'Fanta Orange 330ml', 'Fanta Strawberry 330ml', 'Fanta Anggur 330ml',
        'Pepsi 330ml', 'Pepsi 250ml', 'Seven Up 330ml',
        'Limca 330ml', 'Miranda 330ml', 'Marjan 200ml',
        'Jus Fruit 330ml', 'Tango 330ml',
        
        # Minuman Energi - TIDAK SEHAT
        'Red Bull 250ml', 'Red Bull 325ml', 'Monster Energy 355ml',
        'Power Energy 250ml', 'Krating Daeng 150ml', 'XL Energy 150ml',
        'Viterna 150ml', 'Pocari Sweat 500ml', 'Pocari Sweat 350ml',
        'Gatorade 500ml', 'Gatorade 350ml', 'Isotonic 500ml',
        
        # Jus Kemasan - KURANG SEHAT/TIDAK SEHAT
        'Tropicana Orange 250ml', 'Minute Maid Orange 250ml',
        'Cimory Juice 250ml', 'Nutri Juice 250ml',
        'Dolphin Juice 200ml', 'Floridina 250ml',
        'Sunpride Orange 1000ml', 'Sunkist Orange 250ml',
    ],
    'kalori': [
        # Air
        0, 0, 0, 0, 0,
        # Jus
        60, 70, 65, 100, 75, 80, 85, 30,
        # Teh & Kopi
        80, 90, 70, 100, 60, 5, 2, 10, 80, 50,
        # Susu
        130, 150, 140, 200, 140, 80, 100, 150, 160, 200,
        # Soda
        140, 110, 140, 110, 150, 140, 150, 140, 110, 120, 120, 130, 70, 150, 140,
        # Energi
        110, 160, 160, 130, 100, 120, 120, 120, 100, 130, 100, 100,
        # Jus kemasan
        70, 75, 80, 85, 60, 80, 120, 90,
    ],
    'gula': [
        # Air
        0, 0, 0, 0, 0,
        # Jus
        14, 16, 15, 18, 17, 16, 18, 7,
        # Teh & Kopi
        20, 22, 18, 24, 15, 0, 0, 0.5, 8, 5,
        # Susu
        11, 20, 18, 35, 14, 12, 15, 25, 28, 40,
        # Soda
        39, 30, 35, 27, 40, 36, 38, 38, 30, 32, 32, 34, 17, 40, 38,
        # Energi
        27, 38, 37, 30, 25, 28, 26, 28, 25, 31, 25, 25,
        # Jus kemasan
        16, 17, 18, 19, 14, 18, 28, 20,
    ],
    'lemak': [
        # Air
        0, 0, 0, 0, 0,
        # Jus
        0, 0, 0, 0.5, 0, 0, 0, 0,
        # Teh & Kopi
        0, 0, 0, 0, 0, 0.2, 0, 0, 3, 0.5,
        # Susu
        3.5, 3.5, 3.5, 5, 3.5, 0, 0, 3.5, 4, 8,
        # Soda
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # Energi
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # Jus kemasan
        0, 0, 0, 0, 0, 0, 0, 0,
    ],
    'natrium': [
        # Air
        5, 0, 5, 0, 0,
        # Jus
        10, 8, 15, 15, 20, 10, 12, 5,
        # Teh & Kopi
        0, 5, 0, 30, 0, 0, 0, 5, 50, 10,
        # Susu
        120, 100, 110, 200, 120, 100, 120, 150, 150, 200,
        # Soda
        45, 35, 75, 60, 50, 55, 60, 40, 35, 60, 55, 65, 30, 70, 65,
        # Energi
        200, 250, 350, 220, 150, 180, 160, 300, 280, 280, 250, 270,
        # Jus kemasan
        20, 25, 30, 25, 15, 20, 40, 30,
    ],
    'kategori': [
        # Air
        'sehat', 'sehat', 'sehat', 'sehat', 'sehat',
        # Jus
        'sehat', 'sehat', 'sehat', 'kurang_sehat', 'kurang_sehat', 'kurang_sehat', 'kurang_sehat', 'sehat',
        # Teh & Kopi
        'kurang_sehat', 'kurang_sehat', 'kurang_sehat', 'kurang_sehat', 'kurang_sehat', 'sehat', 'sehat', 'sehat', 'kurang_sehat', 'kurang_sehat',
        # Susu
        'sehat', 'kurang_sehat', 'kurang_sehat', 'tidak_sehat', 'kurang_sehat', 'kurang_sehat', 'kurang_sehat', 'tidak_sehat', 'tidak_sehat', 'tidak_sehat',
        # Soda
        'tidak_sehat', 'tidak_sehat', 'tidak_sehat', 'tidak_sehat', 'tidak_sehat', 'tidak_sehat', 'tidak_sehat', 'tidak_sehat', 'tidak_sehat', 'tidak_sehat', 'tidak_sehat', 'tidak_sehat', 'tidak_sehat', 'tidak_sehat', 'tidak_sehat',
        # Energi
        'tidak_sehat', 'tidak_sehat', 'tidak_sehat', 'tidak_sehat', 'tidak_sehat', 'tidak_sehat', 'tidak_sehat', 'kurang_sehat', 'kurang_sehat', 'kurang_sehat', 'kurang_sehat', 'kurang_sehat',
        # Jus kemasan
        'sehat', 'sehat', 'kurang_sehat', 'kurang_sehat', 'kurang_sehat', 'kurang_sehat', 'tidak_sehat', 'kurang_sehat',
    ]
}

df = pd.DataFrame(data)
df.to_csv('data/training_data_enhanced.csv', index=False)

print(f"Enhanced dataset created: {len(df)} minuman")
print("\nDistribusi kategori:")
print(df['kategori'].value_counts())
print("\nSample data:")
print(df.head(10))
