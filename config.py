"""
Yapay Zeka Destekli Görüntü Sınıflandırıcı - Konfigürasyon Dosyası

Bu modül projenin tüm konfigürasyon ayarlarını içerir.
"""

import os
from pathlib import Path

# Proje Dizinleri
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Vehicles"
MODEL_DIR = BASE_DIR / "models"

# Model dizinini oluşturur
MODEL_DIR.mkdir(exist_ok=True)

# Veri Seti Ayarları
IMAGE_SIZE = (224, 224)  # ResNet ve diğer popüler modeller için standart boyut
BATCH_SIZE = 32
NUM_WORKERS = 4
RANDOM_SEED = 42

# Model Ayarları
NUM_CLASSES = 7
CLASS_NAMES = [
    "Auto Rickshaws",
    "Bikes", 
    "Cars",
    "Motorcycles",
    "Planes",
    "Ships",
    "Trains"
]

# Eğitim Ayarları
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Veri Augmentation Ayarları
AUGMENTATION_CONFIG = {
    "rotation_range": 20,
    "horizontal_flip": True,
    "brightness_range": (0.8, 1.2),
    "zoom_range": 0.2
}

# Model Kaydetme
MODEL_SAVE_PATH = MODEL_DIR / "vehicle_classifier.pth"
BEST_MODEL_PATH = MODEL_DIR / "best_vehicle_classifier.pth"

# Normalizasyon Değerleri (ImageNet ortalamaları)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

