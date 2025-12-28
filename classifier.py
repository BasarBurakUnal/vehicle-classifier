"""
Görüntü Sınıflandırma Modülü

Bu modül eğitilmiş modeli kullanarak yeni görüntüleri sınıflandırır.

Time Complexity:
- classify_image: O(1) - Sabit boyutlu görüntü ve forward pass
- classify_batch: O(n) - n: görüntü sayısı
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Tuple, List, Dict
import os
import config
from data_loader import preprocess_single_image
from model import load_model


class ImageClassifier:
    """
    Görüntü sınıflandırma için yardımcı sınıf.
    
    Attributes:
        model (nn.Module): Eğitilmiş model
        device (torch.device): Hesaplama cihazı
        class_names (List[str]): Sınıf isimleri
    """
    
    def __init__(self, model_path: str = None):
        """
        ImageClassifier sınıfını başlatır.
        
        Args:
            model_path (str, optional): Model dosya yolu. None ise en iyi model yüklenir.
            
        Time Complexity: O(1)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = config.CLASS_NAMES
        
        # Model yolu belirtilmemişse en iyi modeli yükler
        if model_path is None:
            model_path = str(config.BEST_MODEL_PATH)
        
        print(f"Model yükleniyor: {model_path}")
        self.model = load_model(model_path, self.device)
        print(f"Model başarıyla yüklendi! (Cihaz: {self.device})")
    
    def classify_image(self, image_input) -> Tuple[str, float, Dict[str, float]]:
        """
        Tek bir görüntüyü sınıflandırır.
        
        Args:
            image_input: PIL Image veya dosya yolu (str)
            
        Returns:
            Tuple[str, float, Dict[str, float]]: (tahmin edilen sınıf, güven skoru, tüm sınıf olasılıkları)
            
        Time Complexity: O(1) - Sabit boyutlu işlem
        """
        # Görüntüyü yükler ve ön işler
        temp_path = None
        try:
            if isinstance(image_input, str):
                image_tensor = preprocess_single_image(image_input)
            elif isinstance(image_input, Image.Image):
                # PIL Image ise geçici olarak kaydeder ve yükler
                # RGBA veya P modunu RGB'ye dönüştürür (JPEG şeffaflığı desteklemez)
                if image_input.mode in ('RGBA', 'LA', 'P'):
                    image_input = image_input.convert('RGB')
                
                temp_path = "temp_image.jpg"
                image_input.save(temp_path)
                image_tensor = preprocess_single_image(temp_path)
            else:
                raise ValueError("image_input PIL Image veya dosya yolu (str) olmalıdır!")
            
            # Cihaza taşır
            image_tensor = image_tensor.to(self.device)
            
            # Model ile tahmin yapar
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            # En yüksek olasılıklı sınıfı bulur
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_score = confidence.item()
            
            # Tüm sınıfların olasılıklarını hesaplar
            all_probabilities = {}
            for idx, class_name in enumerate(self.class_names):
                all_probabilities[class_name] = probabilities[0][idx].item()
            
            return predicted_class, confidence_score, all_probabilities
        
        finally:
            # Geçici dosyayı temizler
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass  # Silme hatası olsa bile devam eder
    
    def classify_with_top_k(self, image_input, k: int = 3) -> List[Tuple[str, float]]:
        """
        Görüntüyü sınıflandırır ve en yüksek k tahmini döndürür.
        
        Args:
            image_input: PIL Image veya dosya yolu (str)
            k (int): Döndürülecek tahmin sayısı
            
        Returns:
            List[Tuple[str, float]]: [(sınıf adı, güven skoru), ...]
            
        Time Complexity: O(k*log(n)) - n: sınıf sayısı
        """
        _, _, all_probabilities = self.classify_image(image_input)
        
        # Olasılıklara göre sıralar ve ilk k tanesini alır
        sorted_predictions = sorted(
            all_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        return sorted_predictions
    
    def classify_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Birden fazla görüntüyü sınıflandırır.
        
        Args:
            image_paths (List[str]): Görüntü dosya yolları listesi
            
        Returns:
            List[Dict]: Her görüntü için tahmin sonuçları
            
        Time Complexity: O(n) - n: görüntü sayısı
        """
        results = []
        
        for image_path in image_paths:
            try:
                predicted_class, confidence, all_probs = self.classify_image(image_path)
                results.append({
                    'image_path': image_path,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'all_probabilities': all_probs,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'success': False
                })
        
        return results


def classify_single_image(image_path: str, model_path: str = None) -> Dict:
    """
    Tek bir görüntüyü sınıflandırmak için yardımcı fonksiyon.
    
    Args:
        image_path (str): Görüntü dosya yolu
        model_path (str, optional): Model dosya yolu
        
    Returns:
        Dict: Tahmin sonuçları
        
    Time Complexity: O(1)
    """
    classifier = ImageClassifier(model_path)
    predicted_class, confidence, all_probs = classifier.classify_image(image_path)
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'confidence_percentage': f"{confidence * 100:.2f}%",
        'all_probabilities': {k: f"{v*100:.2f}%" for k, v in all_probs.items()}
    }


# Test için örnek kullanım
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Kullanım: python classifier.py <görüntü_yolu>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("\n" + "="*60)
    print("GÖRÜNTÜ SINIFLANDIRMA")
    print("="*60 + "\n")
    
    result = classify_single_image(image_path)
    
    print(f"Tahmin Edilen Sınıf: {result['predicted_class']}")
    print(f"Güven Skoru: {result['confidence_percentage']}")
    print("\nTüm Sınıf Olasılıkları:")
    
    for class_name, prob in sorted(
        result['all_probabilities'].items(),
        key=lambda x: float(x[1].rstrip('%')),
        reverse=True
    ):
        print(f"  {class_name:20s}: {prob}")
    
    print("\n" + "="*60)

