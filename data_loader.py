"""
Veri Yükleme ve Ön İşleme Modülü

Bu modül görüntü verilerinin yüklenmesi, ön işlenmesi ve augmentation 
işlemlerinden sorumludur.

Time Complexity:
- load_and_preprocess_image: O(1) - Sabit boyutlu görüntü işleme
- create_data_loaders: O(n) - n: toplam görüntü sayısı
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, List, Dict
import config


class VehicleDataset(Dataset):
    """
    Araç görüntüleri için özel Dataset sınıfı.
    
    Attributes:
        data_dir (str): Veri setinin bulunduğu dizin
        transform (transforms.Compose): Uygulanacak transformasyonlar
        image_paths (List[str]): Görüntü dosya yolları
        labels (List[int]): Görüntülere karşılık gelen etiketler
        class_names (List[str]): Sınıf isimleri
    """
    
    def __init__(self, data_dir: str, transform=None):
        """
        VehicleDataset sınıfını başlatır.
        
        Args:
            data_dir (str): Veri setinin bulunduğu ana dizin
            transform (transforms.Compose, optional): Uygulanacak transformasyonlar
            
        Time Complexity: O(n*m) - n: sınıf sayısı, m: ortalama görüntü sayısı/sınıf
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = config.CLASS_NAMES
        
        # Veri setini yükler
        self._load_dataset()
    
    def _load_dataset(self):
        """
        Veri setindeki tüm görüntüleri ve etiketleri yükler.
        
        Time Complexity: O(n*m) - n: sınıf sayısı, m: ortalama dosya sayısı/sınıf
        """
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Uyarı: {class_dir} dizini bulunamadı!")
                continue
            
            # Desteklenen görüntü formatları
            valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
            
            for img_name in os.listdir(class_dir):
                # Dosya uzantısını kontrol eder
                if any(img_name.endswith(ext) for ext in valid_extensions):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
        
        print(f"Toplam {len(self.image_paths)} görüntü yüklendi.")
        print(f"Sınıf dağılımı: {self._get_class_distribution()}")
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """
        Sınıflara göre görüntü dağılımını hesaplar.
        
        Returns:
            Dict[str, int]: Sınıf adı ve görüntü sayısı eşleşmeleri
            
        Time Complexity: O(n) - n: toplam görüntü sayısı
        """
        distribution = {class_name: 0 for class_name in self.class_names}
        for label in self.labels:
            distribution[self.class_names[label]] += 1
        return distribution
    
    def __len__(self) -> int:
        """
        Veri setindeki toplam örnek sayısını döndürür.
        
        Returns:
            int: Toplam görüntü sayısı
            
        Time Complexity: O(1)
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Belirtilen indeksteki görüntüyü ve etiketini döndürür.
        
        Args:
            idx (int): Görüntü indeksi
            
        Returns:
            Tuple[torch.Tensor, int]: (görüntü tensörü, etiket)
            
        Time Complexity: O(1) - Sabit boyutlu görüntü işleme
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Görüntüyü yükler ve RGB'ye çevirir
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Hata: {img_path} yüklenemedi - {e}")
            # Boş bir görüntü döndür
            image = Image.new('RGB', config.IMAGE_SIZE, color='white')
        
        # Transformasyonları uygular
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_train_transforms() -> transforms.Compose:
    """
    Eğitim için kullanılacak veri augmentation ve normalizasyon işlemlerini döndürür.
    
    Returns:
        transforms.Compose: Eğitim transformasyonları
        
    Time Complexity: O(1)
    """
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomRotation(config.AUGMENTATION_CONFIG['rotation_range']),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])


def get_val_transforms() -> transforms.Compose:
    """
    Validasyon ve test için kullanılacak normalizasyon işlemlerini döndürür.
    
    Returns:
        transforms.Compose: Validasyon transformasyonları
        
    Time Complexity: O(1)
    """
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])


def create_data_loaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Eğitim, validasyon ve test veri yükleyicilerini oluşturur.
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
        
    Time Complexity: O(n) - n: toplam görüntü sayısı (veri setini yüklerken)
    """
    # Eğitim veri seti (augmentation ile)
    train_dataset = VehicleDataset(
        data_dir=str(config.DATA_DIR),
        transform=get_train_transforms()
    )
    
    # Validasyon ve test için veri seti (augmentation olmadan)
    val_test_dataset = VehicleDataset(
        data_dir=str(config.DATA_DIR),
        transform=get_val_transforms()
    )
    
    # Veri setini böler
    total_size = len(train_dataset)
    train_size = int(config.TRAIN_SPLIT * total_size)
    val_size = int(config.VAL_SPLIT * total_size)
    test_size = total_size - train_size - val_size
    
    # Random seed için generator
    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    
    # Eğitim setini ayırır
    train_dataset, _ = random_split(
        train_dataset, 
        [train_size, total_size - train_size],
        generator=generator
    )
    
    # Validasyon ve test setlerini ayırır
    _, val_test_subset = random_split(
        val_test_dataset,
        [train_size, total_size - train_size],
        generator=generator
    )
    
    val_dataset, test_dataset = random_split(
        val_test_subset,
        [val_size, test_size],
        generator=generator
    )
    
    # DataLoader'ları oluşturur
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows için 0 kullanıyoruz
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"\nVeri Setleri Hazır:")
    print(f"  - Eğitim: {len(train_dataset)} görüntü")
    print(f"  - Validasyon: {len(val_dataset)} görüntü")
    print(f"  - Test: {len(test_dataset)} görüntü")
    
    return train_loader, val_loader, test_loader


def preprocess_single_image(image_path: str) -> torch.Tensor:
    """
    Tek bir görüntüyü model için ön işler.
    
    Args:
        image_path (str): Görüntü dosya yolu
        
    Returns:
        torch.Tensor: Ön işlenmiş görüntü tensörü (1, 3, 224, 224)
        
    Time Complexity: O(1) - Sabit boyutlu görüntü işleme
    """
    transform = get_val_transforms()
    
    # Görüntüyü yükler
    image = Image.open(image_path).convert('RGB')
    
    # Transformasyonu uygular ve batch boyutu ekler
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor

