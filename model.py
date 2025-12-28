"""
Model Tanımlama ve Eğitim Modülü

Bu modül CNN modelinin tanımlanması, eğitilmesi ve değerlendirilmesinden sorumludur.

Time Complexity:
- train_one_epoch: O(n*m) - n: batch sayısı, m: batch_size
- evaluate: O(n*m) - n: batch sayısı, m: batch_size
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List
import time
from tqdm import tqdm
import config
from torchvision import models


class VehicleClassifier(nn.Module):
    """
    Araç sınıflandırma için özel CNN modeli.
    Transfer learning ile ResNet18 kullanır.
    
    Attributes:
        model (nn.Module): ResNet18 tabanlı model
        num_classes (int): Sınıf sayısı
    """
    
    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        """
        VehicleClassifier sınıfını başlatır.
        
        Args:
            num_classes (int): Sınıf sayısı
            pretrained (bool): Önceden eğitilmiş ağırlıkları kullan
            
        Time Complexity: O(1)
        """
        super(VehicleClassifier, self).__init__()
        
        # ResNet18 modelini yükler (transfer learning)
        self.model = models.resnet18(pretrained=pretrained)
        
        # Son fully connected katmanını değiştirir
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Giriş tensörü (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Çıkış tensörü (batch_size, num_classes)
            
        Time Complexity: O(1) - Sabit ağ mimarisi
        """
        return self.model(x)


class ModelTrainer:
    """
    Model eğitimi ve değerlendirmesi için yardımcı sınıf.
    
    Attributes:
        model (nn.Module): Eğitilecek model
        device (torch.device): Hesaplama cihazı (CPU/GPU)
        criterion (nn.Module): Loss fonksiyonu
        optimizer (optim.Optimizer): Optimizer
        scheduler (optim.lr_scheduler): Learning rate scheduler
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        ModelTrainer sınıfını başlatır.
        
        Args:
            model (nn.Module): Eğitilecek model
            device (torch.device): Hesaplama cihazı
            
        Time Complexity: O(1)
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """
        Bir epoch için eğitim yapar.
        
        Args:
            train_loader (DataLoader): Eğitim veri yükleyici
            epoch (int): Mevcut epoch numarası
            
        Returns:
            Tuple[float, float]: (ortalama loss, accuracy)
            
        Time Complexity: O(n*m) - n: batch sayısı, m: forward/backward pass
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Eğitim]')
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # İstatistikleri günceller
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Progress bar günceller
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate(self, val_loader: DataLoader, phase: str = 'Validasyon') -> Tuple[float, float, Dict]:
        """
        Modeli değerlendirir.
        
        Args:
            val_loader (DataLoader): Validasyon/Test veri yükleyici
            phase (str): Değerlendirme fazı adı
            
        Returns:
            Tuple[float, float, Dict]: (loss, accuracy, metrics)
            
        Time Complexity: O(n*m) - n: batch sayısı, m: forward pass
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Sınıf bazlı metrikler için
        class_correct = [0] * config.NUM_CLASSES
        class_total = [0] * config.NUM_CLASSES
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'{phase}')
            
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # İstatistikleri günceller
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Sınıf bazlı doğruluk
                c = (predicted == labels)
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        # Sınıf bazlı accuracy hesaplar
        class_accuracies = {}
        for i in range(config.NUM_CLASSES):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                class_accuracies[config.CLASS_NAMES[i]] = acc
        
        metrics = {
            'overall_accuracy': epoch_acc,
            'class_accuracies': class_accuracies,
            'total_samples': total,
            'correct_predictions': correct
        }
        
        return epoch_loss, epoch_acc, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        Modeli tam eğitim süreciyle eğitir.
        
        Args:
            train_loader (DataLoader): Eğitim veri yükleyici
            val_loader (DataLoader): Validasyon veri yükleyici
            
        Returns:
            Dict: Eğitim geçmişi ve metrikleri
            
        Time Complexity: O(e*n*m) - e: epoch sayısı, n: batch sayısı, m: batch_size
        """
        print(f"\n{'='*60}")
        print(f"Model Eğitimi Başlıyor...")
        print(f"Cihaz: {self.device}")
        print(f"Toplam Epoch: {config.NUM_EPOCHS}")
        print(f"{'='*60}\n")
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(config.NUM_EPOCHS):
            # Eğitim
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validasyon
            val_loss, val_acc, _ = self.evaluate(val_loader, 'Validasyon')
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Sonuçları yazdır
            print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}:")
            print(f"  Eğitim   -> Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            print(f"  Validasyon -> Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
            
            # En iyi modeli kaydeder
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, config.BEST_MODEL_PATH)
                print(f"  ✓ En iyi model kaydedildi! (Accuracy: {val_acc:.2f}%)")
            
            print("-" * 60)
        
        # Eğitim süresi
        training_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Eğitim Tamamlandı!")
        print(f"Toplam Süre: {training_time/60:.2f} dakika")
        print(f"En İyi Validasyon Accuracy: {best_val_acc:.2f}%")
        print(f"{'='*60}\n")
        
        # Son modeli kaydeder
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, config.MODEL_SAVE_PATH)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': best_val_acc,
            'training_time': training_time
        }


def create_model(pretrained: bool = True) -> VehicleClassifier:
    """
    Yeni bir model örneği oluşturur.
    
    Args:
        pretrained (bool): Önceden eğitilmiş ağırlıkları kullan
        
    Returns:
        VehicleClassifier: Model örneği
        
    Time Complexity: O(1)
    """
    model = VehicleClassifier(num_classes=config.NUM_CLASSES, pretrained=pretrained)
    return model


def load_model(model_path: str, device: torch.device) -> VehicleClassifier:
    """
    Kaydedilmiş modeli yükler.
    
    Args:
        model_path (str): Model dosya yolu
        device (torch.device): Hesaplama cihazı
        
    Returns:
        VehicleClassifier: Yüklenmiş model
        
    Time Complexity: O(1)
    """
    model = create_model(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

