"""
Model Eğitim Script'i

Bu script modelin eğitimini başlatır ve sonuçları kaydeder.

Kullanım:
    python train.py
"""

import torch
import matplotlib.pyplot as plt
from data_loader import create_data_loaders
from model import create_model, ModelTrainer
import config


def plot_training_history(history: dict):
    """
    Eğitim geçmişini görselleştirir.
    
    Args:
        history (dict): Eğitim geçmişi verileri
        
    Time Complexity: O(n) - n: epoch sayısı
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss grafiği
    axes[0].plot(history['train_losses'], label='Eğitim Loss', marker='o')
    axes[0].plot(history['val_losses'], label='Validasyon Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Eğitim ve Validasyon Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy grafiği
    axes[1].plot(history['train_accuracies'], label='Eğitim Accuracy', marker='o')
    axes[1].plot(history['val_accuracies'], label='Validasyon Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Eğitim ve Validasyon Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Eğitim grafiği 'training_history.png' olarak kaydedildi.")
    plt.close()


def main():
    """
    Ana eğitim fonksiyonu.
    
    Time Complexity: O(e*n*m) - e: epoch, n: batch sayısı, m: batch_size
    """
    print("\n" + "="*60)
    print("ARAÇ SINIFLANDIRMA MODELİ EĞİTİMİ")
    print("="*60 + "\n")
    
    # Cihazı belirle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan Cihaz: {device}")
    
    if device.type == 'cuda':
        print(f"GPU Adı: {torch.cuda.get_device_name(0)}")
        print(f"GPU Belleği: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print("\n" + "-"*60)
    print("Veri Setleri Yükleniyor...")
    print("-"*60 + "\n")
    
    # Veri yükleyicilerini oluşturur
    train_loader, val_loader, test_loader = create_data_loaders()
    
    print("\n" + "-"*60)
    print("Model Oluşturuluyor...")
    print("-"*60 + "\n")
    
    # Modeli oluşturur
    model = create_model(pretrained=True)
    print(f"Model Parametreleri: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Eğitilebilir Parametreler: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Trainer oluşturur ve eğitir
    trainer = ModelTrainer(model, device)
    history = trainer.train(train_loader, val_loader)
    
    # Test seti üzerinde değerlendirir
    print("\n" + "="*60)
    print("Test Seti Değerlendirmesi")
    print("="*60 + "\n")
    
    test_loss, test_acc, test_metrics = trainer.evaluate(test_loader, 'Test')
    
    print(f"\nTest Sonuçları:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"\nSınıf Bazlı Accuracy:")
    for class_name, acc in test_metrics['class_accuracies'].items():
        print(f"  {class_name:20s}: {acc:.2f}%")
    
    # Eğitim grafiğini çizer
    plot_training_history(history)
    
    print("\n" + "="*60)
    print("EĞİTİM TAMAMLANDI!")
    print(f"Model kaydedildi: {config.MODEL_SAVE_PATH}")
    print(f"En iyi model kaydedildi: {config.BEST_MODEL_PATH}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

