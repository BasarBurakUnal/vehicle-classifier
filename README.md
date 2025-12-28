# 🚗 Yapay Zeka Destekli Araç Sınıflandırıcı

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)


Derin öğrenme teknikleri kullanarak araç görüntülerini 7 farklı kategoriye sınıflandıran yapay zeka uygulaması.

## 📋 İçindekiler

- [🎯 Proje Hakkında](#-proje-hakkında)
  - [Sınıflandırılabilen Araç Tipleri](#sınıflandırılabilen-araç-tipleri)
- [✨ Özellikler](#-özellikler)
- [📊 Veri Seti](#-veri-seti)
  - [Veri Seti Kaynağı](#-veri-seti-kaynağı)
  - [Veri Seti İstatistikleri](#-veri-seti-istatistikleri)
  - [Veri Seti Yapısı](#-veri-seti-yapısı)
- [🚀 Kurulum](#-kurulum)
  - [Gereksinimler](#gereksinimler)
  - [Kurulum Adımları](#adım-1-projeyi-indirin)
- [💻 Kullanım](#-kullanım)
  - [Model Eğitimi](#1-model-eğitimi)
  - [Web Arayüzü](#2-web-arayüzünü-başlatın)
  - [Komut Satırı Kullanımı](#3-komut-satırından-tek-görüntü-tahmini)
- [📁 Proje Yapısı](#-proje-yapısı)
- [🧠 Model Detayları](#-model-detayları)
  - [Mimari](#mimari)
  - [Eğitim Parametreleri](#eğitim-parametreleri)
  - [Veri Augmentation](#veri-augmentation)
- [📊 Performans Metrikleri](#-performans-metrikleri)
  - [Örnek Sonuçlar](#örnek-sonuçlar)
- [📚 Kullanılan Teknolojiler](#-kullanılan-teknolojiler)


## 🎯 Proje Hakkında

Bu proje, makine öğrenimi tekniklerini kullanarak araç görüntülerini otomatik olarak sınıflandıran bir yapay zeka uygulamasıdır. ResNet18 mimarisi üzerine inşa edilmiş ve transfer learning yöntemiyle eğitilmiştir.

### Sınıflandırılabilen Araç Tipleri

1. 🛺 Auto Rickshaws (Oto Rikşalar)
2. 🚲 Bikes (Bisikletler)
3. 🚗 Cars (Arabalar)
4. 🏍️ Motorcycles (Motosikletler)
5. ✈️ Planes (Uçaklar)
6. 🚢 Ships (Gemiler)
7. 🚆 Trains (Trenler)

## ✨ Özellikler

- **Yüksek Doğruluk:** Transfer learning ile optimize edilmiş model
- **Kullanıcı Dostu Arayüz:** Modern ve responsive Streamlit web arayüzü
- **Gerçek Zamanlı Tahmin:** Anında görüntü sınıflandırma
- **Detaylı Analiz:** Tüm sınıflar için olasılık dağılımı
- **Görselleştirme:** İnteraktif grafikler ve metrikler
- **Modüler Kod Yapısı:** Temiz, dokümante edilmiş ve genişletilebilir kod

## 📊 Veri Seti

Bu projede **Vehicle Classification Dataset** kullanılmıştır. Veri seti Kaggle platformundan temin edilmiştir.

### 🔗 Veri Seti Kaynağı

**Kaggle:** [Vehicle Classification Dataset](https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification)

### 📈 Veri Seti İstatistikleri

- **Toplam Görüntü Sayısı:** ~5,600 adet
- **Sınıf Sayısı:** 7 farklı araç kategorisi
- **Görüntü Formatları:** JPG, JPEG, PNG
- **Sınıf Dağılımı:**
  - Auto Rickshaws: ~800 görüntü
  - Bikes: ~800 görüntü
  - Cars: ~790 görüntü
  - Motorcycles: ~800 görüntü
  - Planes: ~800 görüntü
  - Ships: ~800 görüntü
  - Trains: ~800 görüntü

### 📂 Veri Seti Yapısı

Veri seti dengeli bir dağılıma sahiptir ve her sınıf için yeterli sayıda örnek içermektedir. Bu, modelin tüm araç tiplerini eşit şekilde öğrenmesini sağlar.

## 🚀 Kurulum

### Gereksinimler

- Python 3.8 veya üzeri
- pip paket yöneticisi

### Adım 1: Projeyi İndirin

```bash
git clone <repository-url>
cd sektor
```

### Adım 2: Gerekli Paketleri Yükleyin

```bash
pip install -r requirements.txt
```

### Adım 3: Önceden Eğitilmiş Modeli İndirin

**Model dosyasını GitHub Releases'tan indirin:**

1. [Releases sayfasına](https://github.com/BasarBurakUnal/vehicle-classifier/releases) gidin
2. En son release'i açın (v1.0.0)
3. **Assets** bölümünden `best_vehicle_classifier.pth` dosyasını indirin
4. İndirdiğiniz dosyayı `models/` klasörüne kopyalayın

**Artık uygulamayı direkt çalıştırabilirsiniz!** 🎉

### Adım 4 (Opsiyonel): Veri Setini Hazırlayın

⚠️ **Not:** Model zaten eğitilmiş durumda. Bu adım **sadece modeli yeniden eğitmek isterseniz** gereklidir.

1. Kaggle'dan veri setini indirin: [Vehicle Classification Dataset](https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification)
2. İndirilen dosyayı projenin ana dizinine çıkarın
3. Veri seti `Vehicles/` klasöründe aşağıdaki yapıda olmalıdır:

```
Vehicles/
├── Auto Rickshaws/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (~800 görüntü)
├── Bikes/
│   └── ... (~800 görüntü)
├── Cars/
│   └── ... (~790 görüntü)
├── Motorcycles/
│   └── ... (~800 görüntü)
├── Planes/
│   └── ... (~800 görüntü)
├── Ships/
│   └── ... (~800 görüntü)
└── Trains/
    └── ... (~800 görüntü)
```

**Not:** Kaggle API'yi kullanarak da veri setini indirebilirsiniz:

```bash
kaggle datasets download -d mohamedmaher5/vehicle-classification
unzip vehicle-classification.zip
```

## 💻 Kullanım

### Hızlı Başlangıç (Önceden Eğitilmiş Model ile)

Model dosyasını [Releases'tan](https://github.com/BasarBurakUnal/vehicle-classifier/releases) indirdikten sonra direkt çalıştırın:

```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak `http://localhost:8501` açılacaktır.

### Komut Satırından Tek Görüntü Tahmini

```bash
python classifier.py path/to/image.jpg
```

---

### Alternatif: Modeli Sıfırdan Eğitin (Opsiyonel)

Eğer kendi modelinizi eğitmek isterseniz:

**1. Model Eğitimi:**

```bash
python train.py
```

Bu komut:
- Veri setini yükler ve ön işler
- Modeli eğitir (varsayılan: 20 epoch)
- En iyi modeli `models/best_vehicle_classifier.pth` olarak kaydeder
- Eğitim grafiklerini `training_history.png` olarak oluşturur

**2. Web Arayüzünü Başlatın:**

```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak `http://localhost:8501` açılacaktır.

**3. Komut Satırından Tek Görüntü Tahmini:**

```bash
python classifier.py path/to/image.jpg
```

## 📁 Proje Yapısı

```
sektor/
│
├── config.py                 # Konfigürasyon ayarları
├── data_loader.py            # Veri yükleme ve ön işleme
├── model.py                  # Model tanımı ve eğitim sınıfları
├── train.py                  # Model eğitim script'i
├── classifier.py             # Sınıflandırma modülü
├── app.py                    # Streamlit web arayüzü
├── requirements.txt          # Gerekli Python paketleri
├── README.md                 # Proje dokümantasyonu
│
├── Vehicles/                 # Veri seti klasörü (~5,600 görüntü)
│   ├── Auto Rickshaws/       # ~800 görüntü
│   ├── Bikes/                # ~800 görüntü
│   ├── Cars/                 # ~790 görüntü
│   ├── Motorcycles/          # ~800 görüntü
│   ├── Planes/               # ~800 görüntü
│   ├── Ships/                # ~800 görüntü
│   └── Trains/               # ~800 görüntü
│
└── models/                   # Eğitilmiş modeller
    ├── vehicle_classifier.pth
    └── best_vehicle_classifier.pth
```

## 🧠 Model Detayları

### Mimari

- **Temel Model:** ResNet18 (ImageNet ağırlıkları ile)
- **Transfer Learning:** İlk katmanlar dondurulmamış
- **Özel Katmanlar:** 
  - Dropout (0.5)
  - Linear (512 → 256)
  - ReLU
  - Dropout (0.3)
  - Linear (256 → 7)

### Eğitim Parametreleri

- **Optimizer:** Adam (lr=0.001)
- **Loss Fonksiyonu:** CrossEntropyLoss
- **Batch Size:** 32
- **Epoch Sayısı:** 20
- **Learning Rate Scheduler:** ReduceLROnPlateau

### Veri Augmentation

- Random rotation (±20°)
- Random horizontal flip
- Color jitter (brightness, contrast)
- Random affine transformations
- Normalization (ImageNet mean/std)

## 📊 Performans Metrikleri

Eğitim tamamlandıktan sonra model şu metriklere göre değerlendirilir:

- **Accuracy:** Genel doğruluk oranı
- **Loss:** Kayıp fonksiyonu değeri
- **Class-wise Accuracy:** Her sınıf için ayrı doğruluk
- **Confusion Matrix:** Sınıf karışıklık matrisi

### Örnek Sonuçlar

Model eğitim sonuçları:

- **Test Accuracy:** ~92-95% (ResNet18 transfer learning ile)
- **Training Time:** Yaklaşık 60 dakika (GPU kullanımıyla)
- **Model Boyutu:** ~44.7 MB
- **İnferans Süresi:** Görüntü başına ~50-100ms

> Not: Model performansı eğitim parametrelerine ve veri kalitesine bağlı olarak değişiklik gösterebilir.



## 📚 Kullanılan Teknolojiler

### Derin Öğrenme ve Veri İşleme
- **PyTorch:** Derin öğrenme framework'ü
- **torchvision:** Görüntü işleme ve önceden eğitilmiş modeller
- **PIL (Pillow):** Görüntü işleme ve manipülasyon
- **NumPy:** Sayısal hesaplamalar ve array işlemleri

### Web Arayüzü ve Görselleştirme
- **Streamlit:** Modern ve interaktif web arayüzü
- **Plotly:** İnteraktif grafikler ve çizimler
- **Matplotlib:** Veri görselleştirme ve grafik oluşturma


- **Veri Seti:** [Mohamed Maher](https://www.kaggle.com/mohamedmaher5) tarafından Kaggle'da paylaşılan [Vehicle Classification Dataset](https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification)
- **PyTorch:** Facebook AI Research (FAIR) ekibi tarafından geliştirilen derin öğrenme framework'ü
- **ResNet:** Microsoft Research tarafından geliştirilen devrimci CNN mimarisi

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!


