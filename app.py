"""
Streamlit Web Arayüzü

Yapay Zeka Destekli Araç Sınıflandırıcı için kullanıcı dostu web arayüzü.

Kullanım:
    streamlit run app.py
"""

import streamlit as st
import torch
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import os
import config
from classifier import ImageClassifier


# Sayfa yapılandırması
st.set_page_config(
    page_title="Araç Sınıflandırıcı",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 12px;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
        border: none;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        padding: 20px;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classifier():
    """
    Sınıflandırıcıyı yükler ve önbelleğe alır.
    
    Returns:
        ImageClassifier: Yüklenmiş sınıflandırıcı
    """
    try:
        classifier = ImageClassifier()
        return classifier
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {str(e)}")
        st.info("Lütfen önce modeli eğitin: `python train.py`")
        return None


def create_probability_chart(probabilities: dict):
    """
    Olasılık grafiği oluşturur.
    
    Args:
        probabilities (dict): Sınıf olasılıkları
    """
    # Verileri hazırlar
    classes = list(probabilities.keys())
    probs = [probabilities[cls] * 100 for cls in classes]
    
    # Renk skalası
    colors = ['#FF6B6B' if p < 20 else '#4ECDC4' if p < 50 else '#45B7D1' if p < 80 else '#95E1D3' 
              for p in probs]
    
    # Bar chart oluşturur
    fig = go.Figure(data=[
        go.Bar(
            x=probs,
            y=classes,
            orientation='h',
            marker=dict(
                color=probs,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Olasılık (%)")
            ),
            text=[f'{p:.1f}%' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Sınıf Olasılık Dağılımı',
        xaxis_title='Olasılık (%)',
        yaxis_title='Araç Sınıfı',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def main():
    """
    Ana uygulama fonksiyonu.
    """
    # Başlık
    st.markdown("<h1>🚗 Yapay Zeka Destekli Araç Sınıflandırıcı</h1>", unsafe_allow_html=True)
    
    # Sidebar - Bilgi ve Ayarlar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3097/3097170.png", width=100)
        st.title("ℹ️ Proje Bilgileri")
        
        st.markdown("""
        ### 🎯 Proje Hakkında
        Bu uygulama, derin öğrenme teknikleri kullanarak 
        araç görüntülerini 7 farklı kategoriye sınıflandırır.
        
        ### 📊 Sınıflar
        """)
        
        for i, class_name in enumerate(config.CLASS_NAMES, 1):
            st.markdown(f"**{i}.** {class_name}")
        
        st.markdown("---")
        
        st.markdown("""
        ### 🔧 Teknolojiler
        - **Model:** ResNet18 (Transfer Learning)
        - **Framework:** PyTorch
        - **Arayüz:** Streamlit
        - **Veri İşleme:** PIL, NumPy
        """)
        
        st.markdown("---")
        
        # Model bilgileri
        if os.path.exists(config.BEST_MODEL_PATH):
            st.success("✅ Model yüklendi")
            
            # GPU/CPU bilgisi
            device = "GPU" if torch.cuda.is_available() else "CPU"
            st.info(f"🖥️ Cihaz: {device}")
        else:
            st.error("❌ Model bulunamadı!")
            st.warning("Lütfen önce modeli eğitin: `python train.py`")
    
    # Ana içerik
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Görüntü Yükleme")
        
        # Dosya yükleme (çoklu)
        uploaded_files = st.file_uploader(
            "Bir veya birden fazla araç görüntüsü yükleyin",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="JPG, JPEG veya PNG formatında görüntüler yükleyin (çoklu seçim yapabilirsiniz)"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} görüntü yüklendi")
            
            # Görüntüleri grid şeklinde göster
            if len(uploaded_files) <= 4:
                cols = st.columns(min(len(uploaded_files), 4))
                for idx, uploaded_file in enumerate(uploaded_files):
                    with cols[idx % 4]:
                        image = Image.open(uploaded_file)
                        st.image(image, caption=f'Görüntü {idx+1}', use_container_width=True)
            else:
                # Çok fazla görüntü varsa sadece sayıyı göster
                with st.expander(f"📷 {len(uploaded_files)} Görüntü Önizleme"):
                    cols = st.columns(4)
                    for idx, uploaded_file in enumerate(uploaded_files[:8]):  # İlk 8'i göster
                        with cols[idx % 4]:
                            image = Image.open(uploaded_file)
                            st.image(image, caption=f'{idx+1}', use_container_width=True)
                    if len(uploaded_files) > 8:
                        st.info(f"...ve {len(uploaded_files) - 8} görüntü daha")
    
    with col2:
        st.markdown("### 🎯 Tahmin Sonuçları")
        
        if uploaded_files:
            # Tahmin butonu
            if st.button("🔍 Tüm Görüntüleri Analiz Et"):
                with st.spinner(f"{len(uploaded_files)} görüntü analiz ediliyor..."):
                    # Classifier'ı yükle
                    classifier = load_classifier()
                    
                    if classifier is not None:
                        try:
                            # Tüm görüntüler için tahmin yap
                            all_results = []
                            class_counts = {cls: 0 for cls in config.CLASS_NAMES}
                            class_probs_sum = {cls: 0.0 for cls in config.CLASS_NAMES}
                            
                            progress_bar = st.progress(0)
                            for idx, uploaded_file in enumerate(uploaded_files):
                                image = Image.open(uploaded_file)
                                predicted_class, confidence, all_probs = classifier.classify_image(image)
                                
                                all_results.append({
                                    'file_name': uploaded_file.name,
                                    'predicted_class': predicted_class,
                                    'confidence': confidence,
                                    'all_probs': all_probs
                                })
                                
                                # Sınıf sayılarını güncelle
                                class_counts[predicted_class] += 1
                                
                                # Olasılıkları topla
                                for cls, prob in all_probs.items():
                                    class_probs_sum[cls] += prob
                                
                                progress_bar.progress((idx + 1) / len(uploaded_files))
                            
                            progress_bar.empty()
                            st.success(f"✅ {len(uploaded_files)} görüntü analiz edildi!")
                            
                            # Ortalama olasılıkları hesapla
                            avg_probs = {cls: prob / len(uploaded_files) for cls, prob in class_probs_sum.items()}
                            
                            # Ortalama güven skorunu hesapla
                            avg_confidence = sum(r['confidence'] for r in all_results) / len(all_results)
                            
                            # Düşük güvenli görüntüleri say
                            low_confidence_count = sum(1 for r in all_results if r['confidence'] < 0.60)
                            medium_confidence_count = sum(1 for r in all_results if 0.60 <= r['confidence'] < 0.75)
                            
                            # Uyarı Sistemi - Genel Değerlendirme
                            if avg_confidence < 0.60 or low_confidence_count > len(uploaded_files) * 0.5:
                                st.error("🚫 **UYARI:** Yüklenen görüntülerin çoğu araç görüntüsü olmayabilir!")
                                st.warning(f"⚠️ Model, {low_confidence_count} görüntüden emin değil. Lütfen araç görüntüleri yüklediğinizden emin olun.")
                                st.info("💡 **Öneri:** Bu sistem sadece Auto Rickshaws, Bikes, Cars, Motorcycles, Planes, Ships ve Trains sınıflarını tanır.")
                            elif medium_confidence_count > len(uploaded_files) * 0.3:
                                st.warning(f"⚠️ **DİKKAT:** {medium_confidence_count} görüntüde model orta seviye güven gösteriyor. Bazı görüntüler belirtilen sınıflarda olmayabilir.")
                            
                            # En çok tahmin edilen sınıf
                            most_common_class = max(class_counts.items(), key=lambda x: x[1])
                            
                            # Toplu sonuç kartı
                            st.markdown(f"""
                            <div class="metric-card">
                                <h2 style="margin: 0; font-size: 24px;">En Çok Tespit Edilen Sınıf</h2>
                                <h1 style="margin: 10px 0; font-size: 36px;">🚘 {most_common_class[0]}</h1>
                                <h3 style="margin: 0; font-size: 20px;">{most_common_class[1]} / {len(uploaded_files)} görüntü ({most_common_class[1]/len(uploaded_files)*100:.1f}%)</h3>
                                <p style="margin-top: 10px; font-size: 16px;">Ortalama Güven: {avg_confidence*100:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Sınıf dağılımı
                            st.markdown("#### 📊 Toplu Sınıf Dağılımı")
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.markdown("**Tespit Edilen Sınıflar:**")
                                for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                                    if count > 0:
                                        st.metric(cls, f"{count} adet", f"{count/len(uploaded_files)*100:.1f}%")
                            
                            with col_b:
                                st.markdown("**Ortalama Güven Skorları:**")
                                for cls, prob in sorted(avg_probs.items(), key=lambda x: x[1], reverse=True)[:5]:
                                    st.metric(cls, f"{prob*100:.2f}%")
                            
                            # Global değişkene kaydet (grafik için)
                            st.session_state['all_results'] = all_results
                            st.session_state['avg_probs'] = avg_probs
                            st.session_state['class_counts'] = class_counts
                        
                        except Exception as e:
                            st.error(f"Hata: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
        else:
            st.info("👈 Lütfen sol taraftan bir veya daha fazla görüntü yükleyin")
    
    # Detaylı sonuçlar
    if 'avg_probs' in st.session_state:
        st.markdown("---")
        st.markdown("### 📈 Detaylı Analiz")
        
        # Ortalama olasılık grafiği
        fig = create_probability_chart(st.session_state['avg_probs'])
        fig.update_layout(title='Ortalama Güven Skor Dağılımı (Tüm Görüntüler)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Sınıf sayımları grafiği
        if len(uploaded_files) > 1:
            import plotly.graph_objects as go
            
            class_counts = st.session_state['class_counts']
            non_zero_classes = {k: v for k, v in class_counts.items() if v > 0}
            
            if non_zero_classes:
                fig2 = go.Figure(data=[
                    go.Pie(
                        labels=list(non_zero_classes.keys()),
                        values=list(non_zero_classes.values()),
                        hole=0.3,
                        textinfo='label+percent+value',
                        marker=dict(colors=px.colors.qualitative.Vivid)
                    )
                ])
                fig2.update_layout(
                    title=f'Tespit Edilen Sınıf Dağılımı ({len(uploaded_files)} Görüntü)',
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        # Bireysel sonuçlar
        with st.expander(f"📋 Bireysel Görüntü Sonuçları ({len(st.session_state['all_results'])} adet)"):
            import pandas as pd
            
            results_data = []
            for result in st.session_state['all_results']:
                confidence = result['confidence']
                
                # Uyarı durumu belirle
                if confidence < 0.60:
                    status = "🚫 Araç Değil Olabilir"
                elif confidence < 0.75:
                    status = "⚠️ Düşük Güven"
                else:
                    status = "✅ Güvenilir"
                
                results_data.append({
                    'Dosya Adı': result['file_name'],
                    'Tahmin': result['predicted_class'],
                    'Güven (%)': f"{confidence*100:.2f}",
                    'Durum': status
                })
            
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Uyarı açıklaması
            st.markdown("""
            **Durum Açıklamaları:**
            - 🚫 **Araç Değil Olabilir:** Güven < %60 - Bu görüntü muhtemelen belirtilen araç sınıflarından değil
            - ⚠️ **Düşük Güven:** Güven %60-75 - Model bu görüntüden tam emin değil
            - ✅ **Güvenilir:** Güven > %75 - Yüksek olasılıkla doğru tahmin
            """)
        
        # Tüm sınıf olasılıkları (ortalama)
        with st.expander("📊 Ortalama Sınıf Olasılıkları"):
            import pandas as pd
            
            df = pd.DataFrame({
                'Sınıf': list(st.session_state['avg_probs'].keys()),
                'Ortalama Olasılık (%)': [f"{v*100:.2f}" for v in st.session_state['avg_probs'].values()]
            })
            df = df.sort_values('Ortalama Olasılık (%)', ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Alt bilgi
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
        <p>🎓 Yapay Zeka Destekli Görüntü Sınıflandırıcı Projesi</p>
        <p>PyTorch • ResNet18 • Transfer Learning • Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Model kontrolü
    if not os.path.exists(config.BEST_MODEL_PATH):
        st.warning("⚠️ Model dosyası bulunamadı. Lütfen önce modeli eğitin:")
        st.code("python train.py", language="bash")
    
    main()

