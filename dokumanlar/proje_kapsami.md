# Proje Kapsamı

## 1. Proje Özeti

Bu proje, kadın bireylerde Tip-2 diyabet riskini makine öğrenmesi ile tahmin eden ve sonucu anlaşılır bir web arayüzü üzerinden sunan karar destek prototipidir. Sistem tanı koyma amacı taşımaz; risk farkındalığını artırmayı ve kullanıcıyı gerektiğinde sağlık profesyoneline yönlendirmeyi hedefler.

## 2. Amaç ve Hedef

- Kadın bireylerden oluşan veri seti üzerinde diyabet risk olasılığını hesaplamak.
- Hesaplanan olasılığı klinik iletişimi kolaylaştıran risk kategorilerine dönüştürmek.
- Tahmini etkileyen en önemli değişkenleri SHAP tabanlı açıklamalarla göstermek.
- FastAPI tabanlı, test edilebilir ve sürdürülebilir bir prototip sunmak.

## 3. Kapsam Dahilindeki Çıktılar

- `POST /predict` ve `GET /health` endpointleri
- HTML/Jinja tabanlı giriş formu ve sonuç ekranı
- Risk kategorisi üretimi (3 seviye)
- SHAP tabanlı ilk 3 etkenin kullanıcıya sunulması
- Eğitim/değerlendirme rapor dosyaları
- Otomatik testler (`pytest`)

## 4. Kapsam Dışı Konular

- Klinik tanı verme veya tedavi önerisi üretme
- Hastane bilgi yönetim sistemleriyle canlı entegrasyon
- Tıbbi cihaz/ürün regülasyon süreçlerinin tamamlanması
- Gerçek hasta verisi ile prospektif klinik validasyon

## 5. Teknik Mimari

- Backend: FastAPI
- Veri doğrulama: Pydantic
- Modelleme: scikit-learn, imbalanced-learn, XGBoost
- Açıklanabilirlik: SHAP
- Çıktı saklama: `makine_ogrenmesi/artifactler/`

## 6. Risk Kategorisi Kuralı

Sistem olasılık değerini aşağıdaki sabit aralıklara göre sınıflandırır:

- `%0 - %33`: `dusuk`
- `%33 - %66`: `orta`
- `%66 - %100`: `yuksek`

Bu kural kodda ve artifact dosyalarında aynı şekilde tanımlanmıştır.

## 7. Performans Hedefi ve Güncel Durum

TÜBİTAK başvuru metninde iddialı bir performans hedefi (`%90+ accuracy`) tanımlanmıştır. Güncel deney sonuçlarında bu hedefin tamamı karşılanmamıştır. Buna rağmen, proje planındaki B planına uygun şekilde model seçimi tek bir metriğe indirgenmemiş, AUC/F1 dengesi ve kalibrasyon yaklaşımı korunmuştur.

Bu nedenle proje, “hedefin tamamını mutlak sağlamak” yerine “en dengeli ve açıklanabilir karar destek prototipi üretmek” yaklaşımıyla ilerletilmiştir.

## 8. Model Seçim Tutarlılığı Notu

Değerlendirme raporunda test performansı en yüksek model ile deploy edilen model farklı olabilir. Güven ve izlenebilirlik için rapor çıktısında:

- test performansında öne çıkan model
- deploy için seçilen model
- seçim kriteri

alanları ayrı ayrı gösterilir. Böylece dış paydaşlara tek cümlelik ama eksik bir “en iyi model” mesajı yerine şeffaf bir karar çerçevesi sunulur.

## 9. Kanıt ve İzlenebilirlik Dosyaları

- `makine_ogrenmesi/raporlar/degerlendirme/egitim_ozeti.json`
- `makine_ogrenmesi/raporlar/degerlendirme/model_degerlendirme_ozeti.json`
- `makine_ogrenmesi/artifactler/model_metadata.json`
- `makine_ogrenmesi/artifactler/metrik_ozeti.json`
- `dokumanlar/test_senaryolari.md`
