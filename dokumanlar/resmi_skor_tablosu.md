# Resmi Skor Tablosu

## 1. Deploy Özeti

- Model: `xgboost`
- Kalibrasyon: `none`
- İkili sınıflama yöntemi: `accuracy_oncelikli_f1_kisitli`
- İkili sınıflama eşiği: `0.3800`

### 1.1 Hedef Uyum Tablosu

| Metrik | Hedef | Gerçekleşen | Durum |
| --- | --- | --- | --- |
| Accuracy | >= 0.7800 | 0.7727 | Sağlanmadı |
| ROC AUC | >= 0.8000 | 0.8263 | Sağlandı |
| F1 | >= 0.7000 | 0.7244 | Sağlandı |
| Brier iyileşme | >= %10.00 | %0.00 | Sağlanmadı |

### 1.2 Deploy Metrikleri

| Metrik | Değer |
| --- | --- |
| Accuracy | 0.7727 |
| Precision | 0.6301 |
| Recall | 0.8519 |
| F1 | 0.7244 |
| ROC AUC | 0.8263 |
| Brier | 0.1662 |

## 2. Kalibrasyon Etkisi

| Kalibrasyon Durumu | Brier |
| --- | --- |
| Kalibrasyon öncesi | 0.1662 |
| Kalibrasyon sonrası | 0.1662 |
| İyileşme oranı | %0.00 |

## 3. Model Karşılaştırma (Test Seti)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost | 0.7403 | 0.6029 | 0.7593 | 0.6721 | 0.8339 | 0.1656 |
| random_forest | 0.7208 | 0.5753 | 0.7778 | 0.6614 | 0.8143 | 0.1725 |
| logistic_regression | 0.7143 | 0.5833 | 0.6481 | 0.6140 | 0.8081 | 0.1828 |

## 4. Model Değerlendirme ve Deploy Farkı

- Ham model kıyas eşiği: `0.5000`
- Deploy sınıflama eşiği: `0.3800`
- Deploy kalibrasyon yöntemi: `none`

| Metrik | Ham Değerlendirme | Deploy | Fark (Deploy - Ham) |
| --- | --- | --- | --- |
| ACCURACY | 0.7403 | 0.7727 | +0.0325 |
| PRECISION | 0.6029 | 0.6301 | +0.0272 |
| RECALL | 0.7593 | 0.8519 | +0.0926 |
| F1 | 0.6721 | 0.7244 | +0.0523 |
| ROC_AUC | 0.8339 | 0.8263 | -0.0076 |
| BRIER | 0.1656 | 0.1662 | +0.0006 |

Model degerlendirme raporu, ham model adaylarini test setinde 0.50 siniflama esigi ile karsilastirir. Deploy metrikleri ise secilen model + secilen kalibrasyon + optimize edilen ikili siniflama esigi ile hesaplandigi icin fark olusmasi normaldir.

## 5. PIMA Literatür Benchmark ve Metodoloji Notu

| Metrik | Güvenilir Alt Bant | Güvenilir Üst Bant |
| --- | --- | --- |
| ACCURACY | 0.7500 | 0.8400 |
| F1 | 0.6200 | 0.8500 |
| ROC_AUC | 0.8000 | 0.8700 |
| PRECISION | 0.5700 | 0.9300 |
| RECALL | 0.6500 | 0.9000 |
| BRIER | 0.1500 | 0.2000 |

- Kucuk veri setlerinde tek bolme ile raporlanan cok yuksek sonuclarin genellenebilirligi dusuktur.
- SMOTE yalnizca egitim katmaninda uygulanmalidir; test verisine uygulanmasi veri sizintisi olusturur.
- PIMA icin %98-%100 accuracy iddialari cogu durumda overfit veya leakage riski tasir.
- Bu projede stratified split, capraz dogrulama ve kalibrasyon ayrimi korunmustur.

## 6. Dokümandaki Hedefleri Yakalamak İçin Teknik Yol Haritası

1. Veri hacmini artırın: Pima veri seti `768` kayıt olduğu için `%90+ accuracy` hedefi aşırı iddialı kalıyor. Benzer dağılımda en az `3.000+` kayıtlık ek veri, hedef metriklerde anlamlı stabilite sağlar.
2. Özellikleri zenginleştirin: HbA1c, bel çevresi, aile öyküsü detay seviyesi, ilaç kullanımı ve geçmiş gebelik diyabet öyküsü gibi klinik olarak daha ayırt edici değişkenler performansı doğrudan etkiler.
3. Veri kalitesini standartlaştırın: Eksik/0 değerlerin ölçüm kaynaklı mı gerçek sıfır mı olduğu saha bazlı temizlenmeli; etiketleme hataları için en az bir klinik uzmanla çift kontrol yapılmalıdır.
4. Hedefe göre eşik yönetimi uygulayın: F1 odağı için mevcut eşik korunurken, tarama senaryosunda recall odaklı alternatif eşik ayrıca dokümante edilmelidir.
5. Dış doğrulama yapın: Tek veri seti yerine farklı merkezden ayrık bir dış test seti olmadan yüksek metrik iddiası paydaş açısından zayıf kalır.
6. Brier iyileşmesini büyütmek için: Kalibrasyon verisi büyütülüp (ayrı validation katmanı) isotonic/sigmoid seçimi her yeni veri partisinde yeniden yapılmalıdır.
