# Diyabet Risk Tahmini - PIMA Sentetik Benchmark

Bu repo, PIMA Indians Diabetes veri seti üzerinde geliştirilen diyabet risk tahmini çalışmasının ML deneme sürümüdür. Ana odak, ham PIMA üzerinde klinik genellenebilirlik iddiası kurmak değil; `source_id` aile yapısı korunmuş PIMA + sentetik benchmark üzerinde savunulabilir bir modelleme ve raporlama akışı oluşturmaktır.

## Deney Ölçeği

Bu kopya proje üzerinde yalnız tek bir final model eğitilmedi. Veri artırmadan başlayan, literatür profilleriyle genişleyen, daha sonra `source_id` kontrollü sentetik benchmark ve küçük/orta ölçek sweep aşamalarına ayrılan kapsamlı bir deneme süreci yürütüldü.

Repo içinde kayıtlı sonuçlara göre yaklaşık deney ölçeği:

| Kategori | Sayı |
| --- | ---: |
| Toplam kayıtlı model/konfigürasyon denemesi | ~4.957 |
| Üretilen deneysel CSV dosyası | 3.608 |
| Gerçek veri adayı CSV | 1.806 |
| Metadata CSV | 1.802 |
| Conservative 500 veri adayı | 1.704 |
| Small-scale sweep veri adayı | 87 |
| JSON değerlendirme/rapor dosyası | 40+ |
| Kullanılabilir Word raporu | 17 |

Deneme gruplarının yaklaşık dağılımı:

| Deney grubu | Yaklaşık deneme |
| --- | ---: |
| Veri artırmadan ilk model karşılaştırmaları | 10 |
| CV strateji denemeleri | 10 |
| Agresif accuracy araması | 75 |
| Literatür profilleri | 16 |
| İlk veri müdahalesi ve sentetik benchmark denemeleri | 100+ |
| Strict `source_id` 2500/2700/5000 benchmark | 88 |
| Min90 geniş benchmark | 678 |
| Small-scale sweep | 1.635 |
| Conservative 500 sweep | 2.420 |

Bu sayıların amacı skoru şişirmek değildir. Amaç, hangi veri boyutu, üretim yaklaşımı, model ailesi, feature seti ve doğrulama protokolünün daha savunulabilir sonuç verdiğini sistematik olarak görmekti.

## Final Seçim

Final proje adayı 1000/1000 source_id kontrollü sentetik benchmarktır.

| Alan | Değer |
| --- | --- |
| Final benchmark | 1000 negatif / 1000 pozitif |
| Toplam satır | 2000 |
| Orijinal geliştirme satırı | 614 |
| Eklenen sentetik satır | 1386 |
| Model | ExtraTrees |
| Model içi özellik seti | High-signal features |
| Karar eşiği | 0.45 |
| Synthetic holdout accuracy | 0.9393 |
| Synthetic holdout min ana metrik | 0.9343 |
| Synthetic holdout ROC-AUC | 0.9860 |
| Group CV min ana metrik | 0.9030 ± 0.0097 |
| Leakage durumu | Clean |

Min ana metrik; accuracy, precision, recall, specificity, F1 ve balanced accuracy metriklerinin minimumudur. Bu değer tek bir metriği şişirmek yerine sınıflar arası dengeyi kontrol etmek için kullanıldı.

## Kritik Ayrım

Sentetik benchmark sonucu ile original external holdout sonucu aynı anlama gelmez. Sentetik benchmark, kontrollü veri artırımı sonrası modelin ayrıştırma gücünü gösterir. Original external holdout ise gerçek PIMA dağılımına dış kontrol olarak bakar. Bu çalışma klinik tanı aracı değildir ve gerçek dünya genellenebilirliği için bağımsız dış veri setleri gerekir.

## Source ID ve Leakage Kontrolü

Her orijinal PIMA satırı bir kaynak aile olarak kabul edildi.

```text
Orijinal satır: source_id = original_125
Bu satırdan türeyen sentetik satırlar: source_id = original_125
```

Bu yapı sayesinde aynı kaynak aileden gelen örneklerin hem eğitim hem test tarafına düşmesi engellendi. Raporlanan kontroller:

- Train/test source_id kesişimi: 0
- CV fold source_id kesişimi: 0
- Exact duplicate: 0
- Independent synthetic source_id: 0
- External holdout overlap: 0

## Klasör Yapısı

- `uygulama/`: FastAPI uygulaması ve tahmin servisleri
- `makine_ogrenmesi/kaynak/`: ML kaynak kodları
- `makine_ogrenmesi/artifactler/`: final 1000/1000 artifact dosyaları
- `makine_ogrenmesi/veri/ham/diabetes.csv`: ham PIMA veri seti
- `makine_ogrenmesi/veri/deneysel/`: bütün sentetik veri denemeleri, aday CSV dosyaları ve metadata çıktıları
- `makine_ogrenmesi/raporlar/`: JSON, grafik ve Word raporları
- `raporlar/`: teslim raporları
- `betikler/`: eğitim, sweep ve rapor üretim betikleri
- `testler/`: pytest testleri

## Dahil Edilen Deney Aileleri

Repo yalnız final artifact'ten oluşmaz. Hoca deney akışını kontrol edebilsin diye veri artırmadan denemeler, literatür profilleri, leakage kontrollü sentetik benchmarklar, küçük/orta ölçek sweep sonuçları ve risk audit raporları da proje içinde bırakıldı.

Öne çıkan deney aileleri:

- Veri artırmadan benchmark: `betikler/ml_yeniden_kur.py`, `betikler/literatur_deneyleri.py`
- Veri müdahalesi ve yüksek metrik araması: `betikler/veri_mudahale_deneyleri.py`
- Source ID kontrollü min90 benchmark: `betikler/min90_sentetik_benchmark.py`
- Küçük/orta ölçek sweep: `betikler/min90_small_scale_sweep.py`
- 500/650/800/1000 korelasyon ve dağılım raporu: `betikler/pima_korelasyon_deney_dagilim_raporu.py`
- Sentetik risk audit: `betikler/sentetik_risk_audit_raporu_uret.py`

Deneysel veri çıktıları:

- `makine_ogrenmesi/veri/deneysel/`
- `makine_ogrenmesi/veri/deneysel/min90_small_scale/`
- `makine_ogrenmesi/veri/deneysel/conservative_500/`

Deney raporları:

- `makine_ogrenmesi/raporlar/`
- `makine_ogrenmesi/raporlar/degerlendirme/`
- `makine_ogrenmesi/raporlar/grafikler_min90_small_scale/`
- `makine_ogrenmesi/raporlar/grafikler_pima_korelasyon_deney_dagilim/`

Repo dışı bırakılanlar yalnız çalışma çöpüdür: `.venv`, `tmp`, `node_modules`, cache klasörleri, `.DS_Store` ve eski büyük artifact yedekleri.

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

macOS üzerinde XGBoost/OpenMP hatası alınırsa:

```bash
brew install libomp
```

## Uygulamayı Çalıştırma

```bash
python3 -m uvicorn uygulama.main:app --host 127.0.0.1 --port 8000 --reload
```

Adresler:

- Ana sayfa: `http://127.0.0.1:8000/`
- Health: `http://127.0.0.1:8000/health`
- Swagger: `http://127.0.0.1:8000/docs`

## API Örneği

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 2,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree_function": 0.627,
    "age": 50
  }'
```

## Önemli Betikler

```bash
# Final 1000/1000 artifact üretimi
python betikler/final_1000_artifact_uret.py

# Küçük/orta ölçek benchmark sweep
python betikler/min90_small_scale_sweep.py --n-jobs 2

# PIMA korelasyon ve aday dağılım raporu
python betikler/pima_korelasyon_deney_dagilim_raporu.py
```

## Test

```bash
pytest -q
```

## Raporlar

Öne çıkan raporlar:

- `makine_ogrenmesi/raporlar/min90_small_scale_literatur_stili_rapor.docx`
- `makine_ogrenmesi/raporlar/pima_korelasyon_deney_dagilim_raporu.docx`
- `makine_ogrenmesi/raporlar/sentetik_benchmark_risk_audit_raporu.docx`
- `raporlar/PIMA_Tum_Raporlar/`

## Son Not

Bu repo akademik/Ar-Ge amaçlıdır. Yüksek metrikler PIMA + source_id kontrollü sentetik benchmark bağlamında yorumlanmalıdır. Modelin klinik ortamda kullanılabilmesi için bağımsız dış veri setleriyle ayrıca doğrulanması gerekir.
