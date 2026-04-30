# API Sözleşmesi

Bu doküman, `uygulama.main:app` üzerinde sunulan temel endpointlerin sözleşmesini özetler.

## 1. Genel Bilgiler

- Base URL (yerel): `http://127.0.0.1:8000`
- İçerik tipi: `application/json`
- Kimlik doğrulama: Yok (prototip aşaması)

## 2. Endpointler

### 2.1 `GET /health`

Uygulamanın çalışır durumda olduğunu doğrular.

Örnek yanıt:

```json
{
  "durum": "ok",
  "uygulama": "diyabet-risk-tahmini",
  "ortam": "gelistirme"
}
```

### 2.2 `POST /predict`

Tek bir kullanıcı girdisi için diyabet risk tahmini üretir.

İstek gövdesi:

```json
{
  "pregnancies": 2,
  "glucose": 148,
  "blood_pressure": 72,
  "skin_thickness": 35,
  "insulin": 0,
  "bmi": 33.6,
  "diabetes_pedigree_function": 0.627,
  "age": 50
}
```

Başarılı yanıt (200):

```json
{
  "olasilik": 0.58,
  "sinif": 1,
  "risk_kategorisi": "orta",
  "top_faktorler": [
    {
      "ozellik": "glucose",
      "ozellik_degeri": 148,
      "shap_katkisi": 0.21,
      "yon": "arttirici"
    }
  ],
  "kisa_aciklama": "Tahmini risk düzeyi orta seviyede (%58.0)."
}
```

## 3. Alan Kuralları

- `olasilik`: `0 - 1` aralığında ondalık sayı
- `sinif`: `0` veya `1`
- `risk_kategorisi`: `dusuk | orta | yuksek`
- `top_faktorler`: en fazla 3 öğe
- `yon`: `arttirici | azaltici`

## 4. Risk Kategorisi Eşikleri

`risk_kategorisi` alanı aşağıdaki sabit aralıklara göre üretilir:

- `%0 - %33`: `dusuk`
- `%33 - %66`: `orta`
- `%66 - %100`: `yuksek`

## 5. Hata Davranışı

- `422 Unprocessable Entity`: Girdi doğrulama hataları
- `400 Bad Request`: İş kuralı veya tip dönüşümü kaynaklı hatalar
- `500 Internal Server Error`: Artifact eksikliği veya beklenmeyen sistem hataları

## 6. Uyum Notu

Bu sözleşme, kod tarafındaki Pydantic şemaları ve servis katmanı ile uyumlu olacak şekilde güncellenir. Kapsam değişikliklerinde önce bu doküman, ardından test senaryoları güncellenir.
