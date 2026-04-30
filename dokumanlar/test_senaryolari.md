# Test Senaryolari

Bu dokuman manuel kontrol surecini standartlastirmak icin hazirlanmistir.

## On Kosullar

- Uygulama calisiyor olmalidir:
  - `python3 -m uvicorn uygulama.main:app --host 127.0.0.1 --port 8000 --reload`
- Testler yerel ortamda `http://127.0.0.1:8000` adresi uzerinden kosulur.
- API testlerinde `Content-Type: application/json` kullanilir.

## Ortak Gecerli Girdi (Referans)

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

## Senaryolar

### S01 - Health endpoint kontrolu
- Kanal: API
- Girdi: `GET /health`
- Beklenen davranis:
  - HTTP 200 doner.
  - JSON icinde `durum=ok`, `uygulama`, `ortam` alanlari bulunur.

### S02 - Ana sayfa render kontrolu
- Kanal: UI
- Girdi: `GET /`
- Beklenen davranis:
  - HTTP 200 doner.
  - Form ekrani acilir.
  - "Tahmini Hesapla" ve "Tumunu Temizle" butonlari gorunur.

### S03 - Predict endpoint gecerli veri ile calisma
- Kanal: API
- Girdi: Ortak gecerli girdi ile `POST /predict`
- Beklenen davranis:
  - HTTP 200 doner.
  - JSON icinde `olasilik`, `sinif`, `risk_kategorisi`, `top_faktorler`, `kisa_aciklama` alanlari vardir.

### S04 - Predict endpoint alan tipleri kontrolu
- Kanal: API
- Girdi: Ortak gecerli girdi ile `POST /predict`
- Beklenen davranis:
  - `olasilik` 0-1 araliginda sayidir.
  - `sinif` sadece 0 veya 1 olur.
  - `risk_kategorisi` sadece `dusuk|orta|yuksek` olur.
  - `top_faktorler` liste tipindedir.

### S05 - Form submit sonucu
- Kanal: UI
- Girdi: Ana sayfadaki forma gecerli degerler girilip gonderilir.
- Beklenen davranis:
  - Sonuc sayfasi acilir.
  - Risk yuzdesi, risk seviyesi, top 3 faktor ve kisa aciklama gorunur.

### S06 - Form reset butonu
- Kanal: UI
- Girdi: Forma degerler yazilip "Tumunu Temizle" tiklanir.
- Beklenen davranis:
  - Form alanlari temizlenir.

### S07 - Dusuk risk profili A
- Kanal: API
- Girdi:
```json
{
  "pregnancies": 0,
  "glucose": 90,
  "blood_pressure": 70,
  "skin_thickness": 20,
  "insulin": 80,
  "bmi": 22.0,
  "diabetes_pedigree_function": 0.2,
  "age": 22
}
```
- Beklenen davranis:
  - HTTP 200
  - `risk_kategorisi = dusuk`

### S08 - Dusuk risk profili B
- Kanal: API
- Girdi:
```json
{
  "pregnancies": 1,
  "glucose": 100,
  "blood_pressure": 72,
  "skin_thickness": 25,
  "insulin": 90,
  "bmi": 24.0,
  "diabetes_pedigree_function": 0.3,
  "age": 28
}
```
- Beklenen davranis:
  - HTTP 200
  - `risk_kategorisi = dusuk`

### S09 - Orta risk profili A
- Kanal: API
- Girdi:
```json
{
  "pregnancies": 3,
  "glucose": 125,
  "blood_pressure": 78,
  "skin_thickness": 28,
  "insulin": 120,
  "bmi": 29.5,
  "diabetes_pedigree_function": 0.45,
  "age": 36
}
```
- Beklenen davranis:
  - HTTP 200
  - `risk_kategorisi = orta`

### S10 - Orta risk profili B
- Kanal: API
- Girdi:
```json
{
  "pregnancies": 4,
  "glucose": 135,
  "blood_pressure": 80,
  "skin_thickness": 30,
  "insulin": 140,
  "bmi": 31.0,
  "diabetes_pedigree_function": 0.5,
  "age": 40
}
```
- Beklenen davranis:
  - HTTP 200
  - `risk_kategorisi = orta`

### S11 - Yuksek risk profili A
- Kanal: API
- Girdi: Ortak gecerli girdi
- Beklenen davranis:
  - HTTP 200
  - `risk_kategorisi = yuksek`

### S12 - Yuksek risk profili B
- Kanal: API
- Girdi:
```json
{
  "pregnancies": 7,
  "glucose": 165,
  "blood_pressure": 86,
  "skin_thickness": 35,
  "insulin": 180,
  "bmi": 36.0,
  "diabetes_pedigree_function": 0.8,
  "age": 52
}
```
- Beklenen davranis:
  - HTTP 200
  - `risk_kategorisi = yuksek`

### S13 - Gecersiz yas (alt sinir disi)
- Kanal: API
- Girdi: Ortak gecerli girdide `age=17`
- Beklenen davranis:
  - HTTP 422
  - Hata mesaji `age` aralik bilgisini icerir.

### S14 - Gecersiz yas (ust sinir disi)
- Kanal: API
- Girdi: Ortak gecerli girdide `age=121`
- Beklenen davranis:
  - HTTP 422
  - Hata mesaji `age` aralik bilgisini icerir.

### S15 - Gecersiz glucose (negatif)
- Kanal: API
- Girdi: Ortak gecerli girdide `glucose=-1`
- Beklenen davranis:
  - HTTP 422
  - Hata mesaji `glucose` aralik bilgisini icerir.

### S16 - Gecersiz glucose (ust sinir disi)
- Kanal: API
- Girdi: Ortak gecerli girdide `glucose=301`
- Beklenen davranis:
  - HTTP 422
  - Hata mesaji `glucose` aralik bilgisini icerir.

### S17 - Eksik zorunlu alan
- Kanal: API
- Girdi: `age` alani olmadan `POST /predict`
- Beklenen davranis:
  - HTTP 422
  - Hata icinde eksik alan bilgisi gorunur.

### S18 - Fazladan alan gonderimi
- Kanal: API
- Girdi: Ortak gecerli girdiye `extra_field` eklenir.
- Beklenen davranis:
  - HTTP 422
  - Pydantic `extra_forbidden` turu hata doner.

### S19 - Tip uyumsuzlugu
- Kanal: API
- Girdi: Ortak gecerli girdide `bmi="abc"`
- Beklenen davranis:
  - HTTP 422
  - `bmi` icin sayisal tip hatasi doner.

### S20 - Bozuk JSON govdesi
- Kanal: API
- Girdi: JSON formatini bozan govde ile `POST /predict`
- Beklenen davranis:
  - HTTP 422
  - JSON parse hatasi doner.

### S21 - Sinir durum: yas alt limiti
- Kanal: API
- Girdi: Ortak gecerli girdide `age=18`
- Beklenen davranis:
  - HTTP 200
  - Girdi kabul edilir.

### S22 - Sinir durum: yas ust limiti
- Kanal: API
- Girdi: Ortak gecerli girdide `age=120`
- Beklenen davranis:
  - HTTP 200
  - Girdi kabul edilir.

### S23 - Sinir durum: pregnancies=0
- Kanal: API
- Girdi: Ortak gecerli girdide `pregnancies=0`
- Beklenen davranis:
  - HTTP 200
  - Girdi kabul edilir.

### S24 - Sinir durum: 0 degerli klinik alanlar
- Kanal: API
- Girdi: Ortak gecerli girdide `insulin=0`, `skin_thickness=0`, `bmi=0`
- Beklenen davranis:
  - HTTP 200
  - Girdi kabul edilir.
  - Tahmin donusu olusur (imputasyon/pipeline akisi kirilmaz).

### S25 - Artifact eksikligi hatasi
- Kanal: API
- Girdi: `MODEL_ARTIFACT_KLASORU` gecersiz klasore cekilip `POST /predict`
- Beklenen davranis:
  - HTTP 500
  - Hata mesaji artifact/dosya bulunamadi bilgisini icerir.

## Kapsam Ozeti

- Toplam senaryo: 25
- Kapsanan alanlar:
  - Temel endpointler
  - Dusuk/orta/yuksek risk akislari
  - Gecersiz input reddi
  - Sinir durumlari
  - Artifact bagimliligi
