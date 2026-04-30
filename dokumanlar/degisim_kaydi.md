# Değişim Kaydı

Bu dosya, proje kapsamını etkileyen önemli kararları kronolojik olarak izlemek için tutulur.

## 2026-04-05

### Risk Kategorisi Standardizasyonu

- Risk kategorileri tekrar 3 seviyeye indirildi.
- Yeni sabit aralıklar:
  - `%0 - %33`: `dusuk`
  - `%33 - %66`: `orta`
  - `%66 - %100`: `yuksek`
- Etkilenen katmanlar birlikte güncellendi:
  - servis mantığı
  - şema doğrulamaları
  - UI (rozet, pie, doğrusal bar etiket/segmentleri)
  - testler
  - README ve `esik_yapilandirmasi.json`

### Değerlendirme Raporunda Tutarlılık İyileştirmesi

- Değerlendirme özetinde “testte öne çıkan model” ve “deploy için seçilen model” ayrımı netleştirildi.
- “En iyi model seçimi” için kullanılan kriter çıktıda açık biçimde belirtilir hale getirildi.
- `%90+ accuracy` hedefinin sağlanamadığı durumda, proje planındaki B planına uygun gerekçe metni rapora eklendi.

### Dokümantasyon Tamamlama

- Daha önce boş olan aşağıdaki dosyalar içeriklendirilerek izlenebilir hale getirildi:
  - `dokumanlar/proje_kapsami.md`
  - `dokumanlar/risk_kaydi.md`
  - `dokumanlar/degisim_kaydi.md`
  - `dokumanlar/api_sozlesmesi.md`

### Sunum Hazırlık Paketi

- Sunum sürecini uçtan uca desteklemek için aşağıdaki yeni dokümanlar eklendi:
  - `dokumanlar/sunum_iskeleti.md`
  - `dokumanlar/canli_demo_akisi.md`
  - `dokumanlar/soru_cevap_kartlari.md`
  - `dokumanlar/prova_kontrol_listesi.md`
- Sunum akışı, canlı demo senaryoları, soru-cevap kartları ve prova kabul kriterleri tek pakette standartlaştırıldı.

## Not

Bu değişim kaydı, kod deposundaki commit geçmişinin yerine geçmez; kapsam ve karar izi için tamamlayıcı belgedir.
