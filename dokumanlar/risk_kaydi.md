# Risk Kaydı

Bu doküman, proje yürütümü sırasında ortaya çıkan teknik ve operasyonel riskleri takip etmek için hazırlanmıştır.

| Risk No | Risk Tanımı | Olasılık | Etki | Güncel Durum | Aksiyon / B Planı |
|---|---|---|---|---|---|
| R-01 | Performans hedeflerinin (`%90+ accuracy`) tamamının karşılanamaması | Orta | Yüksek | Açık | Değerlendirme tek metrik yerine AUC/F1/kalibrasyon dengesi ile yapılır. Sunumda hedef sapması teknik gerekçesiyle açıkça belirtilir. |
| R-02 | “En iyi model” raporu ile deploy metadata arasında tutarsızlık algısı | Orta | Yüksek | Azaltıldı | Değerlendirme çıktısında testte öne çıkan model ile deploy modeli ayrı alanlarda raporlanır; seçim kriteri ayrıca yazılır. |
| R-03 | Dokümantasyon boş veya güncel kodla uyumsuz kalabilir | Yüksek | Orta | Azaltıldı | `proje_kapsami`, `risk_kaydi`, `degisim_kaydi`, `api_sozlesmesi` düzenli güncellenir; her kapsam değişikliği sonrası doküman kontrolü zorunlu tutulur. |
| R-04 | Risk kategorisi kuralının kod, UI ve artifact arasında farklılaşması | Orta | Yüksek | Azaltıldı | Kural tek kaynak olarak servis ve artifact tarafında sabitlenir; testler ve README aynı aralıklarla tutulur. |
| R-05 | Demo sırasında ortam kaynaklı hata (port çakışması, bağımlılık eksikliği) | Orta | Orta | Açık | Demo öncesi kontrol listesi: `pip install -r requirements.txt`, `pytest -q`, `lsof -i :8000`, `GET /health`. |
| R-06 | SHAP çıktılarının son kullanıcı tarafından zor anlaşılması | Düşük | Orta | Açık | Sonuç ekranında kısa açıklama metni ve ilk 3 etken sade dille gösterilir. |
| R-07 | Gerçek kullanıcı geri bildirimi sayısının hedefin altında kalması | Orta | Orta | Açık | En az 20 çekirdek senaryo ile teknik doğrulama sürdürülür; kullanıcı geri bildirimi toplanan sayıya göre nitel raporlama derinleştirilir. |

## Risk İzleme Kuralı

- Yüksek etki içeren açık riskler, her sprint sonunda yeniden değerlendirilir.
- “Azaltıldı” durumuna geçen risklerin kanıtı ilgili rapor/doküman bağlantısıyla kayıt altına alınır.
- Kritik risklerde kapsam daraltma kararı alınırsa gerekçe `degisim_kaydi.md` dosyasına işlenir.
