# Dagitim Notlari

Bu notlar Ubuntu tabanli bir VPS icin temel dagitim akisidir.

## 1. Hazirlik

- Sunucuda su paketleri kurulu olmali:
  - `python3`
  - `python3-venv`
  - `nginx`
- Proje klasoru ornekte su dizine alinmistir:
  - `/opt/diyabet_risk_tahmini`

## 2. Projeyi Hazirlama

```bash
cd /opt
git clone <repo-url> diyabet_risk_tahmini
cd diyabet_risk_tahmini
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`.env` dosyasini olustur:

```bash
cp .env.example .env
```

Gerekirse `APP_ENV`, `APP_HOST`, `APP_PORT`, `MODEL_ARTIFACT_KLASORU` degerlerini guncelle.

## 3. Gunicorn'u Elle Deneme

```bash
cd /opt/diyabet_risk_tahmini
source .venv/bin/activate
gunicorn -c dagitim/gunicorn_conf.py uygulama.main:app
```

Beklenen: Uygulama hata vermeden ayaga kalkar.

## 4. systemd Servisini Kurma

Servis dosyasini sistem klasorune kopyala:

```bash
sudo cp dagitim/diyabet_risk.service /etc/systemd/system/diyabet_risk.service
```

Servisi etkinlestir ve baslat:

```bash
sudo systemctl daemon-reload
sudo systemctl enable diyabet_risk
sudo systemctl start diyabet_risk
sudo systemctl status diyabet_risk
```

Beklenen: servis `active (running)` gorunur.

## 5. Nginx Reverse Proxy Kurulumu

Nginx site dosyasini yerlestir:

```bash
sudo cp dagitim/nginx.conf /etc/nginx/sites-available/diyabet_risk
sudo ln -s /etc/nginx/sites-available/diyabet_risk /etc/nginx/sites-enabled/diyabet_risk
sudo nginx -t
sudo systemctl reload nginx
```

Beklenen: `nginx -t` basarili olur ve istekler uygulamaya yonlenir.

## 6. HTTPS Notlari

Ornek sertifika kurulumu (Let's Encrypt):

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d ornek-alan-adi.com
```

Sertifika yenileme testi:

```bash
sudo certbot renew --dry-run
```

## 7. Son Kontrol

- `http://<sunucu-ip>/health` veya alan adinda `/health` endpointini test et.
- Ana sayfa ve form submit akisinin calistigini dogrula.
- Loglar:

```bash
sudo journalctl -u diyabet_risk -f
```

## 8. Dikkat Edilecek Noktalar

- `nginx.conf` icindeki `server_name` degerini kendi alan adinla degistir.
- `nginx.conf` ve `diyabet_risk.service` icindeki yol/soket degerleri birbiriyle uyumlu olmali.
- `User` ve `Group` degerleri sunucu kullanicisina gore guncellenebilir.
