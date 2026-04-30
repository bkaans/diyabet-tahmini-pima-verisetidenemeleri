"""Deploy metrikleri icin resmi skor tablosu uretir."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJE_KOKU = Path(__file__).resolve().parents[1]
if str(PROJE_KOKU) not in sys.path:
    sys.path.insert(0, str(PROJE_KOKU))

from makine_ogrenmesi.kaynak.artifact_kaydet import artifactleri_yukle
from makine_ogrenmesi.kaynak.ozellik_yapilandirmasi import HEDEF_KOLONU, OZELLIK_KOLONLARI
from makine_ogrenmesi.kaynak.veri_yukleyici import veri_setini_yukle

LITERATUR_BANTLARI = {
    "accuracy": (0.75, 0.84),
    "f1": (0.62, 0.85),
    "roc_auc": (0.80, 0.87),
    "precision": (0.57, 0.93),
    "recall": (0.65, 0.90),
    "brier": (0.15, 0.20),
}


def argumanlari_oku() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resmi skor tablosu olusturma betigi")
    parser.add_argument(
        "--veri-yolu",
        type=Path,
        default=PROJE_KOKU / "makine_ogrenmesi" / "veri" / "ham" / "diabetes.csv",
    )
    parser.add_argument(
        "--degerlendirme-yolu",
        type=Path,
        default=PROJE_KOKU
        / "makine_ogrenmesi"
        / "raporlar"
        / "degerlendirme"
        / "model_degerlendirme_ozeti.json",
    )
    parser.add_argument(
        "--artifact-klasoru",
        type=Path,
        default=PROJE_KOKU / "makine_ogrenmesi" / "artifactler",
    )
    parser.add_argument(
        "--json-cikti-yolu",
        type=Path,
        default=PROJE_KOKU
        / "makine_ogrenmesi"
        / "raporlar"
        / "degerlendirme"
        / "resmi_skor_tablosu.json",
    )
    parser.add_argument(
        "--md-cikti-yolu",
        type=Path,
        default=PROJE_KOKU / "dokumanlar" / "resmi_skor_tablosu.md",
    )
    parser.add_argument("--test-boyutu", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = argumanlari_oku()
    degerlendirme_ozeti = _json_oku(args.degerlendirme_yolu)
    artifactler = artifactleri_yukle(args.artifact_klasoru)
    x_test, y_test = _test_setini_hazirla(args.veri_yolu, args.test_boyutu, args.random_state)

    en_iyi_pipeline = artifactler["en_iyi_pipeline"]
    kalibrator = artifactler["kalibrator"]
    esik_yapilandirmasi = artifactler["esik_yapilandirmasi"]
    model_metadata = artifactler["model_metadata"]

    deploy_esigi = float(esik_yapilandirmasi["onerilen_ikili_siniflama_esigi"])
    y_olasilik_once = en_iyi_pipeline.predict_proba(x_test)[:, 1]
    y_olasilik_sonra = kalibrator.predict_proba(x_test)[:, 1]

    referans_metrikler = _siniflandirma_metrikleri(y_test, y_olasilik_sonra, 0.50)
    deploy_metrikleri = _siniflandirma_metrikleri(y_test, y_olasilik_sonra, deploy_esigi)
    brier_once = float(brier_score_loss(y_test, y_olasilik_once))
    brier_sonra = float(brier_score_loss(y_test, y_olasilik_sonra))
    brier_iyilesme_orani = (brier_once - brier_sonra) / brier_once if brier_once else 0.0

    hedefler = {
        "accuracy_min": 0.78,
        "roc_auc_min": 0.80,
        "f1_min": 0.70,
        "brier_iyilesme_orani_min": 0.10,
    }
    hedef_durumu = {
        "accuracy_hedefi_saglandi": deploy_metrikleri["accuracy"] >= hedefler["accuracy_min"],
        "roc_auc_hedefi_saglandi": deploy_metrikleri["roc_auc"] >= hedefler["roc_auc_min"],
        "f1_hedefi_saglandi": deploy_metrikleri["f1"] >= hedefler["f1_min"],
        "brier_iyilesme_hedefi_saglandi": brier_iyilesme_orani
        >= hedefler["brier_iyilesme_orani_min"],
    }
    hedef_durumu["genel_durum"] = all(hedef_durumu.values())

    ham_en_iyi = degerlendirme_ozeti.get("en_iyi_model", {})
    degerlendirme_deploy_farki = _degerlendirme_deploy_farki_hazirla(
        ham_en_iyi=ham_en_iyi,
        deploy_metrikleri=deploy_metrikleri,
        deploy_esigi=deploy_esigi,
        deploy_kalibrasyon=model_metadata.get("kalibrasyon_yontemi"),
    )

    resmi_ozet = {
        "olusturma_zamani_utc": datetime.now(timezone.utc).isoformat(),
        "veri": {
            "veri_yolu": str(args.veri_yolu),
            "test_boyutu": args.test_boyutu,
            "random_state": args.random_state,
            "test_ornek_sayisi": int(len(y_test)),
        },
        "hedefler": hedefler,
        "deploy": {
            "model_adi": model_metadata.get("model_adi"),
            "kalibrasyon_yontemi": model_metadata.get("kalibrasyon_yontemi"),
            "ikili_siniflama_yontemi": model_metadata.get("ikili_siniflama_yontemi"),
            "ikili_siniflama_esigi": deploy_esigi,
            "risk_kategorileri": esik_yapilandirmasi.get("risk_kategorileri", {}),
            "metrikler": deploy_metrikleri,
            "esik_05_referans_metrikleri": referans_metrikler,
        },
        "kalibrasyon": {
            "brier_once": brier_once,
            "brier_sonra": brier_sonra,
            "brier_iyilesme_orani": brier_iyilesme_orani,
            "brier_iyilesme_yuzde": brier_iyilesme_orani * 100.0,
        },
        "model_degerlendirme_raporu": {
            "rapor_yolu": str(args.degerlendirme_yolu),
            "en_iyi_model": ham_en_iyi,
            "sirali_sonuclar": degerlendirme_ozeti.get("sirali_sonuclar", []),
        },
        "degerlendirme_deploy_farki": degerlendirme_deploy_farki,
        "literatur_benchmarklari": _literatur_benchmarklarini_hazirla(),
        "hedef_durumu": hedef_durumu,
    }

    args.json_cikti_yolu.parent.mkdir(parents=True, exist_ok=True)
    args.json_cikti_yolu.write_text(
        json.dumps(_json_uyumlu(resmi_ozet), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    args.md_cikti_yolu.parent.mkdir(parents=True, exist_ok=True)
    args.md_cikti_yolu.write_text(_markdown_uret(resmi_ozet), encoding="utf-8")

    print(f"JSON raporu olusturuldu: {args.json_cikti_yolu}")
    print(f"Markdown raporu olusturuldu: {args.md_cikti_yolu}")


def _test_setini_hazirla(
    veri_yolu: Path,
    test_boyutu: float,
    random_state: int,
) -> tuple[Any, np.ndarray]:
    veri_cercevesi = veri_setini_yukle(veri_yolu)
    ozellikler = veri_cercevesi[OZELLIK_KOLONLARI]
    hedef = veri_cercevesi[HEDEF_KOLONU]

    _, x_test, _, y_test = train_test_split(
        ozellikler,
        hedef,
        test_size=test_boyutu,
        random_state=random_state,
        stratify=hedef,
    )
    return x_test, np.asarray(y_test)


def _siniflandirma_metrikleri(
    y_gercek: np.ndarray,
    y_olasilik: np.ndarray,
    esik: float,
) -> dict[str, float]:
    y_tahmin = (np.asarray(y_olasilik) >= esik).astype(int)
    return {
        "accuracy": float(accuracy_score(y_gercek, y_tahmin)),
        "precision": float(precision_score(y_gercek, y_tahmin, zero_division=0)),
        "recall": float(recall_score(y_gercek, y_tahmin, zero_division=0)),
        "f1": float(f1_score(y_gercek, y_tahmin, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_gercek, y_olasilik)),
        "brier": float(brier_score_loss(y_gercek, y_olasilik)),
        "siniflama_esigi": float(esik),
    }


def _json_oku(dosya_yolu: Path) -> dict[str, Any]:
    return json.loads(dosya_yolu.read_text(encoding="utf-8"))


def _degerlendirme_deploy_farki_hazirla(
    ham_en_iyi: dict[str, Any],
    deploy_metrikleri: dict[str, float],
    deploy_esigi: float,
    deploy_kalibrasyon: str | None,
) -> dict[str, Any]:
    metrik_anahtarlari = ("accuracy", "precision", "recall", "f1", "roc_auc", "brier")
    metrik_farklari: dict[str, dict[str, float | None]] = {}
    for anahtar in metrik_anahtarlari:
        ham_deger = (
            float(ham_en_iyi[anahtar])
            if ham_en_iyi and ham_en_iyi.get(anahtar) is not None
            else None
        )
        deploy_deger = float(deploy_metrikleri[anahtar])
        fark = (deploy_deger - ham_deger) if ham_deger is not None else None
        metrik_farklari[anahtar] = {
            "ham_degerlendirme": ham_deger,
            "deploy": deploy_deger,
            "fark": fark,
        }

    return {
        "ham_model_adi": ham_en_iyi.get("model_adi"),
        "ham_degerlendirme_esigi": 0.50,
        "deploy_kalibrasyon_yontemi": deploy_kalibrasyon,
        "deploy_esigi": float(deploy_esigi),
        "metrik_farklari": metrik_farklari,
        "aciklama": (
            "Model degerlendirme raporu, ham model adaylarini test setinde "
            "0.50 siniflama esigi ile karsilastirir. Deploy metrikleri ise "
            "secilen model + secilen kalibrasyon + optimize edilen ikili "
            "siniflama esigi ile hesaplandigi icin fark olusmasi normaldir."
        ),
    }


def _literatur_benchmarklarini_hazirla() -> dict[str, Any]:
    return {
        "veri_seti": "Pima Indians Diabetes (768 satir)",
        "guvenilir_bantlar": {
            metrik: {"alt": alt, "ust": ust}
            for metrik, (alt, ust) in LITERATUR_BANTLARI.items()
        },
        "metodoloji_notlari": [
            "Kucuk veri setlerinde tek bolme ile raporlanan cok yuksek sonuclarin genellenebilirligi dusuktur.",
            "SMOTE yalnizca egitim katmaninda uygulanmalidir; test verisine uygulanmasi veri sizintisi olusturur.",
            "PIMA icin %98-%100 accuracy iddialari cogu durumda overfit veya leakage riski tasir.",
            "Bu projede stratified split, capraz dogrulama ve kalibrasyon ayrimi korunmustur.",
        ],
    }


def _fmt_sayi(deger: float) -> str:
    return f"{deger:.4f}"


def _fmt_yuzde(deger: float) -> str:
    return f"%{deger * 100:.2f}"


def _fmt_arti_eksi(deger: float | None) -> str:
    if deger is None:
        return "-"
    return f"{deger:+.4f}"


def _durum_etiketi(durum: bool) -> str:
    return "Sağlandı" if durum else "Sağlanmadı"


def _markdown_uret(resmi_ozet: dict[str, Any]) -> str:
    deploy = resmi_ozet["deploy"]
    metrikler = deploy["metrikler"]
    kalibrasyon = resmi_ozet["kalibrasyon"]
    hedefler = resmi_ozet["hedefler"]
    hedef_durumu = resmi_ozet["hedef_durumu"]
    model_sonuclari = resmi_ozet["model_degerlendirme_raporu"]["sirali_sonuclar"]
    degerlendirme_deploy_farki = resmi_ozet["degerlendirme_deploy_farki"]
    literatur_benchmarklari = resmi_ozet["literatur_benchmarklari"]

    hedef_satirlari = [
        (
            "Accuracy",
            f">= {_fmt_sayi(hedefler['accuracy_min'])}",
            _fmt_sayi(metrikler["accuracy"]),
            _durum_etiketi(hedef_durumu["accuracy_hedefi_saglandi"]),
        ),
        (
            "ROC AUC",
            f">= {_fmt_sayi(hedefler['roc_auc_min'])}",
            _fmt_sayi(metrikler["roc_auc"]),
            _durum_etiketi(hedef_durumu["roc_auc_hedefi_saglandi"]),
        ),
        (
            "F1",
            f">= {_fmt_sayi(hedefler['f1_min'])}",
            _fmt_sayi(metrikler["f1"]),
            _durum_etiketi(hedef_durumu["f1_hedefi_saglandi"]),
        ),
        (
            "Brier iyileşme",
            f">= {_fmt_yuzde(hedefler['brier_iyilesme_orani_min'])}",
            _fmt_yuzde(kalibrasyon["brier_iyilesme_orani"]),
            _durum_etiketi(hedef_durumu["brier_iyilesme_hedefi_saglandi"]),
        ),
    ]

    model_satirlari = []
    for sonuc in model_sonuclari:
        model_satirlari.append(
            "| {model} | {acc} | {prec} | {rec} | {f1} | {auc} | {brier} |".format(
                model=sonuc.get("model_adi", "-"),
                acc=_fmt_sayi(float(sonuc.get("accuracy", 0.0))),
                prec=_fmt_sayi(float(sonuc.get("precision", 0.0))),
                rec=_fmt_sayi(float(sonuc.get("recall", 0.0))),
                f1=_fmt_sayi(float(sonuc.get("f1", 0.0))),
                auc=_fmt_sayi(float(sonuc.get("roc_auc", 0.0))),
                brier=_fmt_sayi(float(sonuc.get("brier", 0.0))),
            )
        )

    hedef_tablo = "\n".join(
        f"| {metrik} | {hedef} | {gercek} | {durum} |"
        for metrik, hedef, gercek, durum in hedef_satirlari
    )
    model_tablo = "\n".join(model_satirlari) if model_satirlari else "| - | - | - | - | - | - | - |"
    fark_satirlari = []
    for metrik in ("accuracy", "precision", "recall", "f1", "roc_auc", "brier"):
        metrik_farki = degerlendirme_deploy_farki["metrik_farklari"][metrik]
        ham = (
            _fmt_sayi(float(metrik_farki["ham_degerlendirme"]))
            if metrik_farki["ham_degerlendirme"] is not None
            else "-"
        )
        deploy_degeri = _fmt_sayi(float(metrik_farki["deploy"]))
        fark = _fmt_arti_eksi(
            float(metrik_farki["fark"]) if metrik_farki["fark"] is not None else None
        )
        fark_satirlari.append(
            "| {metrik} | {ham} | {deploy} | {fark} |".format(
                metrik=metrik.upper(),
                ham=ham,
                deploy=deploy_degeri,
                fark=fark,
            )
        )
    fark_tablo = "\n".join(fark_satirlari)

    literatur_satirlari = []
    for metrik, bant in literatur_benchmarklari.get("guvenilir_bantlar", {}).items():
        literatur_satirlari.append(
            "| {metrik} | {alt} | {ust} |".format(
                metrik=metrik.upper(),
                alt=_fmt_sayi(float(bant["alt"])),
                ust=_fmt_sayi(float(bant["ust"])),
            )
        )
    literatur_tablo = (
        "\n".join(literatur_satirlari) if literatur_satirlari else "| - | - | - |"
    )
    metodoloji_notlari = literatur_benchmarklari.get("metodoloji_notlari", [])
    metodoloji_notlari_md = (
        "\n".join(f"- {satir}" for satir in metodoloji_notlari)
        if metodoloji_notlari
        else "- Ek metodoloji notu bulunmuyor."
    )

    return f"""# Resmi Skor Tablosu

## 1. Deploy Özeti

- Model: `{deploy.get("model_adi")}`
- Kalibrasyon: `{deploy.get("kalibrasyon_yontemi")}`
- İkili sınıflama yöntemi: `{deploy.get("ikili_siniflama_yontemi")}`
- İkili sınıflama eşiği: `{_fmt_sayi(float(deploy.get("ikili_siniflama_esigi", 0.0)))}`

### 1.1 Hedef Uyum Tablosu

| Metrik | Hedef | Gerçekleşen | Durum |
| --- | --- | --- | --- |
{hedef_tablo}

### 1.2 Deploy Metrikleri

| Metrik | Değer |
| --- | --- |
| Accuracy | {_fmt_sayi(metrikler["accuracy"])} |
| Precision | {_fmt_sayi(metrikler["precision"])} |
| Recall | {_fmt_sayi(metrikler["recall"])} |
| F1 | {_fmt_sayi(metrikler["f1"])} |
| ROC AUC | {_fmt_sayi(metrikler["roc_auc"])} |
| Brier | {_fmt_sayi(metrikler["brier"])} |

## 2. Kalibrasyon Etkisi

| Kalibrasyon Durumu | Brier |
| --- | --- |
| Kalibrasyon öncesi | {_fmt_sayi(kalibrasyon["brier_once"])} |
| Kalibrasyon sonrası | {_fmt_sayi(kalibrasyon["brier_sonra"])} |
| İyileşme oranı | {_fmt_yuzde(kalibrasyon["brier_iyilesme_orani"])} |

## 3. Model Karşılaştırma (Test Seti)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- |
{model_tablo}

## 4. Model Değerlendirme ve Deploy Farkı

- Ham model kıyas eşiği: `{_fmt_sayi(float(degerlendirme_deploy_farki["ham_degerlendirme_esigi"]))}`
- Deploy sınıflama eşiği: `{_fmt_sayi(float(degerlendirme_deploy_farki["deploy_esigi"]))}`
- Deploy kalibrasyon yöntemi: `{degerlendirme_deploy_farki.get("deploy_kalibrasyon_yontemi")}`

| Metrik | Ham Değerlendirme | Deploy | Fark (Deploy - Ham) |
| --- | --- | --- | --- |
{fark_tablo}

{degerlendirme_deploy_farki["aciklama"]}

## 5. PIMA Literatür Benchmark ve Metodoloji Notu

| Metrik | Güvenilir Alt Bant | Güvenilir Üst Bant |
| --- | --- | --- |
{literatur_tablo}

{metodoloji_notlari_md}

## 6. Dokümandaki Hedefleri Yakalamak İçin Teknik Yol Haritası

1. Veri hacmini artırın: Pima veri seti `768` kayıt olduğu için `%90+ accuracy` hedefi aşırı iddialı kalıyor. Benzer dağılımda en az `3.000+` kayıtlık ek veri, hedef metriklerde anlamlı stabilite sağlar.
2. Özellikleri zenginleştirin: HbA1c, bel çevresi, aile öyküsü detay seviyesi, ilaç kullanımı ve geçmiş gebelik diyabet öyküsü gibi klinik olarak daha ayırt edici değişkenler performansı doğrudan etkiler.
3. Veri kalitesini standartlaştırın: Eksik/0 değerlerin ölçüm kaynaklı mı gerçek sıfır mı olduğu saha bazlı temizlenmeli; etiketleme hataları için en az bir klinik uzmanla çift kontrol yapılmalıdır.
4. Hedefe göre eşik yönetimi uygulayın: F1 odağı için mevcut eşik korunurken, tarama senaryosunda recall odaklı alternatif eşik ayrıca dokümante edilmelidir.
5. Dış doğrulama yapın: Tek veri seti yerine farklı merkezden ayrık bir dış test seti olmadan yüksek metrik iddiası paydaş açısından zayıf kalır.
6. Brier iyileşmesini büyütmek için: Kalibrasyon verisi büyütülüp (ayrı validation katmanı) isotonic/sigmoid seçimi her yeni veri partisinde yeniden yapılmalıdır.
"""


def _json_uyumlu(veri: Any) -> Any:
    if isinstance(veri, dict):
        return {str(k): _json_uyumlu(v) for k, v in veri.items()}
    if isinstance(veri, list):
        return [_json_uyumlu(v) for v in veri]
    if isinstance(veri, np.generic):
        return veri.item()
    return veri


if __name__ == "__main__":
    main()
