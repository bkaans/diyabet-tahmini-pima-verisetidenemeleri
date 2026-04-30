"""Yeni maksimum skor arama yardimcilari icin testler."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from makine_ogrenmesi.kaynak.maksimum_skor_arama import (
    BEKLENEN_SATIR_SAYISI,
    BEKLENEN_SINIF_DAGILIMI,
    KlinikOzellikUretici,
    SifirDegerDonusturucu,
    veri_butunlugu_ozeti,
)
from makine_ogrenmesi.kaynak.literatur_deneyleri import literatur_deneyleri_calistir


def test_veri_butunlugu_ozeti_pima_dagilimini_dogrulamali() -> None:
    proje_koku = Path(__file__).resolve().parents[1]
    veri_yolu = proje_koku / "makine_ogrenmesi" / "veri" / "ham" / "diabetes.csv"

    ozet = veri_butunlugu_ozeti(veri_yolu)

    assert ozet["satir_sayisi"] == BEKLENEN_SATIR_SAYISI
    assert ozet["sinif_dagilimi"] == BEKLENEN_SINIF_DAGILIMI
    assert len(str(ozet["sha256"])) == 64


def test_sifir_deger_donusturucu_secilen_kolonlarda_nan_uretmeli() -> None:
    veri = pd.DataFrame(
        {
            "Pregnancies": [0],
            "Glucose": [0],
            "BloodPressure": [0],
            "SkinThickness": [0],
            "Insulin": [0],
            "BMI": [0.0],
            "DiabetesPedigreeFunction": [0.5],
            "Age": [31],
        }
    )

    sonuc = SifirDegerDonusturucu(strategy="nan").transform(veri)

    assert sonuc["Pregnancies"].iloc[0] == 0
    assert sonuc["Glucose"].isna().iloc[0]
    assert sonuc["BMI"].isna().iloc[0]


def test_klinik_ozellik_uretici_ham_kolonlari_koruyup_turetilmis_kolon_eklemeli() -> None:
    veri = pd.DataFrame(
        {
            "Pregnancies": [2],
            "Glucose": [100],
            "BloodPressure": [70],
            "SkinThickness": [20],
            "Insulin": [50],
            "BMI": [30.0],
            "DiabetesPedigreeFunction": [0.5],
            "Age": [40],
        }
    )

    sonuc = KlinikOzellikUretici(strategy="basic").transform(veri)

    assert "Glucose" in sonuc.columns
    assert "Glucose_BMI" in sonuc.columns
    assert sonuc["Glucose_BMI"].iloc[0] == 3000


def test_literatur_deneyleri_ham_csvyi_degistirmeden_rapor_uretmeli(tmp_path: Path) -> None:
    proje_koku = Path(__file__).resolve().parents[1]
    veri_yolu = proje_koku / "makine_ogrenmesi" / "veri" / "ham" / "diabetes.csv"
    baslangic = veri_butunlugu_ozeti(veri_yolu)

    rapor = literatur_deneyleri_calistir(
        veri_yolu=veri_yolu,
        proje_koku=tmp_path,
        n_iter=1,
        n_jobs=1,
        deney_adlari=("hossain_knn",),
        cv_adlari=("stratified_5fold",),
    )

    bitis = veri_butunlugu_ozeti(veri_yolu)
    assert bitis["sha256"] == baslangic["sha256"]
    assert rapor["tamamlanan_deney_sayisi"] == 1
    assert rapor["sonuclar"][0]["ham_csv_degisti_mi"] is False
    assert (
        tmp_path
        / "makine_ogrenmesi"
        / "raporlar"
        / "degerlendirme"
        / "literatur_deneyleri.json"
    ).exists()
