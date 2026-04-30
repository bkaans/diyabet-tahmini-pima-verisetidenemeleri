"""Tahmin servisi testleri."""

from __future__ import annotations

import pytest

from uygulama.semalar.girdi_semalari import TahminGirdisi
from uygulama.servisler.tahmin_servisi import tek_ornek_tahmin_uret


GECERLI_GIRDI = TahminGirdisi(
    pregnancies=2,
    glucose=148,
    blood_pressure=72,
    skin_thickness=35,
    insulin=0,
    bmi=33.6,
    diabetes_pedigree_function=0.627,
    age=50,
)


def test_tek_ornek_tahmin_beklenen_alanlari_donmeli(monkeypatch) -> None:
    monkeypatch.setattr(
        "uygulama.servisler.tahmin_servisi._guvenli_top_faktorleri_uret",
        lambda **_: [
            {
                "ozellik": "glucose",
                "ozellik_degeri": 148.0,
                "shap_katkisi": 0.25,
                "yon": "arttirici",
            }
        ],
    )

    sonuc = tek_ornek_tahmin_uret(GECERLI_GIRDI)

    assert set(sonuc.keys()) == {
        "olasilik",
        "sinif",
        "risk_kategorisi",
        "top_faktorler",
        "kisa_aciklama",
    }
    assert 0 <= float(sonuc["olasilik"]) <= 1
    assert sonuc["sinif"] in (0, 1)
    assert sonuc["risk_kategorisi"] in {"dusuk", "orta", "yuksek"}
    assert isinstance(sonuc["top_faktorler"], list)


def test_tek_ornek_tahmin_eksik_alanli_dictte_hata_vermeli() -> None:
    eksik_girdi = {
        "pregnancies": 2,
        "glucose": 148,
        "blood_pressure": 72,
        "skin_thickness": 35,
        "insulin": 0,
        "bmi": 33.6,
        "diabetes_pedigree_function": 0.627,
    }

    with pytest.raises(ValueError, match="eksik alanlar"):
        tek_ornek_tahmin_uret(eksik_girdi)
