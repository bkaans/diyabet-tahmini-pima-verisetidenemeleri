"""Risk seviyesi ve siniflama mantigi testleri."""

from __future__ import annotations

import pytest

from uygulama.servisler.risk_servisi import (
    risk_esiklerini_al,
    risk_kategorisini_normalize_et,
    risk_ozeti_hazirla,
)


ORNEK_ESIK_YAPILANDIRMASI = {
    "onerilen_ikili_siniflama_esigi": 0.40,
    "risk_kategorileri": {
        "dusuk_ust_esik": 0.33,
        "orta_ust_esik": 0.66,
        "etiketler": ["dusuk", "orta", "yuksek"],
    },
}


@pytest.mark.parametrize(
    ("olasilik", "beklenen_sinif", "beklenen_risk"),
    [
        (0.10, 0, "dusuk"),
        (0.40, 1, "orta"),
        (0.70, 1, "yuksek"),
        (0.85, 1, "yuksek"),
    ],
)
def test_risk_ozeti_hazirla_beklenen_sonuclari_donmeli(
    olasilik: float,
    beklenen_sinif: int,
    beklenen_risk: str,
) -> None:
    sonuc = risk_ozeti_hazirla(olasilik, ORNEK_ESIK_YAPILANDIRMASI)

    assert sonuc["sinif"] == beklenen_sinif
    assert sonuc["risk_kategorisi"] == beklenen_risk


def test_risk_esikleri_ters_sirada_ise_hata_vermeli() -> None:
    gecersiz = {
        "onerilen_ikili_siniflama_esigi": 0.40,
        "risk_kategorileri": {
            "dusuk_ust_esik": 0.80,
            "orta_ust_esik": 0.60,
        },
    }

    with pytest.raises(ValueError, match="buyuk olamaz"):
        risk_esiklerini_al(gecersiz)


@pytest.mark.parametrize(
    ("ham_kategori", "beklenen"),
    [
        ("dusuk", "dusuk"),
        ("çok düşük", "dusuk"),
        ("cok_dusuk", "dusuk"),
        ("orta", "orta"),
        ("medium", "orta"),
        ("yüksek", "yuksek"),
        ("cok_yuksek", "yuksek"),
        ("very-high", "yuksek"),
    ],
)
def test_risk_kategorisi_normalizasyonu_legacy_degerleri_desteklemeli(
    ham_kategori: str,
    beklenen: str,
) -> None:
    assert risk_kategorisini_normalize_et(ham_kategori) == beklenen


def test_risk_kategorisi_normalizasyonu_desteklenmeyen_degerde_hata_vermeli() -> None:
    with pytest.raises(ValueError, match="desteklenmeyen"):
        risk_kategorisini_normalize_et("belirsiz")
