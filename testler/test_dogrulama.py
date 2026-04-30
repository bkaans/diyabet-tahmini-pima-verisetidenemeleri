"""Pydantic sema dogrulama testleri."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from uygulama.semalar.cikti_semalari import TahminCiktisi
from uygulama.semalar.girdi_semalari import TahminGirdisi


def test_tahmin_girdisi_gecerli_veriyi_kabul_etmeli() -> None:
    girdi = TahminGirdisi(
        pregnancies=2,
        glucose=148,
        blood_pressure=72,
        skin_thickness=35,
        insulin=0,
        bmi=33.6,
        diabetes_pedigree_function=0.627,
        age=50,
    )

    assert girdi.age == 50
    assert girdi.pregnancies == 2


def test_tahmin_girdisi_gecersiz_yasi_reddetmeli() -> None:
    with pytest.raises(ValidationError) as hata:
        TahminGirdisi(
            pregnancies=2,
            glucose=148,
            blood_pressure=72,
            skin_thickness=35,
            insulin=0,
            bmi=33.6,
            diabetes_pedigree_function=0.627,
            age=10,
        )

    ilk = hata.value.errors()[0]
    assert ilk["loc"] == ("age",)


def test_tahmin_ciktisi_gecersiz_olasiligi_reddetmeli() -> None:
    with pytest.raises(ValidationError) as hata:
        TahminCiktisi(
            olasilik=1.2,
            sinif=1,
            risk_kategorisi="yuksek",
            top_faktorler=[],
            kisa_aciklama="test",
        )

    ilk = hata.value.errors()[0]
    assert ilk["loc"] == ("olasilik",)
