"""Sema katmani icin ortak dogrulama kurallari."""

from __future__ import annotations

import math


ALAN_ARALIKLARI: dict[str, tuple[float, float]] = {
    "pregnancies": (0, 20),
    "glucose": (0, 300),
    "blood_pressure": (0, 200),
    "skin_thickness": (0, 120),
    "insulin": (0, 1000),
    "bmi": (0, 80),
    "diabetes_pedigree_function": (0, 3),
    "age": (18, 120),
}

RISK_KATEGORILERI = {"dusuk", "orta", "yuksek"}
YON_DEGERLERI = {"arttirici", "azaltici"}


def sayisal_aralik_dogrula(alan_adi: str, deger: float | int) -> float | int:
    """Sayisal bir degerin alan bazli aralik kurallarina uydugunu dogrular."""
    if alan_adi not in ALAN_ARALIKLARI:
        raise ValueError(f"Aralik tanimi bulunamadi: {alan_adi}")

    alt_sinir, ust_sinir = ALAN_ARALIKLARI[alan_adi]
    sayi = float(deger)

    if not math.isfinite(sayi):
        raise ValueError(f"{alan_adi} sonlu bir sayi olmalidir.")
    if sayi < alt_sinir or sayi > ust_sinir:
        raise ValueError(
            f"{alan_adi} degeri {alt_sinir} ile {ust_sinir} araliginda olmalidir."
        )

    return deger


def birim_aralik_dogrula(alan_adi: str, deger: float | int) -> float | int:
    """0-1 araligindaki olasilik degerlerini dogrular."""
    sayi = float(deger)
    if not math.isfinite(sayi):
        raise ValueError(f"{alan_adi} sonlu bir sayi olmalidir.")
    if sayi < 0 or sayi > 1:
        raise ValueError(f"{alan_adi} degeri 0 ile 1 araliginda olmalidir.")
    return deger


def risk_kategorisi_dogrula(deger: str) -> str:
    """Risk kategorisi alaninin desteklenen degerlerden biri oldugunu dogrular."""
    if deger not in RISK_KATEGORILERI:
        raise ValueError("risk_kategorisi sadece dusuk, orta veya yuksek olabilir.")
    return deger


def yon_dogrula(deger: str) -> str:
    """SHAP yon alaninin desteklenen degerlerden biri oldugunu dogrular."""
    if deger not in YON_DEGERLERI:
        raise ValueError("yon sadece arttirici veya azaltici olabilir.")
    return deger
