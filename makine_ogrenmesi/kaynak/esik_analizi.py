"""Esik analizi: Youden J, F2 ve risk kategori mantigi."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def youden_j_esigi_hesapla(
    y_gercek: Sequence[int],
    y_olasilik: Sequence[float],
) -> dict[str, float]:
    """Youden J (TPR - FPR) degerine gore en iyi esigi hesaplar."""
    y_gercek_np = _diziye_cevir(y_gercek)
    y_olasilik_np = _diziye_cevir(y_olasilik)
    _uzunluklari_dogrula(y_gercek_np, y_olasilik_np)
    _olasilik_araligini_dogrula(y_olasilik_np)
    _ikili_etiket_dogrula(y_gercek_np)

    fpr, tpr, thresholds = roc_curve(y_gercek_np, y_olasilik_np)
    youden_j = tpr - fpr
    thresholds_np = np.asarray(thresholds, dtype=float)

    sonlu_indeksler = np.where(np.isfinite(thresholds_np))[0]
    if len(sonlu_indeksler) == 0:
        raise ValueError("Youden J hesabi icin gecerli esik bulunamadi.")

    en_iyi_indeks = sonlu_indeksler[int(np.argmax(youden_j[sonlu_indeksler]))]

    return {
        "esik": float(thresholds_np[en_iyi_indeks]),
        "youden_j": float(youden_j[en_iyi_indeks]),
        "tpr": float(tpr[en_iyi_indeks]),
        "fpr": float(fpr[en_iyi_indeks]),
    }


def f2_esigi_hesapla(
    y_gercek: Sequence[int],
    y_olasilik: Sequence[float],
    beta: float = 2.0,
) -> dict[str, float]:
    """F2 odakli en iyi esigi hesaplar."""
    if beta <= 0:
        raise ValueError("beta degeri sifirdan buyuk olmalidir.")

    y_gercek_np = _diziye_cevir(y_gercek)
    y_olasilik_np = _diziye_cevir(y_olasilik)
    _uzunluklari_dogrula(y_gercek_np, y_olasilik_np)
    _olasilik_araligini_dogrula(y_olasilik_np)
    _ikili_etiket_dogrula(y_gercek_np)

    precision, recall, thresholds = precision_recall_curve(y_gercek_np, y_olasilik_np)
    thresholds_np = np.asarray(thresholds, dtype=float)
    if thresholds_np.size == 0:
        raise ValueError("F2 hesabi icin gecerli esik bulunamadi.")

    precision_np = np.asarray(precision[:-1], dtype=float)
    recall_np = np.asarray(recall[:-1], dtype=float)

    beta_kare = beta**2
    payda = beta_kare * precision_np + recall_np
    f2_skorlari = np.zeros_like(payda)
    gecerli = payda > 0
    f2_skorlari[gecerli] = (1 + beta_kare) * precision_np[gecerli] * recall_np[gecerli] / payda[
        gecerli
    ]

    en_iyi_indeks = int(np.argmax(f2_skorlari))

    return {
        "esik": float(thresholds_np[en_iyi_indeks]),
        "f2_skoru": float(f2_skorlari[en_iyi_indeks]),
        "precision": float(precision_np[en_iyi_indeks]),
        "recall": float(recall_np[en_iyi_indeks]),
        "beta": float(beta),
    }


def risk_esiklerini_olustur(
    youden_j_esigi: float,
    f2_esigi: float,
) -> dict[str, float]:
    """Risk seviyeleri icin sabit sinirlari uretir."""
    _esik_dogrula(youden_j_esigi)
    _esik_dogrula(f2_esigi)

    return {
        "dusuk_ust_esik": 0.33,
        "orta_ust_esik": 0.66,
    }


def risk_kategorisi_belirle(
    olasilik: float,
    dusuk_ust_esik: float,
    orta_ust_esik: float,
) -> str:
    """Olasiliga gore risk kategorisi dondurur."""
    _esik_dogrula(olasilik)
    _esik_dogrula(dusuk_ust_esik)
    _esik_dogrula(orta_ust_esik)
    if dusuk_ust_esik > orta_ust_esik:
        raise ValueError("dusuk_ust_esik, orta_ust_esik degerinden buyuk olamaz.")

    if olasilik < dusuk_ust_esik:
        return "dusuk"
    if olasilik < orta_ust_esik:
        return "orta"
    return "yuksek"


def esik_yapilandirmasi_olustur(
    y_gercek: Sequence[int],
    y_olasilik: Sequence[float],
    beta: float = 2.0,
) -> dict[str, Any]:
    """esik_yapilandirmasi.json icin kayda hazir veri yapisi uretir."""
    youden = youden_j_esigi_hesapla(y_gercek, y_olasilik)
    f2 = f2_esigi_hesapla(y_gercek, y_olasilik, beta=beta)
    risk_esikleri = risk_esiklerini_olustur(youden["esik"], f2["esik"])

    return {
        "ikili_siniflama_esikleri": {
            "youden_j": youden,
            "f2": f2,
        },
        "onerilen_ikili_siniflama_esigi": float(f2["esik"]),
        "onerilen_ikili_siniflama_yontemi": "f2",
        "risk_kategorileri": {
            "dusuk_ust_esik": risk_esikleri["dusuk_ust_esik"],
            "orta_ust_esik": risk_esikleri["orta_ust_esik"],
            "etiketler": ["dusuk", "orta", "yuksek"],
        },
    }


def _diziye_cevir(veri: Sequence[int] | Sequence[float]) -> np.ndarray:
    dizi = np.asarray(veri)
    if dizi.ndim != 1:
        raise ValueError("Girdi dizileri tek boyutlu olmalidir.")
    return dizi


def _uzunluklari_dogrula(*diziler: np.ndarray) -> None:
    if not diziler:
        raise ValueError("En az bir dizi verilmelidir.")
    hedef_uzunluk = len(diziler[0])
    if hedef_uzunluk == 0:
        raise ValueError("Diziler bos olamaz.")
    if any(len(dizi) != hedef_uzunluk for dizi in diziler[1:]):
        raise ValueError("Tum girdi dizilerinin uzunlugu ayni olmalidir.")


def _olasilik_araligini_dogrula(y_olasilik: np.ndarray) -> None:
    if np.any(y_olasilik < 0) or np.any(y_olasilik > 1):
        raise ValueError("Olasilik degerleri 0 ile 1 araliginda olmalidir.")


def _ikili_etiket_dogrula(y_gercek: np.ndarray) -> None:
    benzersiz = set(np.unique(y_gercek).tolist())
    if not benzersiz.issubset({0, 1}):
        raise ValueError("y_gercek yalnizca 0 ve 1 degerlerini icermelidir.")
    if len(benzersiz) < 2:
        raise ValueError("y_gercek icinde hem 0 hem 1 sinifi bulunmalidir.")


def _esik_dogrula(esik: float) -> None:
    if esik < 0 or esik > 1:
        raise ValueError("Esik degeri 0 ile 1 araliginda olmalidir.")
