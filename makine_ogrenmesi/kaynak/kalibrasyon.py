"""Kalibrasyon katmani: sigmoid/isotonic karsilastirma ve secim mantigi."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def brier_skoru_hesapla(y_gercek: Sequence[int], y_olasilik: Sequence[float]) -> float:
    """Brier score degerini hesaplar."""
    y_gercek_np = _diziye_cevir(y_gercek)
    y_olasilik_np = _diziye_cevir(y_olasilik)
    _uzunluklari_dogrula(y_gercek_np, y_olasilik_np)
    _olasilik_araligini_dogrula(y_olasilik_np)
    return float(brier_score_loss(y_gercek_np, y_olasilik_np))


def kalibratorleri_egit(
    model: BaseEstimator,
    x_egitim: Any,
    y_egitim: Sequence[int],
    cv: int = 5,
) -> dict[str, CalibratedClassifierCV]:
    """Sigmoid ve isotonic kalibratorleri egitir."""
    kalibratorler = {
        "sigmoid": _tek_kalibrator_egit(model, x_egitim, y_egitim, method="sigmoid", cv=cv),
        "isotonic": _tek_kalibrator_egit(
            model, x_egitim, y_egitim, method="isotonic", cv=cv
        ),
    }
    return kalibratorler


def kalibrasyon_karsilastir(
    model: BaseEstimator,
    x_kalibrasyon_egitim: Any,
    y_kalibrasyon_egitim: Sequence[int],
    x_degerlendirme: Any,
    y_degerlendirme: Sequence[int],
    cv: int = 5,
) -> dict[str, Any]:
    """Kalibrasyon oncesi/sonrasi Brier score karsilastirmasi yapar."""
    _modelde_predict_proba_kontrolu(model)

    y_olasilik_once = _pozitif_sinif_olasiligi(model, x_degerlendirme)
    brier_once = brier_skoru_hesapla(y_degerlendirme, y_olasilik_once)

    kalibratorler = kalibratorleri_egit(
        model=model,
        x_egitim=x_kalibrasyon_egitim,
        y_egitim=y_kalibrasyon_egitim,
        cv=cv,
    )

    y_olasilik_sigmoid = _pozitif_sinif_olasiligi(kalibratorler["sigmoid"], x_degerlendirme)
    y_olasilik_isotonic = _pozitif_sinif_olasiligi(
        kalibratorler["isotonic"], x_degerlendirme
    )

    brier_skorlari = {
        "once": brier_once,
        "sigmoid": brier_skoru_hesapla(y_degerlendirme, y_olasilik_sigmoid),
        "isotonic": brier_skoru_hesapla(y_degerlendirme, y_olasilik_isotonic),
    }

    en_iyi_yontem = en_iyi_kalibrasyon_yontemini_sec(brier_skorlari)

    return {
        "brier_skorlari": brier_skorlari,
        "en_iyi_yontem": en_iyi_yontem,
        "en_iyi_brier": float(brier_skorlari[en_iyi_yontem]),
        "en_iyi_kalibrator": kalibratorler[en_iyi_yontem],
        "kalibratorler": kalibratorler,
    }


def en_iyi_kalibrasyon_yontemini_sec(brier_skorlari: dict[str, float]) -> str:
    """Sigmoid ve isotonic arasinda en dusuk Brier score'u sec."""
    zorunlu_alanlar = {"sigmoid", "isotonic"}
    eksikler = zorunlu_alanlar - set(brier_skorlari.keys())
    if eksikler:
        eksik = ", ".join(sorted(eksikler))
        raise ValueError(f"Brier skorlarinda zorunlu alanlar eksik: {eksik}.")

    # Esitlik durumunda deterministic secim icin once sigmoid kontrol edilir.
    adaylar = ["sigmoid", "isotonic"]
    return min(adaylar, key=lambda yontem: float(brier_skorlari[yontem]))


def _tek_kalibrator_egit(
    model: BaseEstimator,
    x_egitim: Any,
    y_egitim: Sequence[int],
    method: str,
    cv: int,
) -> CalibratedClassifierCV:
    kalibrator = CalibratedClassifierCV(
        estimator=clone(model),
        method=method,
        cv=cv,
    )
    kalibrator.fit(x_egitim, y_egitim)
    return kalibrator


def _pozitif_sinif_olasiligi(model: BaseEstimator, x_veri: Any) -> np.ndarray:
    olasiliklar = model.predict_proba(x_veri)
    olasiliklar_np = np.asarray(olasiliklar)

    if olasiliklar_np.ndim != 2 or olasiliklar_np.shape[1] < 2:
        raise ValueError("predict_proba cikisi beklenen formatta degil.")

    return olasiliklar_np[:, 1]


def _modelde_predict_proba_kontrolu(model: BaseEstimator) -> None:
    if not hasattr(model, "predict_proba"):
        raise TypeError("Kalibrasyon icin modelde predict_proba metodu olmalidir.")


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
