"""Model degerlendirme, karsilastirma ve secim fonksiyonlari."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


SKORLAMA_METRIKLERI = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "brier": "neg_brier_score",
}


def skorlama_metriklerini_dondur() -> dict[str, str]:
    """Model arama adimlarinda kullanilacak sklearn scoring sozlugunu dondurur."""
    return SKORLAMA_METRIKLERI.copy()


def model_metriklerini_hesapla(
    y_gercek: Sequence[int],
    y_tahmin: Sequence[int],
    y_olasilik: Sequence[float],
) -> dict[str, float]:
    """Temel degerlendirme metriklerini tek noktada hesaplar."""
    y_gercek_np = _diziye_cevir(y_gercek)
    y_tahmin_np = _diziye_cevir(y_tahmin)
    y_olasilik_np = _diziye_cevir(y_olasilik)
    _uzunluklari_dogrula(y_gercek_np, y_tahmin_np, y_olasilik_np)

    return {
        "accuracy": float(accuracy_score(y_gercek_np, y_tahmin_np)),
        "precision": float(precision_score(y_gercek_np, y_tahmin_np, zero_division=0)),
        "recall": float(recall_score(y_gercek_np, y_tahmin_np, zero_division=0)),
        "f1": float(f1_score(y_gercek_np, y_tahmin_np, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_gercek_np, y_olasilik_np)),
        "brier": float(brier_score_loss(y_gercek_np, y_olasilik_np)),
    }


def model_sonuc_ozeti_olustur(
    model_adi: str,
    y_gercek: Sequence[int],
    y_tahmin: Sequence[int],
    y_olasilik: Sequence[float],
) -> dict[str, float | str]:
    """Tek modelin ad + metrikler seklinde ortak sonuc yapisini dondurur."""
    ozet = {"model_adi": model_adi}
    ozet.update(model_metriklerini_hesapla(y_gercek, y_tahmin, y_olasilik))
    return ozet


def confusion_matrix_ozeti_olustur(
    y_gercek: Sequence[int],
    y_tahmin: Sequence[int],
) -> dict[str, int | list]:
    """Confusion matrix'i ortak sozluk formatinda dondurur."""
    y_gercek_np = _diziye_cevir(y_gercek)
    y_tahmin_np = _diziye_cevir(y_tahmin)
    _uzunluklari_dogrula(y_gercek_np, y_tahmin_np)

    matris = confusion_matrix(y_gercek_np, y_tahmin_np, labels=[0, 1])
    return {
        "labels": [0, 1],
        "matris": matris.tolist(),
        "tn": int(matris[0, 0]),
        "fp": int(matris[0, 1]),
        "fn": int(matris[1, 0]),
        "tp": int(matris[1, 1]),
    }


def roc_verisi_hazirla(
    y_gercek: Sequence[int],
    y_olasilik: Sequence[float],
) -> dict[str, float | list[float]]:
    """ROC cizimi icin gerekli noktalar ve AUC degerini dondurur."""
    y_gercek_np = _diziye_cevir(y_gercek)
    y_olasilik_np = _diziye_cevir(y_olasilik)
    _uzunluklari_dogrula(y_gercek_np, y_olasilik_np)

    fpr, tpr, thresholds = roc_curve(y_gercek_np, y_olasilik_np)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "roc_auc": float(roc_auc_score(y_gercek_np, y_olasilik_np)),
    }


def pr_verisi_hazirla(
    y_gercek: Sequence[int],
    y_olasilik: Sequence[float],
) -> dict[str, float | list[float]]:
    """Precision-Recall cizimi icin gerekli noktalar ve PR AUC degerini dondurur."""
    y_gercek_np = _diziye_cevir(y_gercek)
    y_olasilik_np = _diziye_cevir(y_olasilik)
    _uzunluklari_dogrula(y_gercek_np, y_olasilik_np)

    precision, recall, thresholds = precision_recall_curve(y_gercek_np, y_olasilik_np)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "pr_auc": float(auc(recall, precision)),
    }


def en_iyi_modeli_sec(model_sonuclari: Sequence[dict[str, float | str]]) -> dict[str, float | str]:
    """Model secim mantigi: once AUC, sonra Recall, sonra F1, en sonda Accuracy."""
    if not model_sonuclari:
        raise ValueError("Model sonuclari bos olamaz.")

    zorunlu_alanlar = {"model_adi", "roc_auc", "recall", "f1", "accuracy"}
    _sonuc_alanlarini_dogrula(model_sonuclari, zorunlu_alanlar)

    siralanmis = sorted(
        model_sonuclari,
        key=lambda sonuc: (
            float(sonuc["roc_auc"]),
            float(sonuc["recall"]),
            float(sonuc["f1"]),
            float(sonuc["accuracy"]),
        ),
        reverse=True,
    )
    return siralanmis[0]


def model_sonuclarini_sirala(
    model_sonuclari: Sequence[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """Model sonuclarini secim onceligine gore buyukten kucuge siralar."""
    if not model_sonuclari:
        return []

    zorunlu_alanlar = {"model_adi", "roc_auc", "recall", "f1", "accuracy"}
    _sonuc_alanlarini_dogrula(model_sonuclari, zorunlu_alanlar)

    return sorted(
        model_sonuclari,
        key=lambda sonuc: (
            float(sonuc["roc_auc"]),
            float(sonuc["recall"]),
            float(sonuc["f1"]),
            float(sonuc["accuracy"]),
        ),
        reverse=True,
    )


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


def _sonuc_alanlarini_dogrula(
    model_sonuclari: Sequence[dict[str, float | str]],
    zorunlu_alanlar: set[str],
) -> None:
    for sonuc in model_sonuclari:
        eksik_alanlar = zorunlu_alanlar - set(sonuc.keys())
        if eksik_alanlar:
            eksik = ", ".join(sorted(eksik_alanlar))
            raise ValueError(f"Model sonucu icinde zorunlu alanlar eksik: {eksik}.")
