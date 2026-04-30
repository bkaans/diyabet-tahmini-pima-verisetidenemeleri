"""Tahmin sonucuna yonelik aciklama servisleri."""

from __future__ import annotations

from typing import Any

from .risk_servisi import risk_kategorisini_normalize_et


def kisa_aciklama_uret(
    risk_kategorisi: str,
    olasilik: float,
    top_faktorler: list[dict[str, Any]] | None = None,
) -> str:
    """Risk seviyesine gore kullaniciya kisa aciklama metni uretir."""
    try:
        risk_kategorisi = risk_kategorisini_normalize_et(risk_kategorisi)
    except ValueError:
        pass

    olasilik_yuzde = round(float(olasilik) * 100, 1)
    faktor_notu = _faktor_notunu_hazirla(top_faktorler or [])

    if risk_kategorisi == "yuksek":
        return (
            f"Tahmini risk düzeyi yüksek görünüyor (%{olasilik_yuzde}). "
            "Sonucu bir sağlık profesyoneli ile değerlendirmeniz önerilir."
            f"{faktor_notu}"
        )
    if risk_kategorisi == "orta":
        return (
            f"Tahmini risk düzeyi orta seviyede (%{olasilik_yuzde}). "
            "Yaşam tarzı ve klinik takip adımlarıyla desteklenmesi uygun olabilir."
            f"{faktor_notu}"
        )
    if risk_kategorisi == "dusuk":
        return (
            f"Tahmini risk düzeyi düşük görünüyor (%{olasilik_yuzde}). "
            "Sonuç klinik tanı yerine farkındalık amaçlı değerlendirilmelidir."
            f"{faktor_notu}"
        )

    return (
        f"Tahmini risk olasılığı %{olasilik_yuzde}. "
        "Risk seviyesi belirlenemedi, girdi ve eşik ayarlarını kontrol edin."
        f"{faktor_notu}"
    )


def _faktor_notunu_hazirla(top_faktorler: list[dict[str, Any]]) -> str:
    if not top_faktorler:
        return ""

    birinci = top_faktorler[0]
    ozellik = str(birinci.get("ozellik", "özellik")).replace("_", " ")
    yon = str(birinci.get("yon", "arttirici"))
    yon_metin = {"arttirici": "arttırıcı", "azaltici": "azaltıcı"}.get(yon, yon)
    return f" En belirgin faktör: {ozellik} ({yon_metin})."
