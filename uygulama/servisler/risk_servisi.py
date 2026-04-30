"""Risk seviyesi ve ikili sinif karari icin servis fonksiyonlari."""

from __future__ import annotations

from typing import Any

from makine_ogrenmesi.kaynak.esik_analizi import risk_kategorisi_belirle

VARSAYILAN_RISK_ESIKLERI = {
    "dusuk_ust_esik": 0.33,
    "orta_ust_esik": 0.66,
}

RISK_KATEGORISI_ESLEMESI = {
    "dusuk": "dusuk",
    "low": "dusuk",
    "cok_dusuk": "dusuk",
    "cokdusuk": "dusuk",
    "very_low": "dusuk",
    "orta": "orta",
    "medium": "orta",
    "mid": "orta",
    "orta_risk": "orta",
    "yuksek": "yuksek",
    "high": "yuksek",
    "cok_yuksek": "yuksek",
    "cokyuksek": "yuksek",
    "very_high": "yuksek",
}

_TR_HARF_CEVIRIMI = str.maketrans("çğıöşü", "cgiosu")


def ikili_sinif_hesapla(olasilik: float, esik_yapilandirmasi: dict[str, Any]) -> int:
    """Kalibre edilmis olasiliga gore ikili sinif tahmini uretir."""
    esik = onerilen_ikili_siniflama_esigi_al(esik_yapilandirmasi)
    return int(float(olasilik) >= esik)


def risk_kategorisi_hesapla(olasilik: float, esik_yapilandirmasi: dict[str, Any]) -> str:
    """Kalibre edilmis olasiliga gore risk kategorisini dondurur."""
    risk_esikleri = risk_esiklerini_al(esik_yapilandirmasi)
    risk_kategorisi = risk_kategorisi_belirle(
        olasilik=float(olasilik),
        dusuk_ust_esik=float(risk_esikleri["dusuk_ust_esik"]),
        orta_ust_esik=float(risk_esikleri["orta_ust_esik"]),
    )
    return risk_kategorisini_normalize_et(risk_kategorisi)


def risk_ozeti_hazirla(olasilik: float, esik_yapilandirmasi: dict[str, Any]) -> dict[str, Any]:
    """Risk siniflamasini tek sozlukte toplar."""
    olasilik_float = float(olasilik)
    return {
        "olasilik": olasilik_float,
        "sinif": ikili_sinif_hesapla(olasilik_float, esik_yapilandirmasi),
        "risk_kategorisi": risk_kategorisi_hesapla(olasilik_float, esik_yapilandirmasi),
        "onerilen_ikili_siniflama_esigi": onerilen_ikili_siniflama_esigi_al(
            esik_yapilandirmasi
        ),
    }


def risk_kategorisini_normalize_et(risk_kategorisi: str) -> str:
    """Legacy veya farkli formatlardan gelen risk etiketini 3'lu standarda cevirir."""
    sade = str(risk_kategorisi).strip().lower()
    sade = sade.translate(_TR_HARF_CEVIRIMI)
    sade = sade.replace("-", "_").replace(" ", "_")
    while "__" in sade:
        sade = sade.replace("__", "_")

    if sade in RISK_KATEGORISI_ESLEMESI:
        return RISK_KATEGORISI_ESLEMESI[sade]

    if "dusuk" in sade or "low" in sade:
        return "dusuk"
    if "yuksek" in sade or "high" in sade:
        return "yuksek"
    if "orta" in sade or "medium" in sade or "mid" in sade:
        return "orta"

    raise ValueError(
        "risk_kategorisi desteklenmeyen degerde geldi: "
        f"{risk_kategorisi!r}. Beklenen degerler: dusuk, orta, yuksek."
    )


def onerilen_ikili_siniflama_esigi_al(esik_yapilandirmasi: dict[str, Any]) -> float:
    """Esik konfigurasyonundan onerilen ikili siniflama esigini alir."""
    try:
        esik = float(esik_yapilandirmasi["onerilen_ikili_siniflama_esigi"])
    except KeyError as hata:
        raise KeyError("esik_yapilandirmasi icinde 'onerilen_ikili_siniflama_esigi' yok.") from hata

    _birim_aralik_kontrolu(esik, "onerilen_ikili_siniflama_esigi")
    return esik


def risk_esiklerini_al(esik_yapilandirmasi: dict[str, Any]) -> dict[str, float]:
    """Risk seviyesi esiklerini konfigurasyondan alir."""
    risk_kategorileri = esik_yapilandirmasi.get("risk_kategorileri", {})
    dusuk = float(risk_kategorileri.get("dusuk_ust_esik", VARSAYILAN_RISK_ESIKLERI["dusuk_ust_esik"]))
    orta = float(risk_kategorileri.get("orta_ust_esik", VARSAYILAN_RISK_ESIKLERI["orta_ust_esik"]))

    _birim_aralik_kontrolu(dusuk, "dusuk_ust_esik")
    _birim_aralik_kontrolu(orta, "orta_ust_esik")
    if dusuk > orta:
        raise ValueError("risk esiklerinde dusuk_ust_esik, orta_ust_esik degerinden buyuk olamaz.")

    return {
        "dusuk_ust_esik": dusuk,
        "orta_ust_esik": orta,
    }


def _birim_aralik_kontrolu(deger: float, alan_adi: str) -> None:
    if deger < 0 or deger > 1:
        raise ValueError(f"{alan_adi} degeri 0 ile 1 araliginda olmalidir.")
