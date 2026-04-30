"""SHAP tabanli aciklanabilirlik katmani."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator


def shap_hesaplayici_olustur(
    model: BaseEstimator,
    x_arka_plan: pd.DataFrame | np.ndarray,
    max_arka_plan: int = 100,
    random_state: int = 42,
) -> shap.Explainer:
    """Model icin SHAP hesaplayici nesnesi olusturur."""
    arka_plan_df = _ornekle_dataframe(
        _veriyi_dataframe_yap(x_arka_plan),
        max_ornek=max_arka_plan,
        random_state=random_state,
    )
    tahmin_fonksiyonu = _pozitif_sinif_tahmin_fonksiyonu_olustur(
        model=model,
        kolonlar=list(arka_plan_df.columns),
    )
    return shap.Explainer(tahmin_fonksiyonu, arka_plan_df)


def global_shap_ozeti_hesapla(
    model: BaseEstimator,
    x_veri: pd.DataFrame | np.ndarray,
    max_arka_plan: int = 100,
    max_degerlendirme: int = 200,
    random_state: int = 42,
) -> dict[str, Any]:
    """Global SHAP ozetini ozellik bazinda dondurur."""
    veri_df = _veriyi_dataframe_yap(x_veri)
    degerlendirme_df = _ornekle_dataframe(
        veri_df,
        max_ornek=max_degerlendirme,
        random_state=random_state,
    )
    explainer = shap_hesaplayici_olustur(
        model=model,
        x_arka_plan=veri_df,
        max_arka_plan=max_arka_plan,
        random_state=random_state,
    )
    shap_sonuclari = explainer(degerlendirme_df)
    shap_matris = _shap_matrisini_al(shap_sonuclari)

    ortalama_mutlak_shap = np.mean(np.abs(shap_matris), axis=0)
    sirali_indeksler = np.argsort(ortalama_mutlak_shap)[::-1]

    global_siralama = [
        {
            "ozellik": degerlendirme_df.columns[idx],
            "ortalama_mutlak_shap": float(ortalama_mutlak_shap[idx]),
        }
        for idx in sirali_indeksler
    ]

    return {
        "ornek_sayisi": int(len(degerlendirme_df)),
        "ozellik_sayisi": int(degerlendirme_df.shape[1]),
        "global_siralama": global_siralama,
    }


def lokal_shap_yorumlari_hesapla(
    model: BaseEstimator,
    x_arka_plan: pd.DataFrame | np.ndarray,
    x_ornekler: pd.DataFrame | np.ndarray,
    top_n: int = 3,
    max_arka_plan: int = 100,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Lokal SHAP yorumlarini her ornek icin dondurur."""
    if top_n <= 0:
        raise ValueError("top_n degeri sifirdan buyuk olmalidir.")

    arka_plan_df = _veriyi_dataframe_yap(x_arka_plan)
    ornekler_df = _veriyi_dataframe_yap(x_ornekler, kolonlar=arka_plan_df.columns.tolist())

    explainer = shap_hesaplayici_olustur(
        model=model,
        x_arka_plan=arka_plan_df,
        max_arka_plan=max_arka_plan,
        random_state=random_state,
    )
    shap_sonuclari = explainer(ornekler_df)
    shap_matris = _shap_matrisini_al(shap_sonuclari)
    base_degerler = _base_degerleri_al(shap_sonuclari, len(ornekler_df))

    tahmin_olasiliklari = _pozitif_sinif_tahmin_fonksiyonu_olustur(
        model=model,
        kolonlar=list(ornekler_df.columns),
    )(ornekler_df)

    yorumlar: list[dict[str, Any]] = []
    for satir_indeksi in range(len(ornekler_df)):
        katkilar = shap_matris[satir_indeksi]
        satir = ornekler_df.iloc[satir_indeksi]

        top_indeksler = np.argsort(np.abs(katkilar))[::-1][:top_n]
        top_faktorler = [
            {
                "ozellik": str(ornekler_df.columns[idx]),
                "ozellik_degeri": _json_uyumlu_deger(satir.iloc[idx]),
                "shap_katkisi": float(katkilar[idx]),
                "yon": "arttirici" if katkilar[idx] >= 0 else "azaltici",
            }
            for idx in top_indeksler
        ]

        yorumlar.append(
            {
                "ornek_indeksi": int(satir_indeksi),
                "beklenen_olasilik": float(base_degerler[satir_indeksi]),
                "tahmin_olasiligi": float(tahmin_olasiliklari[satir_indeksi]),
                "top_faktorler": top_faktorler,
            }
        )

    return yorumlar


def top_faktorleri_hazirla(
    model: BaseEstimator,
    x_arka_plan: pd.DataFrame | np.ndarray,
    x_ornek: pd.DataFrame | np.ndarray,
    top_n: int = 3,
    max_arka_plan: int = 100,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Uygulama sonucu icin tek ornekte top faktorleri dondurur."""
    yorumlar = lokal_shap_yorumlari_hesapla(
        model=model,
        x_arka_plan=x_arka_plan,
        x_ornekler=x_ornek,
        top_n=top_n,
        max_arka_plan=max_arka_plan,
        random_state=random_state,
    )
    if not yorumlar:
        return []
    return yorumlar[0]["top_faktorler"]


def global_shap_gorseli_kaydet(
    model: BaseEstimator,
    x_veri: pd.DataFrame | np.ndarray,
    cikti_yolu: str | Path,
    max_arka_plan: int = 100,
    max_degerlendirme: int = 200,
    random_state: int = 42,
) -> Path:
    """Global SHAP bar gorselini Turkce eksen/basliklarla PNG olarak kaydeder."""
    global_ozet = global_shap_ozeti_hesapla(
        model=model,
        x_veri=x_veri,
        max_arka_plan=max_arka_plan,
        max_degerlendirme=max_degerlendirme,
        random_state=random_state,
    )

    global_siralama = global_ozet["global_siralama"]
    ozellikler = [item["ozellik"] for item in reversed(global_siralama)]
    etkiler = [item["ortalama_mutlak_shap"] for item in reversed(global_siralama)]

    plt.figure(figsize=(10, 6))
    plt.barh(ozellikler, etkiler, color="#2c7fb8")
    plt.xlabel("Ortalama mutlak SHAP etkisi")
    plt.ylabel("Ozellikler")
    plt.title("Global SHAP ozet grafigi")
    plt.grid(axis="x", linestyle="--", alpha=0.3)

    hedef = Path(cikti_yolu)
    hedef.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(hedef, dpi=150, bbox_inches="tight")
    plt.close()
    return hedef


def _veriyi_dataframe_yap(
    veri: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    kolonlar: list[str] | None = None,
) -> pd.DataFrame:
    if isinstance(veri, pd.DataFrame):
        return veri.copy()

    dizi = np.asarray(veri)
    if dizi.ndim == 1:
        dizi = dizi.reshape(1, -1)
    if dizi.ndim != 2:
        raise ValueError("Veri 2 boyutlu olmalidir.")

    if kolonlar is None:
        kolonlar = [f"ozellik_{i}" for i in range(dizi.shape[1])]
    if len(kolonlar) != dizi.shape[1]:
        raise ValueError("Kolon sayisi veri boyutuyla uyusmuyor.")

    return pd.DataFrame(dizi, columns=kolonlar)


def _ornekle_dataframe(veri_df: pd.DataFrame, max_ornek: int, random_state: int) -> pd.DataFrame:
    if max_ornek <= 0:
        raise ValueError("max_ornek degeri sifirdan buyuk olmalidir.")
    if len(veri_df) <= max_ornek:
        return veri_df.copy()
    return veri_df.sample(n=max_ornek, random_state=random_state).reset_index(drop=True)


def _pozitif_sinif_tahmin_fonksiyonu_olustur(
    model: BaseEstimator,
    kolonlar: list[str],
):
    if not hasattr(model, "predict_proba"):
        raise TypeError("SHAP icin modelde predict_proba metodu olmalidir.")

    def _tahmin(veri: pd.DataFrame | np.ndarray) -> np.ndarray:
        veri_df = _veriyi_dataframe_yap(veri, kolonlar=kolonlar)
        olasiliklar = np.asarray(model.predict_proba(veri_df))
        if olasiliklar.ndim != 2 or olasiliklar.shape[1] < 2:
            raise ValueError("predict_proba cikisi beklenen formatta degil.")
        return olasiliklar[:, 1]

    return _tahmin


def _shap_matrisini_al(shap_sonuclari) -> np.ndarray:
    shap_matris = np.asarray(shap_sonuclari.values)
    if shap_matris.ndim == 3 and shap_matris.shape[2] >= 2:
        return shap_matris[:, :, 1]
    if shap_matris.ndim != 2:
        raise ValueError("SHAP sonuc matris boyutu beklenen formatta degil.")
    return shap_matris


def _base_degerleri_al(shap_sonuclari, ornek_sayisi: int) -> np.ndarray:
    base = np.asarray(shap_sonuclari.base_values)
    if base.ndim == 0:
        return np.full(ornek_sayisi, float(base))
    if base.ndim == 1:
        return base.astype(float)
    if base.ndim == 2 and base.shape[1] >= 2:
        return base[:, 1].astype(float)
    raise ValueError("SHAP base_values formati beklenen yapida degil.")


def _json_uyumlu_deger(deger: Any) -> Any:
    if hasattr(deger, "item"):
        return deger.item()
    return deger
