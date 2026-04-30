"""On isleme adimlari icin yeniden kullanilabilir fonksiyonlar."""

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .ozellik_yapilandirmasi import SIFIRI_EKSIK_SAYILAN_KOLONLAR


def sifirlari_nan_yap(
    veri_cercevesi: pd.DataFrame,
    kolonlar: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Sadece verilen kolonlarda 0 degerlerini NaN olarak isaretler."""
    hedef_kolonlar = list(kolonlar) if kolonlar is not None else SIFIRI_EKSIK_SAYILAN_KOLONLAR
    _kolonlari_dogrula(veri_cercevesi, hedef_kolonlar)

    donusmus_veri = veri_cercevesi.copy()
    for kolon in hedef_kolonlar:
        donusmus_veri[kolon] = donusmus_veri[kolon].where(
            donusmus_veri[kolon] != 0, np.nan
        )
    return donusmus_veri


def sifirlari_nan_donustur_pipeline(veri: pd.DataFrame) -> pd.DataFrame:
    """Pipeline icin 0 -> NaN donusumunu sabit kolon setiyle uygular."""
    if not isinstance(veri, pd.DataFrame):
        raise TypeError("Pipeline girisi pandas DataFrame olmalidir.")
    return sifirlari_nan_yap(veri, kolonlar=SIFIRI_EKSIK_SAYILAN_KOLONLAR)


def median_imputer_olustur() -> SimpleImputer:
    """Median stratejili imputasyon nesnesi dondurur."""
    return SimpleImputer(strategy="median")


def standard_scaler_olustur() -> StandardScaler:
    """StandardScaler nesnesi dondurur."""
    return StandardScaler()


def _kolonlari_dogrula(veri_cercevesi: pd.DataFrame, kolonlar: list[str]) -> None:
    eksik_kolonlar = sorted(set(kolonlar) - set(veri_cercevesi.columns))
    if eksik_kolonlar:
        eksik = ", ".join(eksik_kolonlar)
        raise ValueError(f"0 -> NaN donusumu icin kolonlar bulunamadi: {eksik}.")
