"""Pima Indians Diabetes veri setini yukleme yardimcilari."""

from pathlib import Path

import pandas as pd

from .ozellik_yapilandirmasi import ZORUNLU_KOLONLAR


def veri_setini_yukle(veri_yolu: str | Path) -> pd.DataFrame:
    """CSV veri setini yukler ve zorunlu kolonlari dogrular."""
    yol = Path(veri_yolu)

    if not yol.exists():
        raise FileNotFoundError(f"Veri dosyasi bulunamadi: {yol}")

    if not yol.is_file():
        raise FileNotFoundError(f"Veri yolu bir dosya degil: {yol}")

    veri_cercevesi = pd.read_csv(yol)
    _zorunlu_kolonlari_dogrula(veri_cercevesi)
    return veri_cercevesi


def _zorunlu_kolonlari_dogrula(veri_cercevesi: pd.DataFrame) -> None:
    eksik_kolonlar = sorted(set(ZORUNLU_KOLONLAR) - set(veri_cercevesi.columns))
    if eksik_kolonlar:
        eksik = ", ".join(eksik_kolonlar)
        raise ValueError(
            "Veri setinde zorunlu kolonlar eksik. "
            f"Eksik kolonlar: {eksik}."
        )
