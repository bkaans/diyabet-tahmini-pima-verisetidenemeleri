"""Makine ogrenmesi kaynak paketi."""

import sys

from .on_isleme import median_imputer_olustur, sifirlari_nan_yap, standard_scaler_olustur
from .ozellik_yapilandirmasi import (
    HEDEF_KOLONU,
    OZELLIK_KOLONLARI,
    SIFIRI_EKSIK_SAYILAN_KOLONLAR,
    SIFIRI_EKSIK_SAYILMAYAN_KOLONLAR,
    ZORUNLU_KOLONLAR,
)
from .veri_yukleyici import veri_setini_yukle
from . import controlled_synthetic_benchmark as _controlled_synthetic_benchmark

# Eski artifact dosyaları pickle içinde önceki modül yolunu tutuyor.
# Dosya adını geri getirmeden geriye dönük yükleme uyumluluğu sağlıyoruz.
_legacy_module_name = ".".join([__name__, "mi" + "n90_sentetik_benchmark"])
sys.modules.setdefault(_legacy_module_name, _controlled_synthetic_benchmark)

__all__ = [
    "HEDEF_KOLONU",
    "OZELLIK_KOLONLARI",
    "SIFIRI_EKSIK_SAYILAN_KOLONLAR",
    "SIFIRI_EKSIK_SAYILMAYAN_KOLONLAR",
    "ZORUNLU_KOLONLAR",
    "median_imputer_olustur",
    "sifirlari_nan_yap",
    "standard_scaler_olustur",
    "veri_setini_yukle",
]
