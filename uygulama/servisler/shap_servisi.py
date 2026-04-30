"""Uygulama tahmin akisi icin SHAP tabanli faktor servisi."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import pandas as pd

from makine_ogrenmesi.kaynak.ozellik_yapilandirmasi import OZELLIK_KOLONLARI
from makine_ogrenmesi.kaynak.veri_yukleyici import veri_setini_yukle


PROJE_KOKU = Path(__file__).resolve().parents[2]
VARSAYILAN_VERI_YOLU = PROJE_KOKU / "makine_ogrenmesi" / "veri" / "ham" / "diabetes.csv"

MODEL_TO_GIRDI_ALANI = {
    "Pregnancies": "pregnancies",
    "Glucose": "glucose",
    "BloodPressure": "blood_pressure",
    "SkinThickness": "skin_thickness",
    "Insulin": "insulin",
    "BMI": "bmi",
    "DiabetesPedigreeFunction": "diabetes_pedigree_function",
    "Age": "age",
}


def top_faktorleri_uret(
    model: Any,
    x_ornek: pd.DataFrame,
    top_n: int = 3,
    veri_yolu: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Tek ornek icin SHAP top faktorlerini dondurur."""
    if top_n <= 0:
        raise ValueError("top_n degeri sifirdan buyuk olmalidir.")

    arka_plan = _arka_plan_verisini_yukle(veri_yolu)
    arka_plan = arka_plan[x_ornek.columns.tolist()]

    shap_top_faktorleri = _shap_top_faktorleri_al()
    faktorler = shap_top_faktorleri(
        model=model,
        x_arka_plan=arka_plan,
        x_ornek=x_ornek,
        top_n=top_n,
        max_arka_plan=min(100, len(arka_plan)),
        random_state=42,
    )
    return [_faktor_adi_uyarla(faktor) for faktor in faktorler]


def _shap_top_faktorleri_al():
    """SHAP bagimliligini sadece ihtiyac aninda yukler."""
    from makine_ogrenmesi.kaynak.aciklanabilirlik import (
        top_faktorleri_hazirla as shap_top_faktorleri,
    )

    return shap_top_faktorleri


def _faktor_adi_uyarla(faktor: dict[str, Any]) -> dict[str, Any]:
    ham_ad = str(faktor.get("ozellik", ""))
    yeni_ad = MODEL_TO_GIRDI_ALANI.get(ham_ad, ham_ad)
    return {
        "ozellik": yeni_ad,
        "ozellik_degeri": faktor.get("ozellik_degeri"),
        "shap_katkisi": float(faktor.get("shap_katkisi", 0.0)),
        "yon": str(faktor.get("yon", "arttirici")),
    }


def _arka_plan_yolunu_belirle(veri_yolu: str | Path | None = None) -> Path:
    if veri_yolu is not None:
        return Path(veri_yolu)

    env_yolu = os.getenv("MODEL_VERI_YOLU")
    if env_yolu:
        yol = Path(env_yolu)
        return yol if yol.is_absolute() else (PROJE_KOKU / yol)

    return VARSAYILAN_VERI_YOLU


@lru_cache(maxsize=2)
def _arka_plan_verisini_yukle(veri_yolu: str | Path | None = None) -> pd.DataFrame:
    yol = _arka_plan_yolunu_belirle(veri_yolu)
    veri = veri_setini_yukle(yol)
    return veri[OZELLIK_KOLONLARI].copy()
