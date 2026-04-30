"""Artifact yukleyip tek ornek tahmin ureten servis."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from makine_ogrenmesi.kaynak.artifact_kaydet import artifactleri_yukle
from uygulama.semalar.girdi_semalari import TahminGirdisi

from .aciklama_servisi import kisa_aciklama_uret
from .risk_servisi import risk_ozeti_hazirla
from .shap_servisi import top_faktorleri_uret


PROJE_KOKU = Path(__file__).resolve().parents[2]
VARSAYILAN_ARTIFACT_KLASORU = PROJE_KOKU / "makine_ogrenmesi" / "artifactler"

GIRDI_TO_MODEL_KOLON = {
    "pregnancies": "Pregnancies",
    "glucose": "Glucose",
    "blood_pressure": "BloodPressure",
    "skin_thickness": "SkinThickness",
    "insulin": "Insulin",
    "bmi": "BMI",
    "diabetes_pedigree_function": "DiabetesPedigreeFunction",
    "age": "Age",
}


@lru_cache(maxsize=2)
def artifactleri_yukle_servisi(artifact_klasoru: str | Path | None = None) -> dict[str, Any]:
    """Tahmin icin gerekli artifactleri yukler ve onbellege alir."""
    if artifact_klasoru:
        hedef = Path(artifact_klasoru)
    else:
        env_degeri = os.getenv("MODEL_ARTIFACT_KLASORU")
        if env_degeri:
            env_yolu = Path(env_degeri)
            hedef = env_yolu if env_yolu.is_absolute() else (PROJE_KOKU / env_yolu)
        else:
            hedef = VARSAYILAN_ARTIFACT_KLASORU
    return artifactleri_yukle(hedef)


def artifact_onbellegini_temizle() -> None:
    """Artifact cache'ini temizler."""
    artifactleri_yukle_servisi.cache_clear()


def tek_ornek_tahmin_uret(
    girdi: TahminGirdisi | dict[str, float | int],
    artifact_klasoru: str | Path | None = None,
    include_tum_faktorler: bool = False,
) -> dict[str, Any]:
    """Tek bir girdi icin olasilik, sinif, risk ve aciklama uretir."""
    artifactler = artifactleri_yukle_servisi(artifact_klasoru)
    girdi_dict = _girdiyi_dict_yap(girdi)
    x_ornek = _ornek_dataframe_hazirla(girdi_dict, artifactler["ozellik_sirasi"])

    pipeline = artifactler["en_iyi_pipeline"]
    kalibrator = artifactler["kalibrator"]
    esik_yapilandirmasi = artifactler["esik_yapilandirmasi"]

    _ = _pozitif_sinif_olasiligi_hesapla(pipeline, x_ornek)
    kalibre_olasilik = _pozitif_sinif_olasiligi_hesapla(kalibrator, x_ornek)

    risk_ozeti = risk_ozeti_hazirla(kalibre_olasilik, esik_yapilandirmasi)
    top_faktorler = _guvenli_top_faktorleri_uret(
        model=pipeline,
        x_ornek=x_ornek,
        top_n=3,
    )
    tum_faktorler: list[dict[str, Any]] = []
    if include_tum_faktorler:
        tum_faktorler = _guvenli_top_faktorleri_uret(
            model=pipeline,
            x_ornek=x_ornek,
            top_n=max(3, x_ornek.shape[1]),
        )
    kisa_aciklama = kisa_aciklama_uret(
        risk_kategorisi=risk_ozeti["risk_kategorisi"],
        olasilik=kalibre_olasilik,
        top_faktorler=top_faktorler,
    )

    sonuc = {
        "olasilik": float(kalibre_olasilik),
        "sinif": int(risk_ozeti["sinif"]),
        "risk_kategorisi": str(risk_ozeti["risk_kategorisi"]),
        "top_faktorler": top_faktorler,
        "kisa_aciklama": kisa_aciklama,
    }
    if include_tum_faktorler:
        sonuc["tum_faktorler"] = tum_faktorler
    return sonuc


def _girdiyi_dict_yap(girdi: TahminGirdisi | dict[str, float | int]) -> dict[str, float | int]:
    if isinstance(girdi, TahminGirdisi):
        return girdi.model_dump()
    if isinstance(girdi, dict):
        return girdi
    raise TypeError("girdi tipi TahminGirdisi veya dict olmalidir.")


def _ornek_dataframe_hazirla(
    girdi_dict: dict[str, float | int],
    ozellik_sirasi: list[str],
) -> pd.DataFrame:
    eksik_alanlar = [alan for alan in GIRDI_TO_MODEL_KOLON if alan not in girdi_dict]
    if eksik_alanlar:
        raise ValueError(
            "Tahmin girdisinde eksik alanlar var: " + ", ".join(sorted(eksik_alanlar))
        )

    model_satiri = {
        model_kolon: girdi_dict[girdi_alan]
        for girdi_alan, model_kolon in GIRDI_TO_MODEL_KOLON.items()
    }

    eksik_model_kolonlar = [kolon for kolon in ozellik_sirasi if kolon not in model_satiri]
    if eksik_model_kolonlar:
        raise ValueError(
            "Artifact ozellik sirasinda beklenmeyen kolonlar var: "
            + ", ".join(sorted(eksik_model_kolonlar))
        )

    return pd.DataFrame([model_satiri], columns=ozellik_sirasi)


def _pozitif_sinif_olasiligi_hesapla(model: Any, x_ornek: pd.DataFrame) -> float:
    if not hasattr(model, "predict_proba"):
        raise TypeError("Modelde predict_proba metodu bulunmuyor.")

    olasiliklar = np.asarray(model.predict_proba(x_ornek))
    if olasiliklar.ndim != 2 or olasiliklar.shape[1] < 2:
        raise ValueError("predict_proba cikisi beklenen formatta degil.")
    return float(olasiliklar[0, 1])


def _guvenli_top_faktorleri_uret(
    model: Any,
    x_ornek: pd.DataFrame,
    top_n: int,
) -> list[dict[str, Any]]:
    """SHAP hesabi hata verirse API akisini bozmadan bos liste dondurur."""
    try:
        return top_faktorleri_uret(
            model=model,
            x_ornek=x_ornek,
            top_n=top_n,
        )
    except Exception:
        return []
