"""Model artifactlerini kaydetme ve yukleme yardimcilari."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn


ARTIFACT_DOSYA_ADLARI = {
    "en_iyi_pipeline": "en_iyi_pipeline.joblib",
    "kalibrator": "kalibrator.joblib",
    "esik_yapilandirmasi": "esik_yapilandirmasi.json",
    "ozellik_sirasi": "ozellik_sirasi.json",
    "metrik_ozeti": "metrik_ozeti.json",
    "model_metadata": "model_metadata.json",
}


def artifact_klasoru_hazirla(artifact_klasoru: str | Path) -> Path:
    """Artifact klasorunu olusturur ve Path olarak dondurur."""
    hedef = Path(artifact_klasoru)
    hedef.mkdir(parents=True, exist_ok=True)
    return hedef


def artifactleri_kaydet(
    artifact_klasoru: str | Path,
    en_iyi_pipeline: Any,
    kalibrator: Any,
    esik_yapilandirmasi: dict[str, Any],
    ozellik_sirasi: list[str],
    metrik_ozeti: dict[str, Any],
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Tum zorunlu artifact dosyalarini kaydeder."""
    klasor = artifact_klasoru_hazirla(artifact_klasoru)
    _ozellik_sirasini_dogrula(ozellik_sirasi)

    dosya_yollari = {
        anahtar: klasor / dosya_adi
        for anahtar, dosya_adi in ARTIFACT_DOSYA_ADLARI.items()
    }

    joblib.dump(en_iyi_pipeline, dosya_yollari["en_iyi_pipeline"])
    joblib.dump(kalibrator, dosya_yollari["kalibrator"])

    _json_yaz(dosya_yollari["esik_yapilandirmasi"], esik_yapilandirmasi)
    _json_yaz(dosya_yollari["ozellik_sirasi"], ozellik_sirasi)
    _json_yaz(dosya_yollari["metrik_ozeti"], metrik_ozeti)
    _json_yaz(
        dosya_yollari["model_metadata"],
        _model_metadata_hazirla(model_metadata),
    )

    return dosya_yollari


def artifactleri_yukle(artifact_klasoru: str | Path) -> dict[str, Any]:
    """Kayitli artifact dosyalarini yukler."""
    klasor = Path(artifact_klasoru)
    dosya_yollari = {
        anahtar: klasor / dosya_adi
        for anahtar, dosya_adi in ARTIFACT_DOSYA_ADLARI.items()
    }
    _artifact_dosyalari_var_mi(dosya_yollari)

    return {
        "en_iyi_pipeline": joblib.load(dosya_yollari["en_iyi_pipeline"]),
        "kalibrator": joblib.load(dosya_yollari["kalibrator"]),
        "esik_yapilandirmasi": _json_oku(dosya_yollari["esik_yapilandirmasi"]),
        "ozellik_sirasi": _json_oku(dosya_yollari["ozellik_sirasi"]),
        "metrik_ozeti": _json_oku(dosya_yollari["metrik_ozeti"]),
        "model_metadata": _json_oku(dosya_yollari["model_metadata"]),
    }


def _artifact_dosyalari_var_mi(dosya_yollari: dict[str, Path]) -> None:
    eksik = [str(yol) for yol in dosya_yollari.values() if not yol.exists()]
    if eksik:
        raise FileNotFoundError(
            "Artifact dosyalari eksik: " + ", ".join(eksik)
        )


def _model_metadata_hazirla(model_metadata: dict[str, Any] | None) -> dict[str, Any]:
    temel_bilgiler = {
        "proje_adi": "diyabet_risk_tahmini",
        "veri_seti": "Pima Indians Diabetes",
        "problem_tipi": "binary_classification",
        "olusturulma_zamani_utc": datetime.now(timezone.utc).isoformat(),
        "python_surumu": platform.python_version(),
        "sklearn_surumu": sklearn.__version__,
    }

    if model_metadata:
        temel_bilgiler.update(model_metadata)

    return temel_bilgiler


def _ozellik_sirasini_dogrula(ozellik_sirasi: list[str]) -> None:
    if not ozellik_sirasi:
        raise ValueError("ozellik_sirasi bos olamaz.")
    if not all(isinstance(kolon, str) and kolon for kolon in ozellik_sirasi):
        raise ValueError("ozellik_sirasi yalnizca bos olmayan metinlerden olusmalidir.")


def _json_yaz(dosya_yolu: Path, veri: Any) -> None:
    dosya_yolu.write_text(
        json.dumps(_json_uyumlu_yap(veri), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _json_oku(dosya_yolu: Path) -> Any:
    return json.loads(dosya_yolu.read_text(encoding="utf-8"))


def _json_uyumlu_yap(veri: Any) -> Any:
    if isinstance(veri, dict):
        return {str(k): _json_uyumlu_yap(v) for k, v in veri.items()}
    if isinstance(veri, (list, tuple, set)):
        return [_json_uyumlu_yap(v) for v in veri]
    if isinstance(veri, Path):
        return str(veri)
    if isinstance(veri, np.ndarray):
        return veri.tolist()
    if isinstance(veri, np.generic):
        return veri.item()
    return veri
