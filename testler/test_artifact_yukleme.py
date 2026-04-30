"""Artifact yukleme testleri."""

from __future__ import annotations

from pathlib import Path

import pytest

from makine_ogrenmesi.kaynak.artifact_kaydet import artifactleri_yukle


def test_artifactleri_yukle_beklenen_anahtarlari_dondurmeli() -> None:
    proje_koku = Path(__file__).resolve().parents[1]
    artifact_klasoru = proje_koku / "makine_ogrenmesi" / "artifactler"

    sonuc = artifactleri_yukle(artifact_klasoru)

    assert sorted(sonuc.keys()) == [
        "en_iyi_pipeline",
        "esik_yapilandirmasi",
        "kalibrator",
        "metrik_ozeti",
        "model_metadata",
        "ozellik_sirasi",
    ]


def test_artifactleri_yukle_eksik_klasorde_hata_vermeli(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Artifact dosyalari eksik"):
        artifactleri_yukle(tmp_path)
