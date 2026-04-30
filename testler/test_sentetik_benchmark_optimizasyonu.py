"""Sentetik benchmark optimizasyon akisi testleri."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from makine_ogrenmesi.kaynak.sentetik_benchmark_optimizasyonu import (
    ANA_METRIKLER,
    sentetik_benchmark_calistir,
)


def test_sentetik_benchmark_mini_rapor_ve_csv_uretmeli(tmp_path: Path) -> None:
    proje_koku = Path(__file__).resolve().parents[1]
    veri_yolu = proje_koku / "makine_ogrenmesi" / "veri" / "ham" / "diabetes.csv"

    rapor = sentetik_benchmark_calistir(
        proje_koku=tmp_path,
        veri_yolu=veri_yolu,
        mod="quick",
        hedef_sinif_sayilari=(620,),
        model_adlari=("extra_trees",),
        n_jobs=1,
        artifact_yaz=False,
        word_raporu_yaz=False,
    )

    final_yol = Path(rapor["final_veri"]["yol"])
    assert final_yol.exists()
    final_df = pd.read_csv(final_yol)
    assert "Outcome" in final_df.columns
    assert len(final_df) >= 1000
    assert rapor["en_iyi_sonuc"]["durum"] == "tamamlandi"
    assert set(ANA_METRIKLER) <= set(rapor["en_iyi_sonuc"]["test_metrikleri"])
    assert (
        tmp_path
        / "makine_ogrenmesi"
        / "raporlar"
        / "degerlendirme"
        / "feature_ablation_raporu.json"
    ).exists()
