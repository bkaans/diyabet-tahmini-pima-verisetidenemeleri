"""Small-scale sentetik benchmark hattı için smoke testler."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from makine_ogrenmesi.kaynak.benchmark_scale_sweep import benchmark_scale_sweep_calistir


def test_benchmark_scale_sweep_quick_csv_json_ve_leakage_uretmelidir(tmp_path: Path) -> None:
    proje_koku = Path(__file__).resolve().parents[1]
    veri_yolu = proje_koku / "makine_ogrenmesi" / "veri" / "ham" / "diabetes.csv"

    sonuc = benchmark_scale_sweep_calistir(
        proje_koku=tmp_path,
        veri_yolu=veri_yolu,
        n_jobs=1,
        word_raporu_yaz=False,
        quick=True,
    )

    rapor_dir = tmp_path / "makine_ogrenmesi" / "raporlar"
    veri_dir = tmp_path / "makine_ogrenmesi" / "veri" / "deneysel" / "benchmark_scale_sweep"
    assert (rapor_dir / "benchmark_scale_sweep_results.json").exists()
    assert (rapor_dir / "benchmark_scale_sweep_selection_report.json").exists()
    assert (rapor_dir / "benchmark_scale_sweep_quality_audit.json").exists()

    csv_yollari = sorted(veri_dir.glob("scale_sweep_*_per_class_strength_*.csv"))
    metadata_yollari = sorted(veri_dir.glob("scale_sweep_*_per_class_strength_*_metadata.csv"))
    assert csv_yollari
    assert metadata_yollari

    metadata = pd.read_csv(metadata_yollari[0])
    assert {
        "source_id",
        "is_synthetic",
        "parent_original_index",
        "generation_method",
        "synthetic_strength",
        "noise_profile",
    } <= set(metadata.columns)
    synthetic = metadata[metadata["is_synthetic"].astype(bool)]
    assert not synthetic.empty
    assert synthetic["source_id"].astype(str).str.match(r"^original_\d+$").all()

    selection = sonuc["selection_report"]
    leak = selection["leakage_summary"]
    assert leak["leakage_status"] == "clean"
    assert leak["train_test_source_intersection"] == 0
    assert max(leak["cv_source_intersections"] or [0]) == 0
    assert leak["exact_duplicate_count"] == 0
    assert leak["independent_synthetic_source_id_count"] == 0
