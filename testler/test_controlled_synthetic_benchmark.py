"""Min-0.90 source_id kontrollu sentetik benchmark testleri."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from makine_ogrenmesi.kaynak.controlled_synthetic_benchmark import controlled_benchmark_calistir


def test_controlled_benchmark_benchmark_source_id_guvenli_csv_ve_json_uretmeli(tmp_path: Path) -> None:
    proje_koku = Path(__file__).resolve().parents[1]
    veri_yolu = proje_koku / "makine_ogrenmesi" / "veri" / "ham" / "diabetes.csv"

    sonuc = controlled_benchmark_calistir(
        proje_koku=tmp_path,
        veri_yolu=veri_yolu,
        target_per_class_values=(430,),
        n_jobs=1,
        word_raporu_yaz=False,
        quick=True,
    )

    csv_yolu = (
        tmp_path
        / "makine_ogrenmesi"
        / "veri"
        / "deneysel"
        / "controlled_synthetic_430_per_class_pima.csv"
    )
    meta_yolu = csv_yolu.with_name("controlled_synthetic_430_per_class_pima_metadata.csv")
    assert csv_yolu.exists()
    assert meta_yolu.exists()

    veri = pd.read_csv(csv_yolu)
    metadata = pd.read_csv(meta_yolu)
    assert len(veri) == len(metadata)
    assert {"source_id", "is_synthetic", "parent_original_index", "generation_method", "synthetic_strength", "noise_profile"} <= set(metadata.columns)
    synthetic = metadata[metadata["is_synthetic"].astype(bool)]
    assert not synthetic.empty
    assert synthetic["source_id"].astype(str).str.match(r"^original_\d+$").all()

    leakage = sonuc["leakage_report"]["datasets"][0]
    assert leakage["leakage_status"] == "clean"
    assert leakage["train_test_source_intersection"] == 0
    assert max(leakage["cv_source_intersections"]) == 0
    assert leakage["exact_duplicate_count"] == 0

    assert (
        tmp_path
        / "makine_ogrenmesi"
        / "raporlar"
        / "controlled_benchmark_optimization_results.json"
    ).exists()
