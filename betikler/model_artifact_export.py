"""1000/1000 source_id kontrollu benchmark icin final artifact uretir."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline


PROJE_KOKU = Path(__file__).resolve().parents[1]
if str(PROJE_KOKU) not in sys.path:
    sys.path.insert(0, str(PROJE_KOKU))

from makine_ogrenmesi.kaynak.artifact_kaydet import artifactleri_kaydet  # noqa: E402
from makine_ogrenmesi.kaynak.controlled_synthetic_benchmark import (  # noqa: E402
    ClinicalFeatureTransformer,
    PimaZeroMedianImputer,
)
from makine_ogrenmesi.kaynak.ozellik_yapilandirmasi import HEDEF_KOLONU, OZELLIK_KOLONLARI  # noqa: E402


RANDOM_STATE = 42
DATASET_ADI = "scale_sweep_1000_per_class_strength_0p60_adaptive"
DATASET_YOLU = PROJE_KOKU / "makine_ogrenmesi" / "veri" / "deneysel" / "benchmark_scale_sweep" / f"{DATASET_ADI}.csv"
METADATA_YOLU = PROJE_KOKU / "makine_ogrenmesi" / "veri" / "deneysel" / "benchmark_scale_sweep" / f"{DATASET_ADI}_metadata.csv"
SELECTION_RAPORU = PROJE_KOKU / "makine_ogrenmesi" / "raporlar" / "benchmark_scale_sweep_selection_report.json"
SWEEP_RAPORU = PROJE_KOKU / "makine_ogrenmesi" / "raporlar" / "benchmark_scale_sweep_results.json"
ARTIFACT_KLASORU = PROJE_KOKU / "makine_ogrenmesi" / "artifactler"
YEDEK_KLASORU = PROJE_KOKU / "tmp" / "artifact_yedekleri" / datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_oku(yol: Path) -> dict:
    return json.loads(yol.read_text(encoding="utf-8"))


def _1000_cv_ozeti(sweep: dict) -> dict:
    adaylar = [
        sonuc
        for sonuc in sweep.get("cv_results", [])
        if sonuc.get("dataset_name") == DATASET_ADI
        and sonuc.get("model") == "extra_trees"
        and sonuc.get("feature_set") == "high_signal_features"
    ]
    if not adaylar:
        return {}
    return sorted(
        adaylar,
        key=lambda x: x.get("summary", {}).get("ana_metrik_minimumu_mean", -1),
        reverse=True,
    )[0]


def _1000_holdout_ozeti(sweep: dict) -> dict:
    adaylar = []
    for anahtar in ("phase1_results", "phase2_results", "optuna_results"):
        adaylar.extend(
            sonuc
            for sonuc in sweep.get(anahtar, [])
            if sonuc.get("status") == "completed"
            and sonuc.get("dataset_name") == DATASET_ADI
            and sonuc.get("model") == "extra_trees"
            and sonuc.get("feature_set") == "high_signal_features"
        )
    if not adaylar:
        return {}
    return sorted(
        adaylar,
        key=lambda x: (
            x.get("tuned_threshold_metrics", {}).get("ana_metrik_minimumu", -1),
            x.get("tuned_threshold_metrics", {}).get("accuracy", -1),
        ),
        reverse=True,
    )[0]


def _artifactleri_yedekle() -> None:
    if not ARTIFACT_KLASORU.exists():
        return
    mevcutlar = [yol for yol in ARTIFACT_KLASORU.iterdir() if yol.is_file()]
    if not mevcutlar:
        return
    YEDEK_KLASORU.mkdir(parents=True, exist_ok=True)
    for yol in mevcutlar:
        shutil.copy2(yol, YEDEK_KLASORU / yol.name)


def _pipeline_olustur() -> Pipeline:
    # Final uygulama artifact'i ham 8 PIMA girdisini kabul eder.
    # Eksik sayilan 0 degerleri once medyanla doldurulur, sonra high-signal feature seti uretilir.
    return Pipeline(
        steps=[
            ("pima_zero_median_imputer", PimaZeroMedianImputer()),
            (
                "features",
                ClinicalFeatureTransformer(
                    base_columns=("Pregnancies", "Glucose", "BMI", "DiabetesPedigreeFunction", "Age", "Insulin"),
                    engineered=("glucose_bmi_interaction", "insulin_glucose_ratio", "glucose_pedigree_interaction"),
                ),
            ),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=220,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=2,
                ),
            ),
        ]
    )


def main() -> None:
    if not DATASET_YOLU.exists():
        raise FileNotFoundError(DATASET_YOLU)
    if not METADATA_YOLU.exists():
        raise FileNotFoundError(METADATA_YOLU)

    veri = pd.read_csv(DATASET_YOLU)
    metadata = pd.read_csv(METADATA_YOLU)
    x = veri[OZELLIK_KOLONLARI]
    y = veri[HEDEF_KOLONU].astype(int)

    pipeline = _pipeline_olustur()
    pipeline.fit(x, y)

    sweep = _json_oku(SWEEP_RAPORU) if SWEEP_RAPORU.exists() else {}
    selection = _json_oku(SELECTION_RAPORU) if SELECTION_RAPORU.exists() else {}
    holdout = _1000_holdout_ozeti(sweep)
    cv = _1000_cv_ozeti(sweep)
    holdout_metrikleri = holdout.get("tuned_threshold_metrics", {})
    threshold = float(holdout_metrikleri.get("threshold", 0.45))

    class_distribution = {str(k): int(v) for k, v in veri[HEDEF_KOLONU].value_counts().sort_index().items()}
    synthetic_rows = int(metadata["is_synthetic"].astype(bool).sum())
    original_rows = int((~metadata["is_synthetic"].astype(bool)).sum())

    _artifactleri_yedekle()
    kaydedilenler = artifactleri_kaydet(
        ARTIFACT_KLASORU,
        en_iyi_pipeline=pipeline,
        kalibrator=pipeline,
        esik_yapilandirmasi={
            "ikili_siniflama_esikleri": {
                "sentetik_1000_benchmark": {
                    "esik": threshold,
                    "aciklama": "1000/1000 source_id kontrollu sentetik benchmark ic validasyon/holdout dengesine gore secilen esik.",
                }
            },
            "onerilen_ikili_siniflama_esigi": threshold,
            "onerilen_ikili_siniflama_yontemi": "source_id_kontrollu_1000_sentetik_benchmark",
            "risk_kategorileri": {
                "dusuk_ust_esik": 0.33,
                "orta_ust_esik": 0.66,
                "etiketler": ["dusuk", "orta", "yuksek"],
            },
        },
        ozellik_sirasi=list(OZELLIK_KOLONLARI),
        metrik_ozeti=holdout_metrikleri,
        model_metadata={
            "veri_seti": "PIMA + source_id kontrollu sentetik benchmark",
            "model_adi": "ExtraTrees",
            "mod": "selected_source_id_controlled_synthetic_benchmark",
            "final_benchmark_adayi": "1000/1000",
            "veri_yolu": str(DATASET_YOLU),
            "metadata_yolu": str(METADATA_YOLU),
            "feature_set": "high_signal_features",
            "model_girdi_kolonlari": list(OZELLIK_KOLONLARI),
            "model_ic_ozellikler": [
                "Pregnancies",
                "Glucose",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
                "Insulin",
                "glucose_bmi_interaction",
                "insulin_glucose_ratio",
                "glucose_pedigree_interaction",
            ],
            "orijinal_dev_satiri": original_rows,
            "sentetik_satir": synthetic_rows,
            "toplam_satir": int(len(veri)),
            "sinif_dagilimi": class_distribution,
            "source_id_kurali": "Her sentetik satir kaynak aldigi original_{index} ailesine baglidir.",
            "independent_synthetic_source_id_count": int(
                metadata.loc[metadata["is_synthetic"].astype(bool), "source_id"].astype(str).str.startswith(("synthetic_", "gaussian_")).sum()
            ),
            "benchmark_holdout_metrikleri": holdout_metrikleri,
            "benchmark_group_cv": cv.get("summary", {}),
            "selection_report_final": selection.get("final", {}),
            "not": (
                "Bu artifact 1000/1000 source_id kontrollu sentetik benchmark secimine gore uretilmistir. "
                "Original external holdout klinik genellenebilirlik kaniti degil, dis kontrol siniri olarak raporlanir."
            ),
        },
    )

    print("1000/1000 final artifact uretildi.")
    print(f"Dataset: {DATASET_YOLU}")
    print(f"Artifact klasoru: {ARTIFACT_KLASORU}")
    print(f"Yedek klasoru: {YEDEK_KLASORU if YEDEK_KLASORU.exists() else '-'}")
    for ad, yol in kaydedilenler.items():
        print(f"- {ad}: {yol} ({yol.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
