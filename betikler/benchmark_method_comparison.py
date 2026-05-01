"""Strict 2700 ve controlled_benchmark 2500 sentetik benchmark fark analizi."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone


PROJE_KOKU = Path(__file__).resolve().parents[1]
if str(PROJE_KOKU) not in sys.path:
    sys.path.insert(0, str(PROJE_KOKU))

from makine_ogrenmesi.kaynak.controlled_synthetic_benchmark import (  # noqa: E402
    RANDOM_STATE,
    DatasetCandidate,
    FeatureSpec,
    ModelSpec,
    _apply_imputer,
    _evaluate_holdout,
    _external_holdout_ayir,
    _feature_specs,
    _fit_dev_median_imputer,
    _group_holdout_split,
    _json_clean,
    _model_specs,
)
from makine_ogrenmesi.kaynak.ozellik_yapilandirmasi import HEDEF_KOLONU, OZELLIK_KOLONLARI  # noqa: E402
from makine_ogrenmesi.kaynak.sentetik_benchmark_optimizasyonu import _xgboost_model as _strict_xgb_model  # noqa: E402
from makine_ogrenmesi.kaynak.veri_yukleyici import veri_setini_yukle  # noqa: E402


HIGH_SIGNAL = ["Glucose", "BMI", "Age", "DiabetesPedigreeFunction"]
CIKTILAR = PROJE_KOKU / "makine_ogrenmesi" / "raporlar"
VERI_DIR = PROJE_KOKU / "makine_ogrenmesi" / "veri" / "deneysel"


def main() -> None:
    raw = veri_setini_yukle(PROJE_KOKU / "makine_ogrenmesi" / "veri" / "ham" / "diabetes.csv").copy()
    raw["original_index"] = raw.index.astype(int)
    raw["source_id"] = [f"original_{i}" for i in raw["original_index"]]
    original_dev_raw, external_raw = _external_holdout_ayir(raw, RANDOM_STATE)
    imputer = _fit_dev_median_imputer(original_dev_raw)
    original_dev = _apply_imputer(original_dev_raw, imputer)
    external_holdout = _apply_imputer(external_raw, imputer)

    strict2700 = _dataset_from_files("sentetik_2700_per_class_pima", 2700)
    controlled_benchmark_2500 = _dataset_from_files("controlled_synthetic_2500_per_class_pima", 2500)

    feature_specs = {f.name: f for f in _feature_specs()}
    all_features = feature_specs["all_features"]
    no_skin = feature_specs["no_skinthickness"]
    model_specs = {m.name: m for m in _model_specs(RANDOM_STATE, n_jobs=4, quick=False)}
    new_soft = model_specs["soft_voting_xgb_lgbm_et"]
    strict_xgb = ModelSpec(
        name="strict_xgboost",
        estimator=_strict_xgb_model(RANDOM_STATE, n_jobs=4),
        preprocessing="tree_models_no_scaler",
        resampling="none",
        description="Eski strict hattin XGBoost ayarlari.",
    )

    cross_results = {
        "strict_xgb_no_skin_on_strict2700": _eval("strict_xgb_no_skin_on_strict2700", strict2700, no_skin, strict_xgb),
        "strict_xgb_no_skin_on_controlled_benchmark_2500": _eval("strict_xgb_no_skin_on_controlled_benchmark_2500", controlled_benchmark_2500, no_skin, strict_xgb),
        "new_soft_all_on_strict2700": _eval("new_soft_all_on_strict2700", strict2700, all_features, new_soft),
        "new_soft_all_on_controlled_benchmark_2500": _eval("new_soft_all_on_controlled_benchmark_2500", controlled_benchmark_2500, all_features, new_soft),
    }
    feature_effect = {
        "strict_xgb_strict2700_all_vs_no_skin": {
            "all_features": _eval("strict_xgb_all_on_strict2700", strict2700, all_features, strict_xgb),
            "no_skinthickness": cross_results["strict_xgb_no_skin_on_strict2700"],
        },
        "new_soft_controlled_benchmark_2500_all_vs_no_skin": {
            "all_features": cross_results["new_soft_all_on_controlled_benchmark_2500"],
            "no_skinthickness": _eval("new_soft_no_skin_on_controlled_benchmark_2500", controlled_benchmark_2500, no_skin, new_soft),
        },
    }
    threshold_effect = {
        name: {
            "default_min_main_metric": result["default_threshold_metrics"]["ana_metrik_minimumu"],
            "tuned_min_main_metric": result["tuned_threshold_metrics"]["ana_metrik_minimumu"],
            "delta": result["tuned_threshold_metrics"]["ana_metrik_minimumu"]
            - result["default_threshold_metrics"]["ana_metrik_minimumu"],
            "selected_threshold": result["threshold_report"]["selected_threshold"],
        }
        for name, result in cross_results.items()
    }

    frames = {
        "original_dev": original_dev[OZELLIK_KOLONLARI + [HEDEF_KOLONU]],
        "strict2700_all_rows": strict2700.frame,
        "strict2700_synthetic_only": _synthetic_only(strict2700),
        "controlled_benchmark_2500_all_rows": controlled_benchmark_2500.frame,
        "controlled_benchmark_2500_synthetic_only": _synthetic_only(controlled_benchmark_2500),
        "external_holdout": external_holdout[OZELLIK_KOLONLARI + [HEDEF_KOLONU]],
    }
    distribution = {
        name: {
            "rows": int(len(frame)),
            "class_distribution": _class_distribution(frame),
            "class_feature_stats": _class_feature_stats(frame),
            "class_separation": _class_separation(frame),
        }
        for name, frame in frames.items()
    }
    distribution_shift = {
        name: _distribution_shift(original_dev, frame)
        for name, frame in frames.items()
        if name != "original_dev"
    }

    generation_comparison = {
        "strict2700": {
            "source": "makine_ogrenmesi/kaynak/sentetik_benchmark_optimizasyonu.py",
            "target": "2700 / class",
            "methods": {
                "source_bound_bootstrap_noise": "Yaklasik %58; parent etrafinda std * 0.060 noise.",
                "source_bound_local_gaussian_noise": "Yaklasik %42; feature bazli std * 0.050-0.090 noise ve orta sinif ayrimi guclendirme.",
            },
            "class_signal": "Pozitif sinifta Glucose +16, BMI +2.2, Insulin +18, DPF +0.055; negatifte daha dusuk ters kaydirma.",
            "clipping": "Klinik min/max sinirlari; quantile-preserving yok.",
            "metadata": "source_id, is_synthetic, parent_original_index, generation_method.",
        },
        "controlled_benchmark_2500": {
            "source": "makine_ogrenmesi/kaynak/controlled_synthetic_benchmark.py",
            "target": "2500 / class",
            "methods": {
                "one_parent_gaussian_local": "Yaklasik %24; parent ile sinif hedef merkezi arasinda 0.62-0.82 guc.",
                "class_conditional_source_bound_gaussian": "Yaklasik %34; parent ile sinif hedef merkezi arasinda 0.76-0.94 guc.",
                "quantile_preserving_local_augmentation": "Yaklasik %24; parent-hedef interpolasyonu ve quantile clipping.",
                "controlled_jitter_augmentation": "Yaklasik %18; daha dusuk guclu kontrollu jitter.",
            },
            "class_signal": "Hard-coded class target centers: pozitif Glucose 174/BMI 40.5/DPF 0.86/Age 52; negatif Glucose 86/BMI 24.5/DPF 0.23/Age 29.",
            "clipping": "Klinik min/max + sinif ici quantile sinirlari; strong profilde sinirlar genisletiliyor.",
            "metadata": "source_id, is_synthetic, parent_original_index, generation_method, synthetic_strength, noise_profile.",
        },
    }

    old_on_old = cross_results["strict_xgb_no_skin_on_strict2700"]["tuned_threshold_metrics"]
    old_on_new = cross_results["strict_xgb_no_skin_on_controlled_benchmark_2500"]["tuned_threshold_metrics"]
    new_on_old = cross_results["new_soft_all_on_strict2700"]["tuned_threshold_metrics"]
    new_on_new = cross_results["new_soft_all_on_controlled_benchmark_2500"]["tuned_threshold_metrics"]
    data_effect = old_on_new["ana_metrik_minimumu"] - old_on_old["ana_metrik_minimumu"]
    model_effect_old_data = new_on_old["ana_metrik_minimumu"] - old_on_old["ana_metrik_minimumu"]
    model_effect_new_data = new_on_new["ana_metrik_minimumu"] - old_on_new["ana_metrik_minimumu"]
    strict_sep = distribution["strict2700_all_rows"]["class_separation"]["aggregate"]["mean_abs_cohens_d"]
    controlled_benchmark_sep = distribution["controlled_benchmark_2500_all_rows"]["class_separation"]["aggregate"]["mean_abs_cohens_d"]
    strict_shift = distribution_shift["strict2700_all_rows"]["aggregate"]["avg_abs_z_mean_shift"]
    controlled_benchmark_shift = distribution_shift["controlled_benchmark_2500_all_rows"]["aggregate"]["avg_abs_z_mean_shift"]

    conclusion = {
        "farkin_ana_nedeni": (
            "Ana fark modelden cok veri uretiminden geliyor. Controlled Benchmark uretimi parent satiri sinif hedef merkezlerine "
            "yuksek synthetic_strength ile yaklastirdigi icin Glucose, BMI, Age ve DPF sinif ayrimi belirgin artiyor."
        ),
        "veri_uretimi_etkisi_min_main_metric_delta": data_effect,
        "model_etkisi_strict_data_delta": model_effect_old_data,
        "model_etkisi_controlled_benchmark_data_delta": model_effect_new_data,
        "threshold_etkisi": threshold_effect,
        "feature_set_etkisi": _feature_delta_summary(feature_effect),
        "synthetic_distribution_shift": {
            "strict2700_avg_abs_z_shift": strict_shift,
            "controlled_benchmark_2500_avg_abs_z_shift": controlled_benchmark_shift,
            "strict2700_mean_abs_cohens_d": strict_sep,
            "controlled_benchmark_2500_mean_abs_cohens_d": controlled_benchmark_sep,
            "var_mi": bool(controlled_benchmark_shift > strict_shift * 1.25 or controlled_benchmark_sep > strict_sep * 1.25),
        },
        "rapor_dili": (
            "Yeni skorlar 'ham PIMA genellenebilirliği' olarak değil, source_id aile ayrımı korunmuş fakat "
            "sınıf hedef merkezleriyle güçlendirilmiş kontrollü sentetik benchmark sonucu olarak açıklanmalı."
        ),
    }

    report = {
        "created_for": "strict benchmark and controlled benchmark method comparison",
        "generation_comparison": generation_comparison,
        "distribution": distribution,
        "distribution_shift_vs_original_dev": distribution_shift,
        "cross_model_experiments": _json_clean(cross_results),
        "feature_effect_experiments": _json_clean(feature_effect),
        "threshold_effect": _json_clean(threshold_effect),
        "decomposition": _json_clean(conclusion),
    }
    out = CIKTILAR / "benchmark_method_comparison.json"
    out.write_text(json.dumps(_json_clean(report), ensure_ascii=False, indent=2), encoding="utf-8")

    print("- Farkın ana nedeni:", conclusion["farkin_ana_nedeni"])
    print(f"- Veri üretimi etkisi: strict XGB aynı modelde min ana metrik delta = {data_effect:.4f}")
    print(
        "- Model etkisi:",
        f"strict veri üzerinde delta = {model_effect_old_data:.4f}; controlled_benchmark veri üzerinde delta = {model_effect_new_data:.4f}",
    )
    print("- Threshold etkisi:", _threshold_summary(threshold_effect))
    print("- Feature set etkisi:", conclusion["feature_set_etkisi"])
    print(
        "- Synthetic distribution shift var mı:",
        "Evet" if conclusion["synthetic_distribution_shift"]["var_mi"] else "Sınırlı",
        f"(strict shift={strict_shift:.3f}, controlled_benchmark shift={controlled_benchmark_shift:.3f})",
    )
    print("- Rapor dilinde nasıl açıklanmalı:", conclusion["rapor_dili"])
    print("- JSON:", out)


def _dataset_from_files(name: str, target: int) -> DatasetCandidate:
    frame = pd.read_csv(VERI_DIR / f"{name}.csv")
    metadata = pd.read_csv(VERI_DIR / f"{name}_metadata.csv")
    return DatasetCandidate(
        name=name,
        target_per_class=target,
        frame=frame[OZELLIK_KOLONLARI + [HEDEF_KOLONU]].reset_index(drop=True),
        metadata=metadata.reset_index(drop=True),
        report={"dataset_name": name, "target_per_class": target},
    )


def _eval(label: str, dataset: DatasetCandidate, feature: FeatureSpec, model: ModelSpec) -> dict[str, Any]:
    result = _evaluate_holdout(
        dataset=dataset,
        split=_group_holdout_split(dataset, RANDOM_STATE),
        feature_spec=feature,
        model_spec=model,
        random_state=RANDOM_STATE,
    )
    result["experiment_label"] = label
    return result


def _synthetic_only(dataset: DatasetCandidate) -> pd.DataFrame:
    mask = dataset.metadata["is_synthetic"].astype(bool).to_numpy()
    return dataset.frame.loc[mask, OZELLIK_KOLONLARI + [HEDEF_KOLONU]].reset_index(drop=True)


def _class_distribution(frame: pd.DataFrame) -> dict[str, int]:
    return {str(int(k)): int(v) for k, v in frame[HEDEF_KOLONU].value_counts().sort_index().to_dict().items()}


def _class_feature_stats(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cls in [0, 1]:
        part = frame[frame[HEDEF_KOLONU] == cls]
        result[str(cls)] = {}
        for col in HIGH_SIGNAL:
            s = pd.to_numeric(part[col], errors="coerce")
            result[str(cls)][col] = {
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "q10": float(s.quantile(0.10)),
                "q50": float(s.quantile(0.50)),
                "q90": float(s.quantile(0.90)),
            }
    return result


def _class_separation(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    abs_ds = []
    aucs = []
    y = frame[HEDEF_KOLONU].to_numpy(dtype=int)
    for col in HIGH_SIGNAL:
        neg = frame.loc[frame[HEDEF_KOLONU] == 0, col].astype(float)
        pos = frame.loc[frame[HEDEF_KOLONU] == 1, col].astype(float)
        pooled = np.sqrt(((len(neg) - 1) * neg.var(ddof=1) + (len(pos) - 1) * pos.var(ddof=1)) / max(len(neg) + len(pos) - 2, 1))
        d = float((pos.mean() - neg.mean()) / pooled) if pooled else 0.0
        auc = _single_feature_auc(y, frame[col].astype(float).to_numpy())
        result[col] = {
            "positive_minus_negative_mean": float(pos.mean() - neg.mean()),
            "cohens_d": d,
            "absolute_cohens_d": abs(d),
            "single_feature_auc_directionless": auc,
        }
        abs_ds.append(abs(d))
        aucs.append(auc)
    result["aggregate"] = {
        "mean_abs_cohens_d": float(np.mean(abs_ds)),
        "mean_single_feature_auc_directionless": float(np.mean(aucs)),
    }
    return result


def _single_feature_auc(y: np.ndarray, values: np.ndarray) -> float:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_ranks_pos = float(ranks[y == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(max(auc, 1 - auc))


def _distribution_shift(reference: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    shifts = []
    for cls in [0, 1]:
        ref = reference[reference[HEDEF_KOLONU] == cls]
        part = frame[frame[HEDEF_KOLONU] == cls]
        result[str(cls)] = {}
        for col in HIGH_SIGNAL:
            denom = float(ref[col].astype(float).std(ddof=1)) or 1.0
            shift = float((part[col].astype(float).mean() - ref[col].astype(float).mean()) / denom)
            result[str(cls)][col] = shift
            shifts.append(abs(shift))
    result["aggregate"] = {"avg_abs_z_mean_shift": float(np.mean(shifts)), "max_abs_z_mean_shift": float(np.max(shifts))}
    return result


def _feature_delta_summary(feature_effect: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for name, pair in feature_effect.items():
        keys = list(pair)
        a, b = pair[keys[0]], pair[keys[1]]
        out[name] = {
            keys[0]: a["tuned_threshold_metrics"]["ana_metrik_minimumu"],
            keys[1]: b["tuned_threshold_metrics"]["ana_metrik_minimumu"],
            "delta_second_minus_first": b["tuned_threshold_metrics"]["ana_metrik_minimumu"] - a["tuned_threshold_metrics"]["ana_metrik_minimumu"],
        }
    return out


def _threshold_summary(threshold_effect: dict[str, Any]) -> dict[str, float]:
    return {name: round(v["delta"], 4) for name, v in threshold_effect.items()}


if __name__ == "__main__":
    main()
