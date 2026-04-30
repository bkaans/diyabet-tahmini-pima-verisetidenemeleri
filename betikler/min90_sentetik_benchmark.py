"""Min-0.90 source_id kontrollu sentetik PIMA benchmark komutu."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJE_KOKU = Path(__file__).resolve().parents[1]
if str(PROJE_KOKU) not in sys.path:
    sys.path.insert(0, str(PROJE_KOKU))

from makine_ogrenmesi.kaynak.min90_sentetik_benchmark import (  # noqa: E402
    DEFAULT_TARGETS,
    min90_benchmark_calistir,
)


def argumanlari_al() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PIMA + source_id kontrollu sentetik benchmark uretir ve raporlar."
    )
    parser.add_argument(
        "--veri-yolu",
        type=Path,
        default=PROJE_KOKU / "makine_ogrenmesi" / "veri" / "ham" / "diabetes.csv",
        help="Orijinal PIMA diabetes.csv yolu.",
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        action="append",
        dest="targets",
        help="Denenecek sinif basi hedef satir sayisi. Birden fazla kez verilebilir.",
    )
    parser.add_argument("--quick", action="store_true", help="Kisa smoke test modu.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Destekleyen modeller icin paralel is sayisi.")
    parser.add_argument(
        "--word-raporu-yazma",
        action="store_true",
        help="DOCX raporu uretmeden yalniz JSON/CSV benchmark akisini calistir.",
    )
    return parser.parse_args()


def main() -> None:
    args = argumanlari_al()
    targets = tuple(args.targets) if args.targets else DEFAULT_TARGETS
    sonuc = min90_benchmark_calistir(
        proje_koku=PROJE_KOKU,
        veri_yolu=args.veri_yolu,
        target_per_class_values=targets,
        n_jobs=args.n_jobs,
        word_raporu_yaz=not args.word_raporu_yazma,
        quick=args.quick,
    )
    final = sonuc["final"]
    cv = sonuc["final_cv"]["summary"]
    print(
        json.dumps(
            {
                "final_dataset": final["dataset_name"],
                "final_model": final["model"],
                "final_feature_set": final["feature_set"],
                "synthetic_holdout_min_main_metric": final["tuned_threshold_metrics"][
                    "ana_metrik_minimumu"
                ],
                "synthetic_cv_min_main_metric_mean": cv["ana_metrik_minimumu_mean"],
                "word_report": str(sonuc.get("word_report") or ""),
                "desktop_word_report": str(sonuc.get("desktop_word_report") or ""),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
