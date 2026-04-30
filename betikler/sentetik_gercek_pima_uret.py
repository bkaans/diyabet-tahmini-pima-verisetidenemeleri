"""Sentetik gercek PIMA veri setini uretir."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJE_KOKU = Path(__file__).resolve().parents[1]
if str(PROJE_KOKU) not in sys.path:
    sys.path.insert(0, str(PROJE_KOKU))

from makine_ogrenmesi.kaynak.sentetik_benchmark_optimizasyonu import sentetik_benchmark_calistir


def argumanlari_oku() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal feature sentetik PIMA benchmark veri uretimi.")
    parser.add_argument("--mod", choices=["quick", "full", "target"], default="quick")
    parser.add_argument(
        "--veri-yolu",
        type=Path,
        default=PROJE_KOKU / "makine_ogrenmesi" / "veri" / "ham" / "diabetes.csv",
    )
    parser.add_argument("--hedef-sinif-sayisi", action="append", type=int, default=None)
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--artifact-yazma", action="store_true")
    parser.add_argument("--word-raporu-yazma", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = argumanlari_oku()
    rapor = sentetik_benchmark_calistir(
        proje_koku=PROJE_KOKU,
        veri_yolu=args.veri_yolu,
        mod=args.mod,
        hedef_sinif_sayilari=tuple(args.hedef_sinif_sayisi) if args.hedef_sinif_sayisi else None,
        model_adlari=tuple(args.model) if args.model else None,
        n_jobs=args.n_jobs,
        artifact_yaz=not args.artifact_yazma,
        word_raporu_yaz=not args.word_raporu_yazma,
    )
    en_iyi = rapor["en_iyi_sonuc"]
    print(
        json.dumps(
            {
                "final_veri": rapor["final_veri"]["yol"],
                "feature_set": en_iyi["feature_set"],
                "model": en_iyi["model"],
                "accuracy": en_iyi["test_metrikleri"]["accuracy"],
                "minimum_ana_metrik": en_iyi["test_metrikleri"]["ana_metrik_minimumu"],
                "hedef_gecti": en_iyi["hedef_kapisi"]["gecti"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
