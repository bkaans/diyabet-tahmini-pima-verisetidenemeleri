"""Literatur destekli PIMA deneylerini calistiran CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJE_KOKU = Path(__file__).resolve().parents[1]
if str(PROJE_KOKU) not in sys.path:
    sys.path.insert(0, str(PROJE_KOKU))

from makine_ogrenmesi.kaynak.literatur_deneyleri import literatur_deneyleri_calistir


def argumanlari_oku() -> argparse.Namespace:
    varsayilan_veri = PROJE_KOKU / "makine_ogrenmesi" / "veri" / "ham" / "diabetes.csv"
    parser = argparse.ArgumentParser(
        description="PIMA literatur profillerini ham CSV'yi degistirmeden dener."
    )
    parser.add_argument("--veri-yolu", type=Path, default=varsayilan_veri)
    parser.add_argument("--test-boyutu", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-iter", type=int, default=16)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--artifact-yaz",
        action="store_true",
        help="En iyi literatur sonucunu mevcut artifact klasorune yazar.",
    )
    parser.add_argument(
        "--deney",
        action="append",
        default=None,
        help="Sadece verilen deney adini calistirir. Birden fazla kez verilebilir.",
    )
    return parser.parse_args()


def main() -> None:
    args = argumanlari_oku()
    rapor = literatur_deneyleri_calistir(
        veri_yolu=args.veri_yolu,
        proje_koku=PROJE_KOKU,
        test_boyutu=args.test_boyutu,
        random_state=args.random_state,
        n_iter=args.n_iter,
        n_jobs=args.n_jobs,
        artifact_yaz=args.artifact_yaz,
        deney_adlari=tuple(args.deney) if args.deney else None,
    )
    print("Literatur deneyleri tamamlandi.")
    print(
        json.dumps(
            {
                "tamamlanan_deney_sayisi": rapor["tamamlanan_deney_sayisi"],
                "atlanan_deney_sayisi": rapor["atlanan_deney_sayisi"],
                "en_iyi_cv": {
                    "deney_adi": rapor["en_iyi_cv_sonuc"]["deney_adi"]
                    if rapor["en_iyi_cv_sonuc"]
                    else None,
                    "cv_accuracy": rapor["en_iyi_cv_sonuc"]["cv_metrikleri"]["accuracy"]
                    if rapor["en_iyi_cv_sonuc"]
                    else None,
                    "holdout_accuracy": rapor["en_iyi_cv_sonuc"]["holdout_metrikleri"]["accuracy"]
                    if rapor["en_iyi_cv_sonuc"]
                    else None,
                },
                "onay_kapisi": rapor["onay_kapisi"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
