"""Veriye dokunmadan maksimum accuracy arama betigi."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJE_KOKU = Path(__file__).resolve().parents[1]
if str(PROJE_KOKU) not in sys.path:
    sys.path.insert(0, str(PROJE_KOKU))

from makine_ogrenmesi.kaynak.maksimum_skor_arama import ml_yeniden_kur


def argumanlari_oku() -> argparse.Namespace:
    varsayilan_veri = PROJE_KOKU / "makine_ogrenmesi" / "veri" / "ham" / "diabetes.csv"

    parser = argparse.ArgumentParser(
        description="PIMA uzerinde veriyi degistirmeden maksimum skor arar."
    )
    parser.add_argument(
        "--mod",
        choices=["quick", "tenfold", "repeated", "nested", "aggressive"],
        required=True,
    )
    parser.add_argument("--veri-yolu", type=Path, default=varsayilan_veri)
    parser.add_argument("--test-boyutu", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--aggressive-seeds",
        type=int,
        default=30,
        help="Aggressive modda denenecek ardışık seed sayısı.",
    )
    return parser.parse_args()


def main() -> None:
    args = argumanlari_oku()
    rapor = ml_yeniden_kur(
        mod=args.mod,
        veri_yolu=args.veri_yolu,
        proje_koku=PROJE_KOKU,
        test_boyutu=args.test_boyutu,
        random_state=args.random_state,
        n_iter=args.n_iter,
        n_jobs=args.n_jobs,
        aggressive_seeds=args.aggressive_seeds,
    )
    print("ML yeniden kurulum tamamlandi.")
    print(
        json.dumps(
            {
                "mod": rapor["mod"],
                "final_test_metrikleri": rapor.get("final_test_metrikleri"),
                "onay_kapisi": rapor.get("onay_kapisi"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
