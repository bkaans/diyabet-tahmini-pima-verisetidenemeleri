"""500/500 muhafazakar distribution-matched sentetik sweep komutu."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJE_KOKU = Path(__file__).resolve().parents[1]
if str(PROJE_KOKU) not in sys.path:
    sys.path.insert(0, str(PROJE_KOKU))

from makine_ogrenmesi.kaynak.conservative_500_sweep import conservative_500_sweep_calistir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="500/500 skorlarini koruyarak daha dusuk Cohen's d ve shift arayan detayli sweep."
    )
    parser.add_argument("--veri-yolu", type=Path, default=PROJE_KOKU / "makine_ogrenmesi" / "veri" / "ham" / "diabetes.csv")
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--word-raporu-yazma", action="store_true")
    args = parser.parse_args()

    conservative_500_sweep_calistir(
        proje_koku=PROJE_KOKU,
        veri_yolu=args.veri_yolu,
        n_jobs=args.n_jobs,
        quick=args.quick,
        word_raporu_yaz=not args.word_raporu_yazma,
    )


if __name__ == "__main__":
    main()
