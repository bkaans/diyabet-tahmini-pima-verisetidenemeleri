"""PIMA veri seti icin EDA raporu uretir."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJE_KOKU = Path(__file__).resolve().parents[1]
if str(PROJE_KOKU) not in sys.path:
    sys.path.insert(0, str(PROJE_KOKU))

from makine_ogrenmesi.kaynak.maksimum_skor_arama import eda_raporu_uret


def argumanlari_oku() -> argparse.Namespace:
    varsayilan_veri = PROJE_KOKU / "makine_ogrenmesi" / "veri" / "ham" / "diabetes.csv"
    varsayilan_md = PROJE_KOKU / "makine_ogrenmesi" / "raporlar" / "eda" / "eda_raporu.md"
    varsayilan_json = (
        PROJE_KOKU / "makine_ogrenmesi" / "raporlar" / "eda" / "eda_raporu.json"
    )

    parser = argparse.ArgumentParser(description="Detayli EDA raporu uretir.")
    parser.add_argument("--veri-yolu", type=Path, default=varsayilan_veri)
    parser.add_argument("--rapor-md-yolu", type=Path, default=varsayilan_md)
    parser.add_argument("--rapor-json-yolu", type=Path, default=varsayilan_json)
    return parser.parse_args()


def main() -> None:
    args = argumanlari_oku()
    eda_raporu_uret(
        veri_yolu=args.veri_yolu,
        rapor_md_yolu=args.rapor_md_yolu,
        rapor_json_yolu=args.rapor_json_yolu,
    )
    print(f"EDA Markdown raporu yazildi: {args.rapor_md_yolu}")
    print(f"EDA JSON raporu yazildi: {args.rapor_json_yolu}")


if __name__ == "__main__":
    main()
