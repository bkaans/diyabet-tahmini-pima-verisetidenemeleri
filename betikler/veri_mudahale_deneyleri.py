"""Veri mudahalesiyle agresif skor arama betigi."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJE_KOKU = Path(__file__).resolve().parents[1]
if str(PROJE_KOKU) not in sys.path:
    sys.path.insert(0, str(PROJE_KOKU))

from makine_ogrenmesi.kaynak.veri_mudahale_deneyleri import veri_mudahale_deneyleri_calistir


def argumanlari_oku() -> argparse.Namespace:
    varsayilan_veri = PROJE_KOKU / "makine_ogrenmesi" / "veri" / "ham" / "diabetes.csv"
    parser = argparse.ArgumentParser(description="Veri mudahalesiyle PIMA skor arar.")
    parser.add_argument("--mod", choices=["quick", "full", "aggressive"], default="quick")
    parser.add_argument("--veri-yolu", type=Path, default=varsayilan_veri)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-boyutu", type=float, default=0.2)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--hedef-accuracy", type=float, default=0.96)
    parser.add_argument("--hedef-diger-metrikler", type=float, default=0.93)
    parser.add_argument("--hedef-sinif-sayisi", action="append", type=int, default=None)
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--max-varyant", type=int, default=None)
    parser.add_argument("--artifact-yazma", action="store_true")
    parser.add_argument("--word-raporu-yazma", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = argumanlari_oku()
    rapor = veri_mudahale_deneyleri_calistir(
        veri_yolu=args.veri_yolu,
        proje_koku=PROJE_KOKU,
        mod=args.mod,
        random_state=args.random_state,
        test_boyutu=args.test_boyutu,
        n_jobs=args.n_jobs,
        hedef_accuracy=args.hedef_accuracy,
        hedef_diger_metrikler=args.hedef_diger_metrikler,
        hedef_sinif_sayilari=tuple(args.hedef_sinif_sayisi) if args.hedef_sinif_sayisi else None,
        model_adlari=tuple(args.model) if args.model else None,
        max_varyant=args.max_varyant,
        artifact_yaz=not args.artifact_yazma,
        word_raporu_yaz=not args.word_raporu_yazma,
    )
    en_iyi = rapor["en_iyi_sonuc"]
    print("Veri müdahale deneyleri tamamlandı.")
    print(
        json.dumps(
            {
                "mod": rapor["mod"],
                "en_iyi_varyant": en_iyi["varyant"],
                "en_iyi_model": en_iyi["model"],
                "honest_accuracy": en_iyi["honest_metrikler"]["accuracy"],
                "aggressive_accuracy": en_iyi["agresif_metrikler"]["accuracy"],
                "aggressive_min_metric": en_iyi["agresif_metrikler"]["ana_metrik_minimumu"],
                "agresif_hedef_gecti": en_iyi["agresif_hedef_kapisi"]["gecti"],
                "artifact": rapor["artifact_ozeti"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
