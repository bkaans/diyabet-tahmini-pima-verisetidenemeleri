"""Terminalden model egitim ve grid arama islemini calistirir."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

PROJE_KOKU = Path(__file__).resolve().parents[1]
if str(PROJE_KOKU) not in sys.path:
    sys.path.insert(0, str(PROJE_KOKU))

from makine_ogrenmesi.kaynak.model_egitimi import grid_searchleri_olustur
from makine_ogrenmesi.kaynak.ozellik_yapilandirmasi import HEDEF_KOLONU, OZELLIK_KOLONLARI
from makine_ogrenmesi.kaynak.veri_yukleyici import veri_setini_yukle


def argumanlari_oku() -> argparse.Namespace:
    varsayilan_veri = PROJE_KOKU / "makine_ogrenmesi" / "veri" / "ham" / "diabetes.csv"
    varsayilan_cikti = (
        PROJE_KOKU / "makine_ogrenmesi" / "raporlar" / "degerlendirme" / "egitim_ozeti.json"
    )

    parser = argparse.ArgumentParser(description="Model egitim ve grid arama betigi")
    parser.add_argument("--veri-yolu", type=Path, default=varsayilan_veri)
    parser.add_argument("--test-boyutu", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--cikti-yolu", type=Path, default=varsayilan_cikti)
    return parser.parse_args()


def klasorleri_hazirla(proje_koku: Path) -> None:
    (proje_koku / "makine_ogrenmesi" / "artifactler").mkdir(parents=True, exist_ok=True)
    (proje_koku / "makine_ogrenmesi" / "raporlar" / "degerlendirme").mkdir(
        parents=True, exist_ok=True
    )


def main() -> None:
    args = argumanlari_oku()
    klasorleri_hazirla(PROJE_KOKU)

    print("Egitim baslatiliyor...")
    print(f"Veri yolu: {args.veri_yolu}")

    veri_cercevesi = veri_setini_yukle(args.veri_yolu)
    ozellikler = veri_cercevesi[OZELLIK_KOLONLARI]
    hedef = veri_cercevesi[HEDEF_KOLONU]

    x_egitim, x_test, y_egitim, _ = train_test_split(
        ozellikler,
        hedef,
        test_size=args.test_boyutu,
        random_state=args.random_state,
        stratify=hedef,
    )

    print(f"Egitim satir sayisi: {len(x_egitim)}")
    print(f"Test satir sayisi: {len(x_test)}")

    grid_searchler = grid_searchleri_olustur(scoring="roc_auc", n_jobs=args.n_jobs)
    egitim_ozeti: list[dict[str, object]] = []

    for model_adi, grid_search in grid_searchler.items():
        print(f"[{model_adi}] egitimi basladi...")
        grid_search.fit(x_egitim, y_egitim)
        sonuc = {
            "model_adi": model_adi,
            "en_iyi_cv_roc_auc": float(grid_search.best_score_),
            "en_iyi_parametreler": _json_uyumlu(grid_search.best_params_),
        }
        egitim_ozeti.append(sonuc)
        print(
            f"[{model_adi}] tamamlandi | en_iyi_cv_roc_auc={sonuc['en_iyi_cv_roc_auc']:.4f}"
        )

    en_iyi = max(egitim_ozeti, key=lambda x: x["en_iyi_cv_roc_auc"])
    genel_ozet = {
        "veri_yolu": str(args.veri_yolu),
        "test_boyutu": args.test_boyutu,
        "random_state": args.random_state,
        "en_iyi_model": en_iyi,
        "tum_model_sonuclari": egitim_ozeti,
    }

    args.cikti_yolu.parent.mkdir(parents=True, exist_ok=True)
    args.cikti_yolu.write_text(
        json.dumps(genel_ozet, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Egitim ozet dosyasi yazildi: {args.cikti_yolu}")
    print("Egitim sureci tamamlandi.")


def _json_uyumlu(deger):
    if hasattr(deger, "item"):
        return deger.item()
    if isinstance(deger, dict):
        return {k: _json_uyumlu(v) for k, v in deger.items()}
    if isinstance(deger, list):
        return [_json_uyumlu(v) for v in deger]
    return deger


if __name__ == "__main__":
    main()
