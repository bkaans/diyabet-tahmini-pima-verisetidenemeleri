"""Terminalden model degerlendirme islemini calistirir."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

PROJE_KOKU = Path(__file__).resolve().parents[1]
if str(PROJE_KOKU) not in sys.path:
    sys.path.insert(0, str(PROJE_KOKU))

from makine_ogrenmesi.kaynak.model_degerlendirme import (
    en_iyi_modeli_sec,
    model_sonuc_ozeti_olustur,
    model_sonuclarini_sirala,
)
from makine_ogrenmesi.kaynak.model_egitimi import grid_searchleri_olustur
from makine_ogrenmesi.kaynak.ozellik_yapilandirmasi import HEDEF_KOLONU, OZELLIK_KOLONLARI
from makine_ogrenmesi.kaynak.veri_yukleyici import veri_setini_yukle


def argumanlari_oku() -> argparse.Namespace:
    varsayilan_veri = PROJE_KOKU / "makine_ogrenmesi" / "veri" / "ham" / "diabetes.csv"
    varsayilan_cikti = (
        PROJE_KOKU
        / "makine_ogrenmesi"
        / "raporlar"
        / "degerlendirme"
        / "model_degerlendirme_ozeti.json"
    )

    parser = argparse.ArgumentParser(description="Model degerlendirme betigi")
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

    print("Degerlendirme baslatiliyor...")
    print(f"Veri yolu: {args.veri_yolu}")

    veri_cercevesi = veri_setini_yukle(args.veri_yolu)
    ozellikler = veri_cercevesi[OZELLIK_KOLONLARI]
    hedef = veri_cercevesi[HEDEF_KOLONU]

    x_egitim, x_test, y_egitim, y_test = train_test_split(
        ozellikler,
        hedef,
        test_size=args.test_boyutu,
        random_state=args.random_state,
        stratify=hedef,
    )

    grid_searchler = grid_searchleri_olustur(scoring="roc_auc", n_jobs=args.n_jobs)
    model_sonuclari: list[dict[str, object]] = []

    for model_adi, grid_search in grid_searchler.items():
        print(f"[{model_adi}] degerlendirme icin egitiliyor...")
        grid_search.fit(x_egitim, y_egitim)

        en_iyi_tahminleyici = grid_search.best_estimator_
        y_tahmin = en_iyi_tahminleyici.predict(x_test)
        y_olasilik = en_iyi_tahminleyici.predict_proba(x_test)[:, 1]

        sonuc = model_sonuc_ozeti_olustur(model_adi, y_test, y_tahmin, y_olasilik)
        sonuc["en_iyi_cv_roc_auc"] = float(grid_search.best_score_)
        sonuc["en_iyi_parametreler"] = _json_uyumlu(grid_search.best_params_)
        model_sonuclari.append(sonuc)

        print(
            f"[{model_adi}] tamamlandi | test_roc_auc={float(sonuc['roc_auc']):.4f}"
        )

    sirali_sonuclar = model_sonuclarini_sirala(model_sonuclari)
    en_iyi_test_modeli = en_iyi_modeli_sec(model_sonuclari)
    hedef_durumu = _hedef_durumu_hazirla(en_iyi_test_modeli)

    genel_ozet = {
        "veri_yolu": str(args.veri_yolu),
        "test_boyutu": args.test_boyutu,
        "random_state": args.random_state,
        "en_iyi_model_secim_kriteri": "roc_auc > recall > f1 > accuracy (deploy ile tutarlilik)",
        "en_iyi_model": _json_uyumlu(en_iyi_test_modeli),
        "en_iyi_test_modeli": _json_uyumlu(en_iyi_test_modeli),
        "sirali_sonuclar": _json_uyumlu(sirali_sonuclar),
        "performans_hedef_durumu": _json_uyumlu(hedef_durumu),
    }

    args.cikti_yolu.parent.mkdir(parents=True, exist_ok=True)
    args.cikti_yolu.write_text(
        json.dumps(genel_ozet, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Degerlendirme ozet dosyasi yazildi: {args.cikti_yolu}")
    print("Degerlendirme sureci tamamlandi.")


def _json_uyumlu(deger):
    if hasattr(deger, "item"):
        return deger.item()
    if isinstance(deger, dict):
        return {k: _json_uyumlu(v) for k, v in deger.items()}
    if isinstance(deger, list):
        return [_json_uyumlu(v) for v in deger]
    return deger


def _hedef_durumu_hazirla(en_iyi_test_modeli: dict[str, object]) -> dict[str, object]:
    hedefler = {
        "accuracy_min": 0.90,
        "roc_auc_min": 0.80,
        "f1_min": 0.70,
    }
    accuracy = float(en_iyi_test_modeli["accuracy"])
    roc_auc = float(en_iyi_test_modeli["roc_auc"])
    f1 = float(en_iyi_test_modeli["f1"])
    brier = float(en_iyi_test_modeli["brier"])
    durum = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "f1": f1,
        "brier": brier,
        "hedefler": hedefler,
        "hedefler_saglandi_mi": (
            accuracy >= hedefler["accuracy_min"]
            and roc_auc >= hedefler["roc_auc_min"]
            and f1 >= hedefler["f1_min"]
        ),
    }
    if durum["hedefler_saglandi_mi"]:
        durum["yorum"] = "Tanimli performans hedefleri saglandi."
    else:
        durum["yorum"] = (
            "Hedeflerin tamami saglanamadi. Proje planindaki B-plani devreye alindi ve "
            "AUC/F1 dengesine gore en tutarli model adaylari raporlandi."
        )
    return durum


if __name__ == "__main__":
    main()
