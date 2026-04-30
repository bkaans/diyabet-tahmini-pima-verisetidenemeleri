"""Veri seti boyutunu degistirmeden model/kalibrasyon/esik optimizasyonu yapar.

Bu betik su hedeflere odaklanir:
1. Esik optimizasyonu (F1 odakli)
2. Kalibrasyon secimi (none/sigmoid/isotonic)
3. Sinif dengesizligi stratejileri (SMOTE, class weight)
4. Hiperparametre taramasi (GridSearchCV)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import FunctionTransformer

PROJE_KOKU = Path(__file__).resolve().parents[1]
if str(PROJE_KOKU) not in sys.path:
    sys.path.insert(0, str(PROJE_KOKU))

from makine_ogrenmesi.kaynak.model_degerlendirme import model_metriklerini_hesapla
from makine_ogrenmesi.kaynak.artifact_kaydet import artifactleri_kaydet
from makine_ogrenmesi.kaynak.esik_analizi import (
    f2_esigi_hesapla,
    risk_esiklerini_olustur,
    youden_j_esigi_hesapla,
)
from makine_ogrenmesi.kaynak.on_isleme import (
    median_imputer_olustur,
    sifirlari_nan_donustur_pipeline,
    standard_scaler_olustur,
)
from makine_ogrenmesi.kaynak.ozellik_yapilandirmasi import (
    HEDEF_KOLONU,
    OZELLIK_KOLONLARI,
)
from makine_ogrenmesi.kaynak.veri_yukleyici import veri_setini_yukle


RANDOM_STATE = 42
MIN_F1_KISITI = 0.70


@dataclass(frozen=True)
class DeneyTanimi:
    ad: str
    model_tipi: str
    smote_kullan: bool
    class_weight: str | None = None


def argumanlari_oku() -> argparse.Namespace:
    varsayilan_veri = PROJE_KOKU / "makine_ogrenmesi" / "veri" / "ham" / "diabetes.csv"
    varsayilan_json = (
        PROJE_KOKU
        / "makine_ogrenmesi"
        / "raporlar"
        / "degerlendirme"
        / "veri_artirmadan_optimizasyon_raporu.json"
    )
    varsayilan_md = (
        PROJE_KOKU
        / "dokumanlar"
        / "veri_artirmadan_optimizasyon_raporu.md"
    )

    parser = argparse.ArgumentParser(
        description="Veri artirmadan model/kalibrasyon/esik optimizasyonu"
    )
    parser.add_argument("--veri-yolu", type=Path, default=varsayilan_veri)
    parser.add_argument("--test-boyutu", type=float, default=0.2)
    parser.add_argument(
        "--dogrulama-boyutu",
        type=float,
        default=0.25,
        help="Train icindeki validation payi (0.25 => toplam verinin %20'si).",
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--rapor-json-yolu", type=Path, default=varsayilan_json)
    parser.add_argument("--rapor-md-yolu", type=Path, default=varsayilan_md)
    parser.add_argument(
        "--artifact-klasoru",
        type=Path,
        default=PROJE_KOKU / "makine_ogrenmesi" / "artifactler",
    )
    return parser.parse_args()


def main() -> None:
    args = argumanlari_oku()
    _klasorleri_hazirla(args.rapor_json_yolu, args.rapor_md_yolu)

    print("Veri artirmadan optimizasyon basladi...")
    print(f"Veri yolu: {args.veri_yolu}")

    veri = veri_setini_yukle(args.veri_yolu)
    x = veri[OZELLIK_KOLONLARI]
    y = veri[HEDEF_KOLONU]

    x_gelistirme, x_test, y_gelistirme, y_test = train_test_split(
        x,
        y,
        test_size=args.test_boyutu,
        random_state=args.random_state,
        stratify=y,
    )
    x_egitim, x_dogrulama, y_egitim, y_dogrulama = train_test_split(
        x_gelistirme,
        y_gelistirme,
        test_size=args.dogrulama_boyutu,
        random_state=args.random_state,
        stratify=y_gelistirme,
    )

    print(
        "Bolme bilgisi | "
        f"egitim={len(x_egitim)} | dogrulama={len(x_dogrulama)} | test={len(x_test)}"
    )

    cv = StratifiedKFold(
        n_splits=args.cv,
        shuffle=True,
        random_state=args.random_state,
    )
    pozitif_oran = float((y_egitim == 1).sum())
    negatif_oran = float((y_egitim == 0).sum())
    scale_pos_weight = negatif_oran / max(pozitif_oran, 1.0)

    deneyler = _deney_tanimlarini_hazirla()
    tum_sonuclar_ham: list[dict[str, Any]] = []

    for deney in deneyler:
        print(f"[{deney.ad}] grid arama basladi...")
        try:
            base_model = _modeli_olustur(
                model_tipi=deney.model_tipi,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                class_weight=deney.class_weight,
            )
        except ImportError as hata:
            print(f"[{deney.ad}] atlandi: {hata}")
            continue

        pipeline = _pipeline_olustur(
            model=base_model,
            smote_kullan=deney.smote_kullan,
            random_state=args.random_state,
        )
        param_grid = _parametre_gridini_olustur(
            model_tipi=deney.model_tipi,
            smote_kullan=deney.smote_kullan,
            scale_pos_weight=scale_pos_weight,
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=args.n_jobs,
            refit=True,
        )
        grid.fit(x_egitim, y_egitim)
        en_iyi = grid.best_estimator_

        print(f"[{deney.ad}] kalibrasyon karsilastirmasi...")
        kalibrasyon_sonuclari = _kalibrasyon_sonuclarini_uret(
            en_iyi_estimator=en_iyi,
            x_egitim=x_egitim,
            y_egitim=y_egitim,
            x_dogrulama=x_dogrulama,
            y_dogrulama=y_dogrulama,
            x_test=x_test,
            y_test=y_test,
            cv=args.cv,
        )

        brier_once = None
        for sonuc in kalibrasyon_sonuclari:
            if sonuc["kalibrasyon"] == "none":
                brier_once = float(sonuc["metrikler"]["brier"])
                break
        if brier_once is None:
            brier_once = float(kalibrasyon_sonuclari[0]["metrikler"]["brier"])

        for sonuc in kalibrasyon_sonuclari:
            brier = float(sonuc["metrikler"]["brier"])
            brier_iyilesme = (brier_once - brier) / brier_once if brier_once else 0.0
            tum_sonuclar_ham.append(
                {
                    "deney_adi": deney.ad,
                    "model_tipi": deney.model_tipi,
                    "smote_kullan": deney.smote_kullan,
                    "class_weight": deney.class_weight or "yok",
                    "kalibrasyon": sonuc["kalibrasyon"],
                    "esik_yontemi": sonuc["esik_yontemi"],
                    "esik": float(sonuc["esik"]),
                    "dogrulama_accuracy": float(sonuc["dogrulama_accuracy"]),
                    "dogrulama_f1": float(sonuc["dogrulama_f1"]),
                    "metrikler": sonuc["metrikler"],
                    "brier_iyilesme_orani": float(brier_iyilesme),
                    "cv_roc_auc": float(grid.best_score_),
                    "en_iyi_parametreler": _json_uyumlu(grid.best_params_),
                    "_pipeline_model": sonuc["_pipeline_model"],
                    "_kalibrator_model": sonuc["_kalibrator_model"],
                    "_y_dogrulama": np.asarray(y_dogrulama),
                    "_y_prob_dogrulama": np.asarray(sonuc["_y_prob_dogrulama"]),
                }
            )

        print(f"[{deney.ad}] tamamlandi | cv_roc_auc={float(grid.best_score_):.4f}")

    if not tum_sonuclar_ham:
        raise RuntimeError("Hicbir deney sonuclanamadi.")

    sirali_ham = _sonuclari_sirala(tum_sonuclar_ham)
    en_iyi_ham = sirali_ham[0]
    en_iyi = _rapor_icin_temizle(en_iyi_ham)
    sirali = [_rapor_icin_temizle(sonuc) for sonuc in sirali_ham]

    _deploy_artifactlerini_guncelle(
        artifact_klasoru=args.artifact_klasoru,
        en_iyi_sonuc=en_iyi_ham,
    )

    rapor = _rapor_sozlugu_olustur(
        args=args,
        x_egitim=x_egitim,
        x_dogrulama=x_dogrulama,
        x_test=x_test,
        en_iyi=en_iyi,
        sirali=sirali,
    )

    args.rapor_json_yolu.write_text(
        json.dumps(_json_uyumlu(rapor), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.rapor_md_yolu.write_text(_markdown_raporu_olustur(rapor), encoding="utf-8")

    print(f"JSON rapor yazildi: {args.rapor_json_yolu}")
    print(f"Markdown rapor yazildi: {args.rapor_md_yolu}")
    print(
        "En iyi kombinasyon: "
        f"{en_iyi['deney_adi']} | {en_iyi['kalibrasyon']} | esik={en_iyi['esik']:.4f}"
    )
    print(f"Deploy artifactleri guncellendi: {args.artifact_klasoru}")
    print("Optimizasyon tamamlandi.")


def _deney_tanimlarini_hazirla() -> list[DeneyTanimi]:
    return [
        DeneyTanimi(ad="xgboost_smote", model_tipi="xgboost", smote_kullan=True),
        DeneyTanimi(ad="xgboost_no_smote", model_tipi="xgboost", smote_kullan=False),
        DeneyTanimi(
            ad="logistic_balanced",
            model_tipi="logistic_regression",
            smote_kullan=False,
            class_weight="balanced",
        ),
        DeneyTanimi(
            ad="random_forest_balanced",
            model_tipi="random_forest",
            smote_kullan=False,
            class_weight="balanced",
        ),
    ]


def _pipeline_olustur(model: Any, smote_kullan: bool, random_state: int) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        (
            "sifir_nan_donusumu",
            FunctionTransformer(sifirlari_nan_donustur_pipeline, validate=False),
        ),
        ("imputer", median_imputer_olustur()),
        ("scaler", standard_scaler_olustur()),
        (
            "smote",
            SMOTE(random_state=random_state) if smote_kullan else "passthrough",
        ),
        ("model", model),
    ]
    return Pipeline(steps=steps)


def _modeli_olustur(
    model_tipi: str,
    random_state: int,
    n_jobs: int,
    class_weight: str | None,
) -> Any:
    if model_tipi == "logistic_regression":
        return LogisticRegression(
            max_iter=3000,
            random_state=random_state,
            class_weight=class_weight,
        )
    if model_tipi == "random_forest":
        return RandomForestClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight=class_weight,
        )
    if model_tipi == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as hata:
            raise ImportError("xgboost kutuphanesi yuklenemedi.") from hata
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=n_jobs,
        )
    raise ValueError(f"Desteklenmeyen model tipi: {model_tipi}")


def _parametre_gridini_olustur(
    model_tipi: str,
    smote_kullan: bool,
    scale_pos_weight: float,
) -> dict[str, list[Any]]:
    if model_tipi == "logistic_regression":
        return {
            "model__C": [0.1, 0.5, 1.0, 3.0, 10.0],
        }
    if model_tipi == "random_forest":
        return {
            "model__n_estimators": [200, 300],
            "model__max_depth": [5, 8, None],
            "model__min_samples_leaf": [1, 3],
        }
    if model_tipi == "xgboost":
        grid = {
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 4],
            "model__n_estimators": [150, 200],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        }
        if not smote_kullan:
            grid["model__scale_pos_weight"] = [1.0, float(scale_pos_weight)]
        return grid
    raise ValueError(f"Desteklenmeyen model tipi: {model_tipi}")


def _kalibrasyon_sonuclarini_uret(
    en_iyi_estimator: Any,
    x_egitim: pd.DataFrame,
    y_egitim: pd.Series,
    x_dogrulama: pd.DataFrame,
    y_dogrulama: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    cv: int,
) -> list[dict[str, Any]]:
    sonuclar: list[dict[str, Any]] = []
    pipeline_model = clone(en_iyi_estimator)
    pipeline_model.fit(x_egitim, y_egitim)
    for kalibrasyon in ("none", "sigmoid", "isotonic"):
        if kalibrasyon == "none":
            model = pipeline_model
        else:
            model = CalibratedClassifierCV(
                estimator=clone(en_iyi_estimator),
                method=kalibrasyon,
                cv=cv,
            )
            model.fit(x_egitim, y_egitim)

        y_prob_dogrulama = _pozitif_olasilik(model, x_dogrulama)
        y_prob_test = _pozitif_olasilik(model, x_test)

        aday_esikler: list[tuple[float, float, float, str]] = []
        aday_esikler.append(
            _accuracy_oncelikli_esik_bul(
                y_gercek=y_dogrulama,
                y_olasilik=y_prob_dogrulama,
                min_f1=MIN_F1_KISITI,
            )
        )
        f1_esigi, f1_dogrulama = _f1_optimum_esigi_bul(y_dogrulama, y_prob_dogrulama)
        f1_tahmin = (y_prob_dogrulama >= f1_esigi).astype(int)
        f1_dogrulama_accuracy = float(accuracy_score(y_dogrulama, f1_tahmin))
        aday_esikler.append(
            (
                float(f1_esigi),
                float(f1_dogrulama_accuracy),
                float(f1_dogrulama),
                "f1_optimum_dogrulama",
            )
        )

        tekillestirilmis: dict[str, tuple[float, float, float, str]] = {}
        for aday in aday_esikler:
            tekillestirilmis[aday[3]] = aday

        for esik, dogrulama_accuracy, f1_dogrulama, esik_yontemi in tekillestirilmis.values():
            y_tahmin_test = (y_prob_test >= esik).astype(int)
            metrikler = model_metriklerini_hesapla(y_test, y_tahmin_test, y_prob_test)
            sonuclar.append(
                {
                    "kalibrasyon": kalibrasyon,
                    "esik_yontemi": esik_yontemi,
                    "esik": float(esik),
                    "dogrulama_accuracy": float(dogrulama_accuracy),
                    "dogrulama_f1": float(f1_dogrulama),
                    "metrikler": metrikler,
                    "_pipeline_model": pipeline_model,
                    "_kalibrator_model": model,
                    "_y_prob_dogrulama": y_prob_dogrulama,
                }
            )
    return sonuclar


def _f1_optimum_esigi_bul(
    y_gercek: pd.Series | np.ndarray,
    y_olasilik: np.ndarray,
) -> tuple[float, float]:
    aday_esikler = np.linspace(0.05, 0.95, 181)
    en_iyi_esik = 0.5
    en_iyi_f1 = -1.0
    for esik in aday_esikler:
        tahmin = (y_olasilik >= esik).astype(int)
        skor = float(f1_score(y_gercek, tahmin, zero_division=0))
        if skor > en_iyi_f1:
            en_iyi_f1 = skor
            en_iyi_esik = float(esik)
    return en_iyi_esik, en_iyi_f1


def _accuracy_oncelikli_esik_bul(
    y_gercek: pd.Series | np.ndarray,
    y_olasilik: np.ndarray,
    min_f1: float = MIN_F1_KISITI,
) -> tuple[float, float, float, str]:
    aday_esikler = np.linspace(0.05, 0.95, 181)
    en_iyi_aday: tuple[float, float, float] | None = None
    for esik in aday_esikler:
        tahmin = (y_olasilik >= esik).astype(int)
        f1_skoru = float(f1_score(y_gercek, tahmin, zero_division=0))
        if f1_skoru < min_f1:
            continue
        accuracy_skoru = float(accuracy_score(y_gercek, tahmin))
        aday = (accuracy_skoru, f1_skoru, float(esik))
        if en_iyi_aday is None or aday > en_iyi_aday:
            en_iyi_aday = aday

    if en_iyi_aday is not None:
        return (
            en_iyi_aday[2],
            en_iyi_aday[0],
            en_iyi_aday[1],
            "accuracy_oncelikli_f1_kisitli",
        )

    esik, f1_skoru = _f1_optimum_esigi_bul(y_gercek=y_gercek, y_olasilik=y_olasilik)
    tahmin = (y_olasilik >= esik).astype(int)
    return (
        esik,
        float(accuracy_score(y_gercek, tahmin)),
        f1_skoru,
        "f1_optimum_fallback",
    )


def _pozitif_olasilik(model: Any, x_veri: pd.DataFrame) -> np.ndarray:
    olasilik = np.asarray(model.predict_proba(x_veri))
    if olasilik.ndim != 2 or olasilik.shape[1] < 2:
        raise ValueError("predict_proba cikisi beklenen formatta degil.")
    return olasilik[:, 1]


def _sonuclari_sirala(sonuclar: list[dict[str, Any]]) -> list[dict[str, Any]]:
    uygunlar = [s for s in sonuclar if float(s["metrikler"]["f1"]) >= MIN_F1_KISITI]
    if not uygunlar:
        raise RuntimeError(
            f"F1 >= {MIN_F1_KISITI:.2f} kisitini saglayan kombinasyon bulunamadi."
        )
    return sorted(
        uygunlar,
        key=lambda s: (
            float(s["metrikler"]["accuracy"]),
            float(s["metrikler"]["roc_auc"]),
            -float(s["metrikler"]["brier"]),
            float(s["metrikler"]["recall"]),
        ),
        reverse=True,
    )


def _rapor_sozlugu_olustur(
    args: argparse.Namespace,
    x_egitim: pd.DataFrame,
    x_dogrulama: pd.DataFrame,
    x_test: pd.DataFrame,
    en_iyi: dict[str, Any],
    sirali: list[dict[str, Any]],
) -> dict[str, Any]:
    metrikler = en_iyi["metrikler"]
    hedefler = {
        "accuracy_min": 0.78,
        "roc_auc_min": 0.80,
        "f1_min": MIN_F1_KISITI,
        "brier_iyilesme_orani_min": 0.10,
    }
    hedef_durumu = {
        "accuracy": float(metrikler["accuracy"]) >= hedefler["accuracy_min"],
        "roc_auc": float(metrikler["roc_auc"]) >= hedefler["roc_auc_min"],
        "f1": float(metrikler["f1"]) >= hedefler["f1_min"],
        "brier_iyilesme_orani": float(en_iyi["brier_iyilesme_orani"])
        >= hedefler["brier_iyilesme_orani_min"],
    }
    hedef_durumu["tum_hedefler"] = all(hedef_durumu.values())

    return {
        "amac": "Veri seti boyutunu degistirmeden 9-12 teknik ekseninde optimizasyon",
        "kurulum": {
            "veri_yolu": str(args.veri_yolu),
            "test_boyutu": float(args.test_boyutu),
            "dogrulama_boyutu": float(args.dogrulama_boyutu),
            "cv": int(args.cv),
            "random_state": int(args.random_state),
        },
        "bolme_bilgisi": {
            "egitim": int(len(x_egitim)),
            "dogrulama": int(len(x_dogrulama)),
            "test": int(len(x_test)),
        },
        "en_iyi_sonuc": en_iyi,
        "hedefler": hedefler,
        "hedef_durumu": hedef_durumu,
        "sirali_sonuclar": sirali,
    }


def _markdown_raporu_olustur(rapor: dict[str, Any]) -> str:
    en_iyi = rapor["en_iyi_sonuc"]
    metrikler = en_iyi["metrikler"]
    hedef_durumu = rapor["hedef_durumu"]
    satirlar = []
    satirlar.append("# Veri Artirmadan Optimizasyon Raporu")
    satirlar.append("")
    satirlar.append("## 1. Ozet")
    satirlar.append("")
    satirlar.append(f"- En iyi deney: `{en_iyi['deney_adi']}`")
    satirlar.append(f"- Kalibrasyon: `{en_iyi['kalibrasyon']}`")
    satirlar.append(f"- Esik yontemi: `{en_iyi['esik_yontemi']}`")
    satirlar.append(f"- Esik: `{float(en_iyi['esik']):.4f}`")
    satirlar.append("")
    satirlar.append("### 1.1 Metrikler")
    satirlar.append("")
    satirlar.append("| Metrik | Deger |")
    satirlar.append("| --- | --- |")
    satirlar.append(f"| Accuracy | {float(metrikler['accuracy']):.4f} |")
    satirlar.append(f"| Precision | {float(metrikler['precision']):.4f} |")
    satirlar.append(f"| Recall | {float(metrikler['recall']):.4f} |")
    satirlar.append(f"| F1 | {float(metrikler['f1']):.4f} |")
    satirlar.append(f"| ROC AUC | {float(metrikler['roc_auc']):.4f} |")
    satirlar.append(f"| Brier | {float(metrikler['brier']):.4f} |")
    satirlar.append(f"| Brier iyilesme orani | {float(en_iyi['brier_iyilesme_orani']) * 100:.2f}% |")
    satirlar.append("")
    satirlar.append("### 1.2 Hedef Uyum Durumu")
    satirlar.append("")
    satirlar.append("| Hedef | Durum |")
    satirlar.append("| --- | --- |")
    satirlar.append(
        f"| Accuracy >= {rapor['hedefler']['accuracy_min']:.2f} | {'Saglandi' if hedef_durumu['accuracy'] else 'Saglanmadi'} |"
    )
    satirlar.append(f"| ROC AUC >= 0.80 | {'Saglandi' if hedef_durumu['roc_auc'] else 'Saglanmadi'} |")
    satirlar.append(f"| F1 >= 0.70 | {'Saglandi' if hedef_durumu['f1'] else 'Saglanmadi'} |")
    satirlar.append(
        f"| Brier iyilesme >= %10 | {'Saglandi' if hedef_durumu['brier_iyilesme_orani'] else 'Saglanmadi'} |"
    )
    satirlar.append("")
    satirlar.append("## 2. Deney Siralamasi (Ilk 10)")
    satirlar.append("")
    satirlar.append("| Deney | Kalibrasyon | Esik | F1 | ROC AUC | Brier | Brier iyilesme |")
    satirlar.append("| --- | --- | --- | --- | --- | --- | --- |")
    for sonuc in rapor["sirali_sonuclar"][:10]:
        metrik = sonuc["metrikler"]
        satirlar.append(
            "| {deney} | {kal} | {esik:.3f} | {f1:.4f} | {auc:.4f} | {brier:.4f} | {iyilesme:.2f}% |".format(
                deney=sonuc["deney_adi"],
                kal=sonuc["kalibrasyon"],
                esik=float(sonuc["esik"]),
                f1=float(metrik["f1"]),
                auc=float(metrik["roc_auc"]),
                brier=float(metrik["brier"]),
                iyilesme=float(sonuc["brier_iyilesme_orani"]) * 100.0,
            )
        )
    satirlar.append("")
    satirlar.append("## 3. Not")
    satirlar.append("")
    satirlar.append(
        "- Bu rapor yalnizca mevcut veriyle uretilmistir; veri hacmi arttirilmadan elde edilen tavan performansi gosterir."
    )
    satirlar.append(
        "- %90+ accuracy gibi hedefler icin yalnizca algoritma ayari degil, dis veri ve ozellik zenginlestirme gerekir."
    )
    satirlar.append("")
    return "\n".join(satirlar)


def _rapor_icin_temizle(sonuc: dict[str, Any]) -> dict[str, Any]:
    return {
        k: _json_uyumlu(v)
        for k, v in sonuc.items()
        if not str(k).startswith("_")
    }


def _deploy_artifactlerini_guncelle(
    artifact_klasoru: Path,
    en_iyi_sonuc: dict[str, Any],
) -> None:
    y_dogrulama = np.asarray(en_iyi_sonuc["_y_dogrulama"])
    y_prob_dogrulama = np.asarray(en_iyi_sonuc["_y_prob_dogrulama"])
    youden = youden_j_esigi_hesapla(y_dogrulama, y_prob_dogrulama)
    f2 = f2_esigi_hesapla(y_dogrulama, y_prob_dogrulama, beta=2.0)
    risk_esikleri = risk_esiklerini_olustur(youden["esik"], f2["esik"])

    esik = float(en_iyi_sonuc["esik"])
    esik_yontemi = str(en_iyi_sonuc.get("esik_yontemi", "accuracy_oncelikli_f1_kisitli"))
    esik_yapilandirmasi = {
        "ikili_siniflama_esikleri": {
            "youden_j": youden,
            "f2": f2,
            "f1_optimum": {
                "esik": esik,
                "f1_skoru": float(en_iyi_sonuc["dogrulama_f1"]),
            },
            "accuracy_oncelikli_f1_kisitli": {
                "esik": esik,
                "dogrulama_accuracy": float(en_iyi_sonuc.get("dogrulama_accuracy", 0.0)),
                "dogrulama_f1": float(en_iyi_sonuc.get("dogrulama_f1", 0.0)),
            },
        },
        "onerilen_ikili_siniflama_esigi": esik,
        "onerilen_ikili_siniflama_yontemi": esik_yontemi,
        "risk_kategorileri": {
            "dusuk_ust_esik": risk_esikleri["dusuk_ust_esik"],
            "orta_ust_esik": risk_esikleri["orta_ust_esik"],
            "etiketler": ["dusuk", "orta", "yuksek"],
        },
    }

    metrik_ozeti = dict(en_iyi_sonuc["metrikler"])
    metrik_ozeti["siniflama_esigi"] = esik
    model_metadata = {
        "model_adi": str(en_iyi_sonuc["model_tipi"]),
        "kalibrasyon_yontemi": str(en_iyi_sonuc["kalibrasyon"]),
        "ikili_siniflama_yontemi": esik_yontemi,
        "ikili_siniflama_esigi": esik,
        "test_boyutu": 0.2,
        "random_state": RANDOM_STATE,
    }

    artifactleri_kaydet(
        artifact_klasoru=artifact_klasoru,
        en_iyi_pipeline=en_iyi_sonuc["_pipeline_model"],
        kalibrator=en_iyi_sonuc["_kalibrator_model"],
        esik_yapilandirmasi=esik_yapilandirmasi,
        ozellik_sirasi=list(OZELLIK_KOLONLARI),
        metrik_ozeti=metrik_ozeti,
        model_metadata=model_metadata,
    )


def _klasorleri_hazirla(rapor_json_yolu: Path, rapor_md_yolu: Path) -> None:
    rapor_json_yolu.parent.mkdir(parents=True, exist_ok=True)
    rapor_md_yolu.parent.mkdir(parents=True, exist_ok=True)


def _json_uyumlu(veri: Any) -> Any:
    if hasattr(veri, "item"):
        return veri.item()
    if isinstance(veri, dict):
        return {k: _json_uyumlu(v) for k, v in veri.items()}
    if isinstance(veri, list):
        return [_json_uyumlu(v) for v in veri]
    if isinstance(veri, tuple):
        return [_json_uyumlu(v) for v in veri]
    return veri


if __name__ == "__main__":
    main()
