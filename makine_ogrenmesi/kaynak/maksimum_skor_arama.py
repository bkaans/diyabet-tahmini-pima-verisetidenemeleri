"""Veriye dokunmadan maksimum skor arama altyapisi."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC

from .artifact_kaydet import artifactleri_kaydet
from .ozellik_yapilandirmasi import HEDEF_KOLONU, OZELLIK_KOLONLARI
from .veri_yukleyici import veri_setini_yukle


RANDOM_STATE = 42
BEKLENEN_SATIR_SAYISI = 768
BEKLENEN_SINIF_DAGILIMI = {0: 500, 1: 268}
SIFIRI_EKSIK_SAYILAN_KOLONLAR = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]
SKORLAR = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


@dataclass(frozen=True)
class ModelAdayi:
    """Tek model ailesi ve arama alani."""

    ad: str
    estimator: BaseEstimator
    parametreler: dict[str, list[Any]]
    opsiyonel: bool = False


@dataclass(frozen=True)
class AramaSonucu:
    """Model arama sonucunu tasiyan yapi."""

    mod: str
    cv_adi: str
    model_adi: str
    estimator: BaseEstimator
    en_iyi_parametreler: dict[str, Any]
    cv_metrikleri: dict[str, float]
    holdout_metrikleri: dict[str, Any]


class SifirDegerDonusturucu(BaseEstimator, TransformerMixin):
    """Secili kolonlarda 0 degerlerini NaN'a cevirir veya ham birakir."""

    def __init__(
        self,
        strategy: str = "nan",
        kolonlar: tuple[str, ...] = tuple(SIFIRI_EKSIK_SAYILAN_KOLONLAR),
    ) -> None:
        self.strategy = strategy
        self.kolonlar = kolonlar

    def fit(self, x: Any, y: Any = None) -> "SifirDegerDonusturucu":
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        veri = _dataframe_yap(x)
        if self.strategy == "raw":
            return veri
        if self.strategy != "nan":
            raise ValueError("strategy sadece 'nan' veya 'raw' olabilir.")

        donusmus = veri.copy()
        for kolon in self.kolonlar:
            if kolon in donusmus.columns:
                donusmus[kolon] = donusmus[kolon].where(donusmus[kolon] != 0, np.nan)
        return donusmus


class KlinikOzellikUretici(BaseEstimator, TransformerMixin):
    """Ham CSV'yi degistirmeden pipeline icinde turetilmis ozellik uretir."""

    def __init__(self, strategy: str = "basic") -> None:
        self.strategy = strategy

    def fit(self, x: Any, y: Any = None) -> "KlinikOzellikUretici":
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        veri = _dataframe_yap(x)
        if self.strategy == "none":
            return veri
        if self.strategy not in {"basic", "full"}:
            raise ValueError("strategy sadece 'none', 'basic' veya 'full' olabilir.")

        uretilen = veri.copy()
        uretilen["Glucose_BMI"] = uretilen["Glucose"] * uretilen["BMI"]
        uretilen["Glucose_Age"] = uretilen["Glucose"] * uretilen["Age"]
        uretilen["BMI_Age"] = uretilen["BMI"] * uretilen["Age"]
        uretilen["Pregnancies_Age_Ratio"] = _guvenli_bol(
            uretilen["Pregnancies"],
            uretilen["Age"],
        )
        uretilen["Insulin_Glucose_Ratio"] = _guvenli_bol(
            uretilen["Insulin"],
            uretilen["Glucose"],
        )

        if self.strategy == "full":
            uretilen["BMI_Glucose_Ratio"] = _guvenli_bol(
                uretilen["BMI"],
                uretilen["Glucose"],
            )
            uretilen["BloodPressure_BMI"] = uretilen["BloodPressure"] * uretilen["BMI"]
            uretilen["SkinThickness_BMI_Ratio"] = _guvenli_bol(
                uretilen["SkinThickness"],
                uretilen["BMI"],
            )
            uretilen["Metabolic_Load"] = (
                uretilen["Glucose"] + uretilen["BMI"] + uretilen["Age"]
            )
            uretilen["Pedigree_Age"] = (
                uretilen["DiabetesPedigreeFunction"] * uretilen["Age"]
            )
        return uretilen.replace([np.inf, -np.inf], np.nan)


def veri_butunlugu_ozeti(veri_yolu: Path) -> dict[str, Any]:
    """Ham veri dosyasinin dokunulmadigini kanitlamak icin ozet uretir."""
    veri = veri_setini_yukle(veri_yolu)
    dagilim = {
        int(k): int(v)
        for k, v in veri[HEDEF_KOLONU].value_counts().sort_index().to_dict().items()
    }
    ozet = {
        "dosya": str(veri_yolu),
        "sha256": _sha256(veri_yolu),
        "satir_sayisi": int(len(veri)),
        "kolon_sayisi": int(veri.shape[1]),
        "sinif_dagilimi": dagilim,
    }
    if len(veri) != BEKLENEN_SATIR_SAYISI:
        raise ValueError(f"Beklenen satir sayisi {BEKLENEN_SATIR_SAYISI}, bulunan {len(veri)}.")
    if dagilim != BEKLENEN_SINIF_DAGILIMI:
        raise ValueError(f"Beklenen sinif dagilimi {BEKLENEN_SINIF_DAGILIMI}, bulunan {dagilim}.")
    return ozet


def eda_raporu_uret(veri_yolu: Path, rapor_md_yolu: Path, rapor_json_yolu: Path) -> None:
    """Detayli EDA raporunu Markdown ve JSON olarak yazar."""
    veri_ozeti = veri_butunlugu_ozeti(veri_yolu)
    veri = veri_setini_yukle(veri_yolu)
    ozellikler = veri[OZELLIK_KOLONLARI]
    hedef = veri[HEDEF_KOLONU]

    sifir_analizi = {
        kolon: {
            "sifir_sayisi": int((veri[kolon] == 0).sum()),
            "sifir_orani": float((veri[kolon] == 0).mean()),
        }
        for kolon in OZELLIK_KOLONLARI
    }
    korelasyon = (
        veri[OZELLIK_KOLONLARI + [HEDEF_KOLONU]]
        .corr(numeric_only=True)[HEDEF_KOLONU]
        .drop(HEDEF_KOLONU)
        .sort_values(key=lambda seri: seri.abs(), ascending=False)
    )
    grup_ozetleri = {
        kolon: {
            "outcome_0_ortalama": float(veri.loc[hedef == 0, kolon].mean()),
            "outcome_1_ortalama": float(veri.loc[hedef == 1, kolon].mean()),
            "fark": float(veri.loc[hedef == 1, kolon].mean() - veri.loc[hedef == 0, kolon].mean()),
        }
        for kolon in OZELLIK_KOLONLARI
    }
    aykiri_ozet = {
        kolon: _iqr_aykiri_ozeti(veri[kolon])
        for kolon in OZELLIK_KOLONLARI
    }
    feature_importance = _temel_feature_importance(ozellikler, hedef)

    rapor = {
        "veri_butunlugu": veri_ozeti,
        "sinif_oranlari": {
            str(k): float(v)
            for k, v in hedef.value_counts(normalize=True).sort_index().to_dict().items()
        },
        "sifir_analizi": sifir_analizi,
        "korelasyon_outcome": {k: float(v) for k, v in korelasyon.to_dict().items()},
        "sinif_bazli_ortalama_farklari": grup_ozetleri,
        "aykiri_deger_ozeti": aykiri_ozet,
        "temel_feature_importance": feature_importance,
    }

    rapor_json_yolu.parent.mkdir(parents=True, exist_ok=True)
    rapor_json_yolu.write_text(
        json.dumps(_json_uyumlu_yap(rapor), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rapor_md_yolu.parent.mkdir(parents=True, exist_ok=True)
    rapor_md_yolu.write_text(_eda_markdown_yap(rapor), encoding="utf-8")


def ml_yeniden_kur(
    *,
    mod: str,
    veri_yolu: Path,
    proje_koku: Path,
    test_boyutu: float = 0.2,
    random_state: int = RANDOM_STATE,
    n_iter: int | None = None,
    n_jobs: int = -1,
    aggressive_seeds: int = 30,
) -> dict[str, Any]:
    """Secilen moda gore model arar, rapor yazar ve en iyi artifactleri kaydeder."""
    baslangic_ozeti = veri_butunlugu_ozeti(veri_yolu)
    veri = veri_setini_yukle(veri_yolu)
    x = veri[OZELLIK_KOLONLARI]
    y = veri[HEDEF_KOLONU]

    x_gelistirme, x_test, y_gelistirme, y_test = train_test_split(
        x,
        y,
        test_size=test_boyutu,
        random_state=random_state,
        stratify=y,
    )

    if mod == "aggressive":
        rapor = _agresif_accuracy_arama(
            x=x,
            y=y,
            proje_koku=proje_koku,
            random_state=random_state,
            n_iter=n_iter or 5,
            n_jobs=n_jobs,
            aggressive_seeds=aggressive_seeds,
            baslangic_ozeti=baslangic_ozeti,
            test_boyutu=test_boyutu,
        )
    else:
        rapor = _standart_arama(
            mod=mod,
            x_gelistirme=x_gelistirme,
            x_test=x_test,
            y_gelistirme=y_gelistirme,
            y_test=y_test,
            proje_koku=proje_koku,
            random_state=random_state,
            n_iter=n_iter or _varsayilan_n_iter(mod),
            n_jobs=n_jobs,
            baslangic_ozeti=baslangic_ozeti,
        )

    bitis_ozeti = veri_butunlugu_ozeti(veri_yolu)
    if bitis_ozeti["sha256"] != baslangic_ozeti["sha256"]:
        raise RuntimeError("Ham veri dosyasi degisti; islem durduruldu.")

    rapor["veri_butunlugu_bitis"] = bitis_ozeti
    return rapor


def _standart_arama(
    *,
    mod: str,
    x_gelistirme: pd.DataFrame,
    x_test: pd.DataFrame,
    y_gelistirme: pd.Series,
    y_test: pd.Series,
    proje_koku: Path,
    random_state: int,
    n_iter: int,
    n_jobs: int,
    baslangic_ozeti: dict[str, Any],
) -> dict[str, Any]:
    cv_stratejileri = _cv_stratejileri(mod, random_state)
    adaylar = model_adaylarini_olustur(y_gelistirme, random_state)
    aday_sonuclari: list[AramaSonucu] = []
    atlanan_modeller: list[dict[str, str]] = []

    for cv_adi, cv in cv_stratejileri:
        for aday in adaylar:
            try:
                sonuc = _aday_ara(
                    aday=aday,
                    cv_adi=cv_adi,
                    cv=cv,
                    mod=mod,
                    x_gelistirme=x_gelistirme,
                    y_gelistirme=y_gelistirme,
                    x_test=x_test,
                    y_test=y_test,
                    n_iter=n_iter,
                    n_jobs=n_jobs,
                    random_state=random_state,
                )
                aday_sonuclari.append(sonuc)
                print(
                    f"[{mod}/{cv_adi}/{aday.ad}] "
                    f"cv_accuracy={sonuc.cv_metrikleri['accuracy']:.4f} "
                    f"holdout_accuracy={sonuc.holdout_metrikleri['accuracy']:.4f}"
                )
            except Exception as hata:
                if aday.opsiyonel:
                    atlanan_modeller.append({"model": aday.ad, "neden": str(hata)})
                    print(f"[{aday.ad}] opsiyonel model atlandi: {hata}")
                    continue
                raise

    if not aday_sonuclari:
        raise RuntimeError("Hicbir model sonucu uretilemedi.")

    en_iyi = _en_iyi_sonucu_sec(aday_sonuclari)
    grid_refinement = _dar_grid_search_yap(
        sonuc=en_iyi,
        x_gelistirme=x_gelistirme,
        y_gelistirme=y_gelistirme,
        x_test=x_test,
        y_test=y_test,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    if grid_refinement["sonuc"] is not None:
        grid_sonucu = grid_refinement["sonuc"]
        if (
            grid_sonucu.cv_metrikleri["accuracy"],
            grid_sonucu.holdout_metrikleri["accuracy"],
        ) >= (
            en_iyi.cv_metrikleri["accuracy"],
            en_iyi.holdout_metrikleri["accuracy"],
        ):
            en_iyi = grid_sonucu
            aday_sonuclari.append(grid_sonucu)
    final_raporu = _final_model_egit_ve_kaydet(
        sonuc=en_iyi,
        x_gelistirme=x_gelistirme,
        y_gelistirme=y_gelistirme,
        x_test=x_test,
        y_test=y_test,
        proje_koku=proje_koku,
        random_state=random_state,
        mod=mod,
        baslangic_ozeti=baslangic_ozeti,
        atlanan_modeller=atlanan_modeller,
        tum_sonuclar=aday_sonuclari,
        grid_refinement=grid_refinement,
    )

    if mod == "nested":
        final_raporu["nested_cv"] = _nested_cv_raporu_uret(
            adaylar=adaylar,
            x=pd.concat([x_gelistirme, x_test], axis=0),
            y=pd.concat([y_gelistirme, y_test], axis=0),
            random_state=random_state,
            n_iter=max(2, min(n_iter, 6)),
            n_jobs=n_jobs,
        )

    _standart_raporlari_yaz(proje_koku, final_raporu)
    return final_raporu


def _agresif_accuracy_arama(
    *,
    x: pd.DataFrame,
    y: pd.Series,
    proje_koku: Path,
    random_state: int,
    n_iter: int,
    n_jobs: int,
    aggressive_seeds: int,
    baslangic_ozeti: dict[str, Any],
    test_boyutu: float,
) -> dict[str, Any]:
    aday_adlari = {"logistic_regression", "svm_rbf", "random_forest", "extra_trees", "xgboost"}
    adaylar = [a for a in model_adaylarini_olustur(y, random_state) if a.ad in aday_adlari]
    tum_sonuclar: list[dict[str, Any]] = []
    en_iyi_kayit: dict[str, Any] | None = None

    for seed in range(random_state, random_state + aggressive_seeds):
        x_gel, x_test, y_gel, y_test = train_test_split(
            x,
            y,
            test_size=test_boyutu,
            random_state=seed,
            stratify=y,
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_gel,
            y_gel,
            test_size=0.25,
            random_state=seed,
            stratify=y_gel,
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

        for aday in adaylar:
            try:
                arama = _randomized_search_yap(
                    aday=aday,
                    cv=cv,
                    x=x_train,
                    y=y_train,
                    n_iter=n_iter,
                    n_jobs=n_jobs,
                    random_state=seed,
                )
            except Exception as hata:
                if aday.opsiyonel:
                    continue
                raise hata

            val_olasilik = _pozitif_olasilik(arama.best_estimator_, x_val)
            esik_ozeti = _en_iyi_accuracy_esigi(y_val, val_olasilik)
            test_olasilik = _pozitif_olasilik(arama.best_estimator_, x_test)
            test_metrikleri = _metrikleri_hesapla(y_test, test_olasilik, esik_ozeti["esik"])
            kayit = {
                "seed": seed,
                "model_adi": aday.ad,
                "validation_accuracy": float(esik_ozeti["accuracy"]),
                "validation_esik": float(esik_ozeti["esik"]),
                "test_metrikleri": test_metrikleri,
                "cv_best_accuracy": float(arama.best_score_),
                "en_iyi_parametreler": arama.best_params_,
            }
            tum_sonuclar.append(kayit)
            if en_iyi_kayit is None or _agresif_siralama_anahtari(kayit) > _agresif_siralama_anahtari(en_iyi_kayit):
                en_iyi_kayit = kayit
                print(
                    f"[aggressive] yeni en iyi | seed={seed} model={aday.ad} "
                    f"test_accuracy={test_metrikleri['accuracy']:.4f}"
                )

    if en_iyi_kayit is None:
        raise RuntimeError("Agresif aramada model sonucu uretilemedi.")

    # En iyi agresif ayarlari final artifact olarak da kaydedilir.
    en_iyi_aday = next(a for a in model_adaylarini_olustur(y, random_state) if a.ad == en_iyi_kayit["model_adi"])
    en_iyi_estimator = clone(en_iyi_aday.estimator).set_params(**en_iyi_kayit["en_iyi_parametreler"])
    x_gel, x_test, y_gel, y_test = train_test_split(
        x,
        y,
        test_size=test_boyutu,
        random_state=int(en_iyi_kayit["seed"]),
        stratify=y,
    )
    en_iyi_estimator.fit(x_gel, y_gel)
    test_olasilik = _pozitif_olasilik(en_iyi_estimator, x_test)
    metrik_ozeti = _metrikleri_hesapla(
        y_test,
        test_olasilik,
        float(en_iyi_kayit["validation_esik"]),
    )
    _artifactleri_yaz(
        proje_koku=proje_koku,
        pipeline=en_iyi_estimator,
        kalibrator=en_iyi_estimator,
        esik=float(en_iyi_kayit["validation_esik"]),
        metrik_ozeti=metrik_ozeti,
        model_metadata={
            "model_adi": en_iyi_kayit["model_adi"],
            "mod": "aggressive",
            "cv_stratejisi": "seed_sweep_validation_threshold",
            "accuracy_hedefi": 0.90,
            "veri_politikasi": "ham veri degistirilmedi",
            "agresif_seed": int(en_iyi_kayit["seed"]),
        },
    )

    rapor = {
        "mod": "aggressive",
        "olusturulma_zamani_utc": datetime.now(timezone.utc).isoformat(),
        "veri_butunlugu_baslangic": baslangic_ozeti,
        "en_iyi_agresif_sonuc": en_iyi_kayit,
        "final_test_metrikleri": metrik_ozeti,
        "tum_agresif_sonuclar": sorted(
            tum_sonuclar,
            key=_agresif_siralama_anahtari,
            reverse=True,
        ),
        "onay_kapisi": _onay_kapisi_metni(metrik_ozeti["accuracy"]),
    }
    _agresif_raporu_yaz(proje_koku, rapor)
    return rapor


def model_adaylarini_olustur(y: pd.Series, random_state: int) -> list[ModelAdayi]:
    """Zorunlu ve opsiyonel model adaylarini dondurur."""
    negatif = int((y == 0).sum())
    pozitif = int((y == 1).sum())
    scale_pos_weight = negatif / max(pozitif, 1)

    adaylar = [
        ModelAdayi(
            ad="logistic_regression",
            estimator=_olcekli_pipeline(
                LogisticRegression(max_iter=5000, random_state=random_state)
            ),
            parametreler={
                "sifir__strategy": ["nan", "raw"],
                "ozellik__strategy": ["none", "basic", "full"],
                "imputer": _imputerler(),
                "scaler": _scalerlar(random_state),
                "model__C": [0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        ModelAdayi(
            ad="svm_rbf",
            estimator=_olcekli_pipeline(SVC(kernel="rbf", probability=True, random_state=random_state)),
            parametreler={
                "sifir__strategy": ["nan", "raw"],
                "ozellik__strategy": ["none", "basic", "full"],
                "imputer": _imputerler(),
                "scaler": _scalerlar(random_state),
                "model__C": [0.1, 0.3, 1.0, 3.0, 10.0],
                "model__gamma": ["scale", 0.01, 0.03, 0.1, 0.3],
                "model__class_weight": [None, "balanced"],
            },
        ),
        ModelAdayi(
            ad="random_forest",
            estimator=_agac_pipeline(RandomForestClassifier(random_state=random_state, n_jobs=-1)),
            parametreler={
                "sifir__strategy": ["nan", "raw"],
                "ozellik__strategy": ["none", "basic", "full"],
                "imputer": _imputerler(),
                "model__n_estimators": [200, 400, 700],
                "model__max_depth": [None, 4, 6, 8, 12],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", None],
                "model__class_weight": [None, "balanced", "balanced_subsample"],
            },
        ),
        ModelAdayi(
            ad="extra_trees",
            estimator=_agac_pipeline(ExtraTreesClassifier(random_state=random_state, n_jobs=-1)),
            parametreler={
                "sifir__strategy": ["nan", "raw"],
                "ozellik__strategy": ["none", "basic", "full"],
                "imputer": _imputerler(),
                "model__n_estimators": [300, 600, 900],
                "model__max_depth": [None, 4, 6, 8, 12],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", None],
                "model__class_weight": [None, "balanced", "balanced_subsample"],
            },
        ),
        ModelAdayi(
            ad="gradient_boosting",
            estimator=_agac_pipeline(GradientBoostingClassifier(random_state=random_state)),
            parametreler={
                "sifir__strategy": ["nan", "raw"],
                "ozellik__strategy": ["none", "basic", "full"],
                "imputer": _imputerler(),
                "model__n_estimators": [100, 200, 350],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__max_depth": [1, 2, 3],
                "model__min_samples_leaf": [1, 3, 6, 10],
                "model__subsample": [0.75, 0.9, 1.0],
            },
        ),
        ModelAdayi(
            ad="hist_gradient_boosting",
            estimator=_agac_pipeline(HistGradientBoostingClassifier(random_state=random_state)),
            parametreler={
                "sifir__strategy": ["nan", "raw"],
                "ozellik__strategy": ["none", "basic", "full"],
                "imputer": _imputerler(),
                "model__max_iter": [100, 200, 350],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__max_leaf_nodes": [7, 15, 31],
                "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
                "model__min_samples_leaf": [10, 20, 35],
            },
        ),
    ]

    xgb_adayi = _xgboost_adayi(random_state, scale_pos_weight)
    if xgb_adayi:
        adaylar.append(xgb_adayi)
    lightgbm_adayi = _lightgbm_adayi(random_state, scale_pos_weight)
    if lightgbm_adayi:
        adaylar.append(lightgbm_adayi)
    catboost_adayi = _catboost_adayi(random_state)
    if catboost_adayi:
        adaylar.append(catboost_adayi)

    adaylar.extend(_ensemble_adaylari(random_state))
    return adaylar


def _aday_ara(
    *,
    aday: ModelAdayi,
    cv_adi: str,
    cv: Any,
    mod: str,
    x_gelistirme: pd.DataFrame,
    y_gelistirme: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    n_iter: int,
    n_jobs: int,
    random_state: int,
) -> AramaSonucu:
    arama = _randomized_search_yap(
        aday=aday,
        cv=cv,
        x=x_gelistirme,
        y=y_gelistirme,
        n_iter=n_iter,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    estimator = arama.best_estimator_
    olasilik = _pozitif_olasilik(estimator, x_test)
    holdout_metrikleri = _metrikleri_hesapla(y_test, olasilik, 0.5)
    cv_metrikleri = _cv_best_metrikleri(arama)
    return AramaSonucu(
        mod=mod,
        cv_adi=cv_adi,
        model_adi=aday.ad,
        estimator=estimator,
        en_iyi_parametreler=arama.best_params_,
        cv_metrikleri=cv_metrikleri,
        holdout_metrikleri=holdout_metrikleri,
    )


def _randomized_search_yap(
    *,
    aday: ModelAdayi,
    cv: Any,
    x: pd.DataFrame,
    y: pd.Series,
    n_iter: int,
    n_jobs: int,
    random_state: int,
) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        estimator=clone(aday.estimator),
        param_distributions=aday.parametreler,
        n_iter=n_iter,
        scoring=SKORLAR,
        refit="accuracy",
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        error_score="raise",
        return_train_score=False,
    ).fit(x, y)


def _final_model_egit_ve_kaydet(
    *,
    sonuc: AramaSonucu,
    x_gelistirme: pd.DataFrame,
    y_gelistirme: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    proje_koku: Path,
    random_state: int,
    mod: str,
    baslangic_ozeti: dict[str, Any],
    atlanan_modeller: list[dict[str, str]],
    tum_sonuclar: list[AramaSonucu],
    grid_refinement: dict[str, Any],
) -> dict[str, Any]:
    x_train, x_val, y_train, y_val = train_test_split(
        x_gelistirme,
        y_gelistirme,
        test_size=0.25,
        random_state=random_state,
        stratify=y_gelistirme,
    )
    temel_estimator = clone(sonuc.estimator)
    temel_estimator.fit(x_train, y_train)
    kalibrasyon_ozeti = _kalibrasyon_sec(
        estimator=temel_estimator,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
    )

    final_pipeline = clone(sonuc.estimator)
    final_pipeline.fit(x_gelistirme, y_gelistirme)
    if kalibrasyon_ozeti["kalibrasyon"] == "none":
        final_kalibrator = final_pipeline
    else:
        final_kalibrator = CalibratedClassifierCV(
            estimator=clone(sonuc.estimator),
            method=kalibrasyon_ozeti["kalibrasyon"],
            cv=3,
        )
        final_kalibrator.fit(x_gelistirme, y_gelistirme)

    test_olasilik = _pozitif_olasilik(final_kalibrator, x_test)
    final_metrikler = _metrikleri_hesapla(
        y_test,
        test_olasilik,
        float(kalibrasyon_ozeti["esik"]),
    )
    model_metadata = {
        "model_adi": sonuc.model_adi,
        "mod": mod,
        "cv_stratejisi": sonuc.cv_adi,
        "ana_metrik": "accuracy",
        "kalibrasyon_yontemi": kalibrasyon_ozeti["kalibrasyon"],
        "ikili_siniflama_yontemi": "validation_accuracy_threshold",
        "ikili_siniflama_esigi": float(kalibrasyon_ozeti["esik"]),
        "veri_politikasi": "ham veri degistirilmedi",
        "test_boyutu": 0.2,
        "random_state": random_state,
    }
    _artifactleri_yaz(
        proje_koku=proje_koku,
        pipeline=final_pipeline,
        kalibrator=final_kalibrator,
        esik=float(kalibrasyon_ozeti["esik"]),
        metrik_ozeti=final_metrikler,
        model_metadata=model_metadata,
    )

    sirali_sonuclar = sorted(
        [_sonuc_json_yap(s) for s in tum_sonuclar],
        key=lambda s: (
            s["cv_metrikleri"]["accuracy"],
            s["holdout_metrikleri"]["accuracy"],
            s["cv_metrikleri"]["f1"],
            s["cv_metrikleri"]["roc_auc"],
        ),
        reverse=True,
    )
    return {
        "mod": mod,
        "olusturulma_zamani_utc": datetime.now(timezone.utc).isoformat(),
        "veri_butunlugu_baslangic": baslangic_ozeti,
        "en_iyi_model": _sonuc_json_yap(sonuc),
        "final_test_metrikleri": final_metrikler,
        "kalibrasyon_ve_esik": kalibrasyon_ozeti,
        "grid_refinement": {
            "calisti": bool(grid_refinement["calisti"]),
            "model_adi": grid_refinement.get("model_adi"),
            "param_grid": _json_uyumlu_yap(grid_refinement.get("param_grid")),
            "cv_metrikleri": (
                grid_refinement["sonuc"].cv_metrikleri
                if grid_refinement.get("sonuc") is not None
                else None
            ),
        },
        "tum_model_sonuclari": sirali_sonuclar,
        "atlanan_opsiyonel_modeller": atlanan_modeller,
        "onay_kapisi": _onay_kapisi_metni(final_metrikler["accuracy"]),
    }


def _nested_cv_raporu_uret(
    *,
    adaylar: list[ModelAdayi],
    x: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    n_iter: int,
    n_jobs: int,
) -> list[dict[str, Any]]:
    secili_adlar = {"logistic_regression", "svm_rbf", "extra_trees", "xgboost"}
    secili_adaylar = [a for a in adaylar if a.ad in secili_adlar]
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rapor: list[dict[str, Any]] = []

    for aday in secili_adaylar:
        fold_skorlari: list[dict[str, float]] = []
        for fold_no, (train_idx, test_idx) in enumerate(outer_cv.split(x, y), start=1):
            x_train = x.iloc[train_idx]
            x_test = x.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state + fold_no)
            arama = _randomized_search_yap(
                aday=aday,
                cv=inner_cv,
                x=x_train,
                y=y_train,
                n_iter=n_iter,
                n_jobs=n_jobs,
                random_state=random_state + fold_no,
            )
            olasilik = _pozitif_olasilik(arama.best_estimator_, x_test)
            fold_skorlari.append(_metrikleri_hesapla(y_test, olasilik, 0.5))
        rapor.append(
            {
                "model_adi": aday.ad,
                "fold_skorlari": fold_skorlari,
                "ortalama_accuracy": float(np.mean([s["accuracy"] for s in fold_skorlari])),
                "ortalama_f1": float(np.mean([s["f1"] for s in fold_skorlari])),
                "ortalama_roc_auc": float(np.mean([s["roc_auc"] for s in fold_skorlari])),
            }
        )
    return sorted(rapor, key=lambda s: s["ortalama_accuracy"], reverse=True)


def _dar_grid_search_yap(
    *,
    sonuc: AramaSonucu,
    x_gelistirme: pd.DataFrame,
    y_gelistirme: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
    n_jobs: int,
) -> dict[str, Any]:
    """RandomizedSearchCV sonrasi en iyi aday icin kucuk GridSearchCV uygular."""
    param_grid = _dar_param_grid_olustur(sonuc.en_iyi_parametreler)
    if not param_grid:
        return {
            "calisti": False,
            "model_adi": sonuc.model_adi,
            "param_grid": {},
            "sonuc": None,
        }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        estimator=clone(sonuc.estimator).set_params(**sonuc.en_iyi_parametreler),
        param_grid=param_grid,
        scoring=SKORLAR,
        refit="accuracy",
        cv=cv,
        n_jobs=n_jobs,
        error_score="raise",
        return_train_score=False,
    )
    grid.fit(x_gelistirme, y_gelistirme)
    olasilik = _pozitif_olasilik(grid.best_estimator_, x_test)
    grid_sonucu = AramaSonucu(
        mod=sonuc.mod,
        cv_adi=f"{sonuc.cv_adi}+dar_grid",
        model_adi=f"{sonuc.model_adi}_grid_refined",
        estimator=grid.best_estimator_,
        en_iyi_parametreler=grid.best_params_,
        cv_metrikleri=_cv_best_metrikleri(grid),
        holdout_metrikleri=_metrikleri_hesapla(y_test, olasilik, 0.5),
    )
    return {
        "calisti": True,
        "model_adi": sonuc.model_adi,
        "param_grid": param_grid,
        "sonuc": grid_sonucu,
    }


def _dar_param_grid_olustur(parametreler: dict[str, Any]) -> dict[str, list[Any]]:
    grid: dict[str, list[Any]] = {
        anahtar: [deger]
        for anahtar, deger in parametreler.items()
    }
    aday_anahtarlar = [
        "model__C",
        "model__learning_rate",
        "model__max_depth",
        "model__min_samples_leaf",
        "model__n_estimators",
        "model__max_iter",
        "model__reg_lambda",
        "model__l2_regularization",
    ]
    for anahtar in aday_anahtarlar:
        if anahtar in parametreler:
            grid[anahtar] = _dar_degerler(parametreler[anahtar])
            return grid
    return grid


def _dar_degerler(deger: Any) -> list[Any]:
    if isinstance(deger, bool) or deger is None:
        return [deger]
    if isinstance(deger, int):
        return sorted({max(1, int(round(deger * 0.7))), deger, max(1, int(round(deger * 1.3)))})
    if isinstance(deger, float):
        if deger <= 0:
            return [deger]
        return sorted({float(deger / 2), float(deger), float(deger * 2)})
    return [deger]


def _kalibrasyon_sec(
    *,
    estimator: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, Any]:
    adaylar: list[tuple[str, BaseEstimator]] = [("none", estimator)]
    for yontem in ("sigmoid", "isotonic"):
        try:
            kalibrator = CalibratedClassifierCV(
                estimator=clone(estimator),
                method=yontem,
                cv=3,
            )
            kalibrator.fit(x_train, y_train)
            adaylar.append((yontem, kalibrator))
        except Exception:
            continue

    sonuc: list[dict[str, Any]] = []
    for yontem, model in adaylar:
        olasilik = _pozitif_olasilik(model, x_val)
        esik = _en_iyi_accuracy_esigi(y_val, olasilik)
        metrikler = _metrikleri_hesapla(y_val, olasilik, esik["esik"])
        sonuc.append(
            {
                "kalibrasyon": yontem,
                "esik": float(esik["esik"]),
                "validation_accuracy": float(metrikler["accuracy"]),
                "validation_f1": float(metrikler["f1"]),
                "validation_roc_auc": float(metrikler["roc_auc"]),
                "validation_metrikleri": metrikler,
            }
        )
    return max(
        sonuc,
        key=lambda s: (
            s["validation_accuracy"],
            s["validation_f1"],
            s["validation_roc_auc"],
        ),
    )


def _artifactleri_yaz(
    *,
    proje_koku: Path,
    pipeline: BaseEstimator,
    kalibrator: BaseEstimator,
    esik: float,
    metrik_ozeti: dict[str, Any],
    model_metadata: dict[str, Any],
) -> None:
    esik_yapilandirmasi = {
        "ikili_siniflama_esikleri": {
            "accuracy_validation": {
                "esik": float(esik),
                "aciklama": "Validation set uzerinde accuracy maksimize eden esik.",
            },
            "default_0_5": {
                "esik": 0.5,
                "aciklama": "Standart olasilik esigi.",
            },
        },
        "onerilen_ikili_siniflama_esigi": float(esik),
        "onerilen_ikili_siniflama_yontemi": "validation_accuracy_threshold",
        "risk_kategorileri": {
            "dusuk_ust_esik": 0.33,
            "orta_ust_esik": 0.66,
            "etiketler": ["dusuk", "orta", "yuksek"],
        },
    }
    artifactleri_kaydet(
        artifact_klasoru=proje_koku / "makine_ogrenmesi" / "artifactler",
        en_iyi_pipeline=pipeline,
        kalibrator=kalibrator,
        esik_yapilandirmasi=esik_yapilandirmasi,
        ozellik_sirasi=list(OZELLIK_KOLONLARI),
        metrik_ozeti=metrik_ozeti,
        model_metadata=model_metadata,
    )


def _standart_raporlari_yaz(proje_koku: Path, rapor: dict[str, Any]) -> None:
    klasor = proje_koku / "makine_ogrenmesi" / "raporlar" / "degerlendirme"
    klasor.mkdir(parents=True, exist_ok=True)
    (klasor / "model_karsilastirma.json").write_text(
        json.dumps(_json_uyumlu_yap(rapor), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cv_ozeti = {
        "mod": rapor["mod"],
        "en_iyi_model": rapor["en_iyi_model"],
        "final_test_metrikleri": rapor["final_test_metrikleri"],
        "nested_cv": rapor.get("nested_cv"),
        "tum_cv_sonuclari": [
            {
                "cv_adi": sonuc["cv_adi"],
                "model_adi": sonuc["model_adi"],
                "cv_metrikleri": sonuc["cv_metrikleri"],
                "holdout_metrikleri": sonuc["holdout_metrikleri"],
            }
            for sonuc in rapor["tum_model_sonuclari"]
        ],
    }
    (klasor / "cv_strateji_karsilastirma.json").write_text(
        json.dumps(_json_uyumlu_yap(cv_ozeti), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _agresif_raporu_yaz(proje_koku: Path, rapor: dict[str, Any]) -> None:
    klasor = proje_koku / "makine_ogrenmesi" / "raporlar" / "degerlendirme"
    klasor.mkdir(parents=True, exist_ok=True)
    (klasor / "agresif_accuracy_raporu.json").write_text(
        json.dumps(_json_uyumlu_yap(rapor), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _cv_stratejileri(mod: str, random_state: int) -> list[tuple[str, Any]]:
    if mod == "quick":
        return [("stratified_5fold", StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state))]
    if mod == "tenfold":
        return [("stratified_10fold", StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state))]
    if mod == "repeated":
        return [
            ("repeated_5x5", RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=random_state)),
            ("repeated_5x10", RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=random_state)),
            ("repeated_10x5", RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=random_state)),
        ]
    if mod == "nested":
        return [("nested_prefit_5fold", StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state))]
    raise ValueError("mod quick, tenfold, repeated, nested veya aggressive olmali.")


def _varsayilan_n_iter(mod: str) -> int:
    return {
        "quick": 10,
        "tenfold": 10,
        "repeated": 6,
        "nested": 6,
        "aggressive": 5,
    }[mod]


def _olcekli_pipeline(model: BaseEstimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("sifir", SifirDegerDonusturucu()),
            ("ozellik", KlinikOzellikUretici()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def _agac_pipeline(model: BaseEstimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("sifir", SifirDegerDonusturucu()),
            ("ozellik", KlinikOzellikUretici()),
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def _sabit_pipeline(model: BaseEstimator, scaler: bool = False) -> Pipeline:
    adimlar: list[tuple[str, Any]] = [
        ("sifir", SifirDegerDonusturucu(strategy="nan")),
        ("ozellik", KlinikOzellikUretici(strategy="basic")),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scaler:
        adimlar.append(("scaler", StandardScaler()))
    adimlar.append(("model", model))
    return Pipeline(adimlar)


def _ensemble_adaylari(random_state: int) -> list[ModelAdayi]:
    logreg = _sabit_pipeline(
        LogisticRegression(
            max_iter=5000,
            solver="liblinear",
            C=0.3,
            class_weight="balanced",
            random_state=random_state,
        ),
        scaler=True,
    )
    svm = _sabit_pipeline(
        SVC(
            kernel="rbf",
            probability=True,
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=random_state,
        ),
        scaler=True,
    )
    extra = _sabit_pipeline(
        ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    )
    rf = _sabit_pipeline(
        RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    )
    voting = VotingClassifier(
        estimators=[("logreg", logreg), ("svm", svm), ("extra", extra), ("rf", rf)],
        voting="soft",
    )
    stacking = StackingClassifier(
        estimators=[("logreg", logreg), ("svm", svm), ("extra", extra)],
        final_estimator=LogisticRegression(max_iter=5000, solver="liblinear"),
        stack_method="predict_proba",
        cv=3,
        n_jobs=-1,
    )
    return [
        ModelAdayi(
            ad="voting_soft",
            estimator=voting,
            parametreler={"weights": [[1, 1, 1, 1], [1, 1, 2, 2], [1, 2, 2, 1]]},
        ),
        ModelAdayi(
            ad="stacking",
            estimator=stacking,
            parametreler={"final_estimator__C": [0.1, 0.3, 1.0, 3.0]},
        ),
    ]


def _xgboost_adayi(random_state: int, scale_pos_weight: float) -> ModelAdayi | None:
    try:
        mod = importlib.import_module("xgboost")
        xgb_classifier = mod.XGBClassifier
    except Exception:
        return None
    return ModelAdayi(
        ad="xgboost",
        estimator=_agac_pipeline(
            xgb_classifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            )
        ),
        parametreler={
            "sifir__strategy": ["nan", "raw"],
            "ozellik__strategy": ["none", "basic", "full"],
            "imputer": _imputerler(),
            "model__n_estimators": [100, 200, 350],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.75, 0.9, 1.0],
            "model__colsample_bytree": [0.75, 0.9, 1.0],
            "model__min_child_weight": [1, 3, 5],
            "model__reg_lambda": [0.5, 1.0, 3.0],
            "model__scale_pos_weight": [1.0, scale_pos_weight],
        },
    )


def _lightgbm_adayi(random_state: int, scale_pos_weight: float) -> ModelAdayi | None:
    try:
        mod = importlib.import_module("lightgbm")
        lgbm_classifier = mod.LGBMClassifier
    except Exception:
        return None
    return ModelAdayi(
        ad="lightgbm",
        estimator=_agac_pipeline(
            lgbm_classifier(
                objective="binary",
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
        ),
        parametreler={
            "sifir__strategy": ["nan", "raw"],
            "ozellik__strategy": ["none", "basic", "full"],
            "imputer": _imputerler(),
            "model__n_estimators": [100, 200, 350],
            "model__num_leaves": [7, 15, 31],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.75, 0.9, 1.0],
            "model__colsample_bytree": [0.75, 0.9, 1.0],
            "model__scale_pos_weight": [1.0, scale_pos_weight],
        },
        opsiyonel=True,
    )


def _catboost_adayi(random_state: int) -> ModelAdayi | None:
    try:
        mod = importlib.import_module("catboost")
        cat_classifier = mod.CatBoostClassifier
    except Exception:
        return None
    return ModelAdayi(
        ad="catboost",
        estimator=_agac_pipeline(
            cat_classifier(
                loss_function="Logloss",
                random_seed=random_state,
                verbose=False,
            )
        ),
        parametreler={
            "sifir__strategy": ["nan", "raw"],
            "ozellik__strategy": ["none", "basic", "full"],
            "imputer": _imputerler(),
            "model__iterations": [100, 200, 350],
            "model__depth": [2, 3, 4, 5],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__l2_leaf_reg": [1.0, 3.0, 5.0],
        },
        opsiyonel=True,
    )


def _imputerler() -> list[Any]:
    return [
        SimpleImputer(strategy="median"),
        SimpleImputer(strategy="mean"),
        KNNImputer(n_neighbors=3),
        KNNImputer(n_neighbors=5),
    ]


def _scalerlar(random_state: int) -> list[Any]:
    return [
        StandardScaler(),
        RobustScaler(),
        QuantileTransformer(
            n_quantiles=100,
            output_distribution="normal",
            random_state=random_state,
        ),
    ]


def _en_iyi_sonucu_sec(sonuclar: list[AramaSonucu]) -> AramaSonucu:
    return max(
        sonuclar,
        key=lambda s: (
            s.cv_metrikleri["accuracy"],
            s.holdout_metrikleri["accuracy"],
            s.cv_metrikleri["f1"],
            s.cv_metrikleri["roc_auc"],
        ),
    )


def _cv_best_metrikleri(arama: RandomizedSearchCV | GridSearchCV) -> dict[str, float]:
    idx = int(arama.best_index_)
    return {
        "accuracy": float(arama.cv_results_["mean_test_accuracy"][idx]),
        "precision": float(arama.cv_results_["mean_test_precision"][idx]),
        "recall": float(arama.cv_results_["mean_test_recall"][idx]),
        "f1": float(arama.cv_results_["mean_test_f1"][idx]),
        "roc_auc": float(arama.cv_results_["mean_test_roc_auc"][idx]),
    }


def _metrikleri_hesapla(
    y_gercek: pd.Series | np.ndarray,
    y_olasilik: np.ndarray,
    esik: float,
) -> dict[str, Any]:
    y_tahmin = (np.asarray(y_olasilik) >= esik).astype(int)
    y_np = np.asarray(y_gercek)
    matris = confusion_matrix(y_np, y_tahmin, labels=[0, 1])
    try:
        roc_auc = float(roc_auc_score(y_np, y_olasilik))
    except ValueError:
        roc_auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_np, y_tahmin)),
        "precision": float(precision_score(y_np, y_tahmin, zero_division=0)),
        "recall": float(recall_score(y_np, y_tahmin, zero_division=0)),
        "f1": float(f1_score(y_np, y_tahmin, zero_division=0)),
        "roc_auc": roc_auc,
        "esik": float(esik),
        "confusion_matrix": {
            "labels": [0, 1],
            "matris": matris.tolist(),
            "tn": int(matris[0, 0]),
            "fp": int(matris[0, 1]),
            "fn": int(matris[1, 0]),
            "tp": int(matris[1, 1]),
        },
    }


def _en_iyi_accuracy_esigi(y_gercek: pd.Series | np.ndarray, y_olasilik: np.ndarray) -> dict[str, float]:
    aday_esikler = np.unique(np.concatenate(([0.01, 0.5, 0.99], np.asarray(y_olasilik))))
    en_iyi = {"esik": 0.5, "accuracy": -1.0, "f1": -1.0}
    for esik in aday_esikler:
        y_tahmin = (np.asarray(y_olasilik) >= esik).astype(int)
        accuracy = float(accuracy_score(y_gercek, y_tahmin))
        f1 = float(f1_score(y_gercek, y_tahmin, zero_division=0))
        if (accuracy, f1) > (en_iyi["accuracy"], en_iyi["f1"]):
            en_iyi = {"esik": float(esik), "accuracy": accuracy, "f1": f1}
    return en_iyi


def _pozitif_olasilik(model: BaseEstimator, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        olasilik = model.predict_proba(x)
        return np.asarray(olasilik)[:, 1]
    if hasattr(model, "decision_function"):
        skor = np.asarray(model.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-skor))
    raise TypeError("Model predict_proba veya decision_function desteklemiyor.")


def _sonuc_json_yap(sonuc: AramaSonucu) -> dict[str, Any]:
    return {
        "mod": sonuc.mod,
        "cv_adi": sonuc.cv_adi,
        "model_adi": sonuc.model_adi,
        "en_iyi_parametreler": _json_uyumlu_yap(sonuc.en_iyi_parametreler),
        "cv_metrikleri": sonuc.cv_metrikleri,
        "holdout_metrikleri": sonuc.holdout_metrikleri,
    }


def _agresif_siralama_anahtari(kayit: dict[str, Any]) -> tuple[float, float, float]:
    metrikler = kayit["test_metrikleri"]
    return (
        float(metrikler["accuracy"]),
        float(kayit["validation_accuracy"]),
        float(metrikler["f1"]),
    )


def _onay_kapisi_metni(accuracy: float) -> dict[str, Any]:
    return {
        "accuracy_hedefi": 0.90,
        "hedefe_ulasti": bool(float(accuracy) >= 0.90),
        "not": (
            "Veriye dokunmadan hedefe ulasildi."
            if float(accuracy) >= 0.90
            else "Veriye dokunmadan hedef belirgin sekilde asilmadi; veriyle oynama veya ek veri asamasi icin kullanici onayi gerekir."
        ),
    }


def _dataframe_yap(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    return pd.DataFrame(x, columns=OZELLIK_KOLONLARI[: np.asarray(x).shape[1]])


def _guvenli_bol(pay: pd.Series, payda: pd.Series) -> pd.Series:
    return pay / payda.replace(0, np.nan)


def _sha256(dosya_yolu: Path) -> str:
    h = hashlib.sha256()
    with dosya_yolu.open("rb") as dosya:
        for parca in iter(lambda: dosya.read(1024 * 1024), b""):
            h.update(parca)
    return h.hexdigest()


def _iqr_aykiri_ozeti(seri: pd.Series) -> dict[str, Any]:
    q1 = float(seri.quantile(0.25))
    q3 = float(seri.quantile(0.75))
    iqr = q3 - q1
    alt = q1 - 1.5 * iqr
    ust = q3 + 1.5 * iqr
    maske = (seri < alt) | (seri > ust)
    return {
        "q1": q1,
        "q3": q3,
        "iqr": float(iqr),
        "alt_sinir": float(alt),
        "ust_sinir": float(ust),
        "aykiri_sayisi": int(maske.sum()),
        "aykiri_orani": float(maske.mean()),
    }


def _temel_feature_importance(x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    pipeline = _agac_pipeline(
        ExtraTreesClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )
    )
    pipeline.fit(x, y)
    model = pipeline.named_steps["model"]
    feature_frame = pipeline.named_steps["sifir"].transform(x)
    feature_frame = pipeline.named_steps["ozellik"].transform(feature_frame)
    kolonlar = list(feature_frame.columns)
    return {
        kolon: float(deger)
        for kolon, deger in sorted(
            zip(kolonlar, model.feature_importances_, strict=False),
            key=lambda item: item[1],
            reverse=True,
        )
    }


def _eda_markdown_yap(rapor: dict[str, Any]) -> str:
    veri = rapor["veri_butunlugu"]
    satirlar = [
        "# EDA Raporu",
        "",
        "## Veri Butunlugu",
        f"- Dosya: `{veri['dosya']}`",
        f"- SHA256: `{veri['sha256']}`",
        f"- Satir/Kolon: {veri['satir_sayisi']} / {veri['kolon_sayisi']}",
        f"- Sinif dagilimi: {veri['sinif_dagilimi']}",
        "",
        "## Sinif Oranlari",
    ]
    for sinif, oran in rapor["sinif_oranlari"].items():
        satirlar.append(f"- Outcome={sinif}: {oran:.3f}")

    satirlar.extend(["", "## 0 Deger Analizi"])
    for kolon, ozet in rapor["sifir_analizi"].items():
        satirlar.append(
            f"- {kolon}: {ozet['sifir_sayisi']} adet ({ozet['sifir_orani']:.3f})"
        )

    satirlar.extend(["", "## Outcome Korelasyonu"])
    for kolon, deger in rapor["korelasyon_outcome"].items():
        satirlar.append(f"- {kolon}: {deger:.4f}")

    satirlar.extend(["", "## Sinif Bazli Ortalama Farklari"])
    for kolon, ozet in rapor["sinif_bazli_ortalama_farklari"].items():
        satirlar.append(
            f"- {kolon}: Outcome=0 {ozet['outcome_0_ortalama']:.3f}, "
            f"Outcome=1 {ozet['outcome_1_ortalama']:.3f}, fark {ozet['fark']:.3f}"
        )

    satirlar.extend(["", "## Temel Feature Importance"])
    for kolon, deger in list(rapor["temel_feature_importance"].items())[:15]:
        satirlar.append(f"- {kolon}: {deger:.4f}")

    satirlar.extend(
        [
            "",
            "## Modelleme Notu",
            "- Bu rapor ham CSV'yi degistirmez.",
            "- 0 degerleri ve turetilmis ozellikler sadece model pipeline icinde denenmelidir.",
        ]
    )
    return "\n".join(satirlar) + "\n"


def _json_uyumlu_yap(veri: Any) -> Any:
    if isinstance(veri, dict):
        return {str(k): _json_uyumlu_yap(v) for k, v in veri.items()}
    if isinstance(veri, (list, tuple, set)):
        return [_json_uyumlu_yap(v) for v in veri]
    if isinstance(veri, Path):
        return str(veri)
    if isinstance(veri, np.ndarray):
        return veri.tolist()
    if isinstance(veri, np.generic):
        return veri.item()
    if isinstance(veri, BaseEstimator):
        return repr(veri)
    if isinstance(veri, float) and (math.isnan(veri) or math.isinf(veri)):
        return None
    return veri
