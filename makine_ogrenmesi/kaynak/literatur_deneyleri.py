"""Literaturdeki PIMA yaklasimlarini ham CSV'yi bozmadan dener."""

from __future__ import annotations

import importlib
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from .maksimum_skor_arama import (
    RANDOM_STATE,
    SKORLAR,
    KlinikOzellikUretici,
    SifirDegerDonusturucu,
    _artifactleri_yaz,
    _cv_best_metrikleri,
    _en_iyi_accuracy_esigi,
    _json_uyumlu_yap,
    _metrikleri_hesapla,
    _onay_kapisi_metni,
    _pozitif_olasilik,
    veri_butunlugu_ozeti,
)
from .ozellik_yapilandirmasi import HEDEF_KOLONU, OZELLIK_KOLONLARI
from .veri_yukleyici import veri_setini_yukle


RAPOR_DOSYA_ADI = "literatur_deneyleri.json"
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


@dataclass(frozen=True)
class LiteraturReferansi:
    """Deneyin dayandigi literatur notu."""

    ad: str
    yil: int
    kategori: str
    raporlanan_skor: str
    url: str


@dataclass(frozen=True)
class LiteraturDeneyi:
    """Tek bir literatur eslesmesi ve calistirma politikasi."""

    ad: str
    profil: str
    referans: LiteraturReferansi
    pipeline_aciklamasi: str
    estimator: BaseEstimator | None
    cv_adlari: tuple[str, ...]
    parametreler: dict[str, list[Any]] | None = None
    n_iter: int | None = None
    veri_mudahalesi_sayilir_mi: bool = False
    veri_mudahalesi_etiketi: str = "Pipeline ici on isleme; ham CSV degismez."
    atlama_nedeni: str | None = None


def literatur_deneyleri_calistir(
    *,
    veri_yolu: Path,
    proje_koku: Path,
    random_state: int = RANDOM_STATE,
    test_boyutu: float = 0.2,
    n_iter: int = 16,
    n_jobs: int = -1,
    artifact_yaz: bool = False,
    deney_adlari: tuple[str, ...] | None = None,
    cv_adlari: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Literatur profillerini calistirir ve JSON rapor yazar."""
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

    deneyler = _deneyleri_olustur(y_gelistirme, random_state, n_jobs)
    if deney_adlari is not None:
        secili = set(deney_adlari)
        deneyler = [deney for deney in deneyler if deney.ad in secili]

    sonuc_kayitlari: list[dict[str, Any]] = []
    artifact_adayi: tuple[dict[str, Any], BaseEstimator] | None = None

    for deney in deneyler:
        calisilacak_cvler = deney.cv_adlari
        if cv_adlari is not None:
            secili_cv = set(cv_adlari)
            calisilacak_cvler = tuple(cv for cv in deney.cv_adlari if cv in secili_cv)
        for cv_adi in calisilacak_cvler:
            if deney.estimator is None:
                sonuc_kayitlari.append(
                    _atlanan_kayit_yap(
                        deney=deney,
                        cv_adi=cv_adi,
                        baslangic_ozeti=baslangic_ozeti,
                        bitis_ozeti=veri_butunlugu_ozeti(veri_yolu),
                    )
                )
                print(f"[literatur/{deney.ad}/{cv_adi}] atlandi: {deney.atlama_nedeni}")
                continue

            cv = _cv_olustur(cv_adi, random_state)
            try:
                kayit, estimator = _deneyi_degerlendir(
                    deney=deney,
                    cv_adi=cv_adi,
                    cv=cv,
                    x_gelistirme=x_gelistirme,
                    y_gelistirme=y_gelistirme,
                    x_test=x_test,
                    y_test=y_test,
                    random_state=random_state,
                    n_iter=n_iter,
                    n_jobs=n_jobs,
                    baslangic_ozeti=baslangic_ozeti,
                    veri_yolu=veri_yolu,
                )
            except Exception as hata:
                kayit = _atlanan_kayit_yap(
                    deney=deney,
                    cv_adi=cv_adi,
                    baslangic_ozeti=baslangic_ozeti,
                    bitis_ozeti=veri_butunlugu_ozeti(veri_yolu),
                    neden=str(hata),
                )
                estimator = None
            sonuc_kayitlari.append(kayit)

            if kayit["durum"] == "tamamlandi":
                print(
                    f"[literatur/{deney.ad}/{cv_adi}] "
                    f"cv_accuracy={kayit['cv_metrikleri']['accuracy']:.4f} "
                    f"holdout_accuracy={kayit['holdout_metrikleri']['accuracy']:.4f}"
                )
                if estimator is not None and (
                    artifact_adayi is None
                    or _sonuc_siralama_anahtari(kayit) > _sonuc_siralama_anahtari(artifact_adayi[0])
                ):
                    artifact_adayi = (kayit, estimator)

    bitis_ozeti = veri_butunlugu_ozeti(veri_yolu)
    if bitis_ozeti["sha256"] != baslangic_ozeti["sha256"]:
        raise RuntimeError("Ham diabetes.csv degisti; literatur deneyleri durduruldu.")

    tamamlananlar = [kayit for kayit in sonuc_kayitlari if kayit["durum"] == "tamamlandi"]
    en_iyi_cv = max(tamamlananlar, key=_sonuc_siralama_anahtari) if tamamlananlar else None
    en_iyi_holdout = (
        max(tamamlananlar, key=_holdout_siralama_anahtari) if tamamlananlar else None
    )
    artifact_ozeti: dict[str, Any] = {"yazildi": False}
    if artifact_yaz and artifact_adayi is not None:
        artifact_ozeti = _artifact_adayini_yaz(
            proje_koku=proje_koku,
            x_gelistirme=x_gelistirme,
            y_gelistirme=y_gelistirme,
            x_test=x_test,
            y_test=y_test,
            random_state=random_state,
            kayit=artifact_adayi[0],
            estimator=artifact_adayi[1],
        )

    hedef_accuracy = 0.90
    en_iyi_accuracy = (
        float(en_iyi_cv["cv_metrikleri"]["accuracy"])
        if en_iyi_cv is not None
        else 0.0
    )
    rapor = {
        "mod": "literatur_deneyleri",
        "olusturulma_zamani_utc": datetime.now(timezone.utc).isoformat(),
        "veri_politikasi": {
            "ham_csv_degistirilmedi": True,
            "satir_ekleme_silme_cogaltma_yok": True,
            "sentetik_veri_yok": True,
            "harici_veri_yok": True,
            "pipeline_ici_imputation_scaling_feature_engineering_var": True,
        },
        "veri_butunlugu_baslangic": baslangic_ozeti,
        "veri_butunlugu_bitis": bitis_ozeti,
        "deney_sayisi": len(sonuc_kayitlari),
        "tamamlanan_deney_sayisi": len(tamamlananlar),
        "atlanan_deney_sayisi": len(sonuc_kayitlari) - len(tamamlananlar),
        "en_iyi_cv_sonuc": en_iyi_cv,
        "en_iyi_holdout_sonuc": en_iyi_holdout,
        "artifact_ozeti": artifact_ozeti,
        "sonuclar": sorted(
            sonuc_kayitlari,
            key=lambda kayit: (
                kayit["durum"] == "tamamlandi",
                _sonuc_siralama_anahtari(kayit) if kayit["durum"] == "tamamlandi" else (0, 0, 0, 0),
            ),
            reverse=True,
        ),
        "onay_kapisi": _onay_kapisi_metni(en_iyi_accuracy),
        "notlar": [
            "KNN imputation, scaling ve feature engineering sadece pipeline icindedir.",
            "Feature selection iceren deneyler kalici uygulanmadi; raporda onay gerektiren veri mudahalesi olarak etiketlendi.",
            "Oversampling, sentetik veri, satir eleme ve harici veri bu komutta kullanilmadi.",
        ],
    }
    _raporu_yaz(proje_koku, rapor)
    return rapor


def _deneyi_degerlendir(
    *,
    deney: LiteraturDeneyi,
    cv_adi: str,
    cv: StratifiedKFold,
    x_gelistirme: pd.DataFrame,
    y_gelistirme: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
    n_iter: int,
    n_jobs: int,
    baslangic_ozeti: dict[str, Any],
    veri_yolu: Path,
) -> tuple[dict[str, Any], BaseEstimator]:
    if deney.estimator is None:
        raise ValueError(deney.atlama_nedeni or "Estimator olusturulamadi.")

    if deney.parametreler:
        arama = RandomizedSearchCV(
            estimator=clone(deney.estimator),
            param_distributions=deney.parametreler,
            n_iter=_n_iter_sinirla(deney.parametreler, deney.n_iter or n_iter),
            scoring=SKORLAR,
            refit="accuracy",
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
            error_score="raise",
            return_train_score=False,
        )
        arama.fit(x_gelistirme, y_gelistirme)
        estimator = arama.best_estimator_
        cv_metrikleri = _cv_best_metrikleri(arama)
        cv_foldlari: list[dict[str, Any]] = []
        en_iyi_parametreler = arama.best_params_
        tuning_ozeti = {
            "yontem": "RandomizedSearchCV",
            "n_iter": int(arama.n_iter),
            "arama_alani": _json_uyumlu_yap(deney.parametreler),
        }
    else:
        cv_ozeti = _manuel_cv_degerlendir(
            estimator=deney.estimator,
            x=x_gelistirme,
            y=y_gelistirme,
            cv=cv,
        )
        cv_metrikleri = cv_ozeti["ortalama_metrikler"]
        cv_foldlari = cv_ozeti["fold_metrikleri"]
        estimator = clone(deney.estimator)
        estimator.fit(x_gelistirme, y_gelistirme)
        en_iyi_parametreler = {}
        tuning_ozeti = {"yontem": "sabit_literatur_profili"}

    holdout_metrikleri = _holdout_metrikleri_hesapla(estimator, x_test, y_test)
    bitis_ozeti = veri_butunlugu_ozeti(veri_yolu)
    kayit = _temel_kayit_yap(deney, cv_adi, baslangic_ozeti, bitis_ozeti)
    kayit.update(
        {
            "durum": "tamamlandi",
            "cv_metrikleri": cv_metrikleri,
            "cv_fold_metrikleri": cv_foldlari,
            "holdout_metrikleri": holdout_metrikleri,
            "en_iyi_parametreler": _json_uyumlu_yap(en_iyi_parametreler),
            "tuning_ozeti": tuning_ozeti,
        }
    )
    return kayit, estimator


def _deneyleri_olustur(
    y: pd.Series,
    random_state: int,
    n_jobs: int,
) -> list[LiteraturDeneyi]:
    negatif = int((y == 0).sum())
    pozitif = int((y == 1).sum())
    scale_pos_weight = negatif / max(pozitif, 1)
    lgbm_modeli, lgbm_atlama = _lightgbm_classifier_olustur(random_state, scale_pos_weight)
    xgb_modeli, xgb_atlama = _xgboost_classifier_olustur(random_state, scale_pos_weight)

    hossain = LiteraturReferansi(
        ad="Hossain et al.",
        yil=2022,
        kategori="Ham veriye en yakin / detay sinirli",
        raporlanan_skor="Accuracy 90%, AUC 0.936 civari raporlanmis",
        url="https://www.sciencedirect.com/org/science/article/pii/S1546221822011122",
    )
    amma = LiteraturReferansi(
        ad="Amma N.G.",
        yil=2024,
        kategori="Model tarafi oynanmis voting ensemble",
        raporlanan_skor="Accuracy 88.89%",
        url="https://www.sciencedirect.com/science/article/pii/S1110866524000045",
    )
    altamimi = LiteraturReferansi(
        ad="Altamimi et al.",
        yil=2024,
        kategori="KNN imputer + XGB/RF/ETC soft voting",
        raporlanan_skor="Accuracy 98.59% raporlanmis; agir optimize edilmis hat",
        url="https://link.springer.com/article/10.1186/s12874-024-02324-0",
    )
    ansari = LiteraturReferansi(
        ad="Ansari et al.",
        yil=2025,
        kategori="SVM + preprocessing + feature selection + resampling",
        raporlanan_skor="SVM accuracy 91.5%",
        url="https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1620268/full",
    )

    deneyler = [
        LiteraturDeneyi(
            ad="hossain_knn",
            profil="Hossain",
            referans=hossain,
            pipeline_aciklamasi="0->NaN, median imputation, StandardScaler, KNN.",
            estimator=_olcekli_pipeline(
                KNeighborsClassifier(n_neighbors=7, weights="distance", p=2),
                imputer=SimpleImputer(strategy="median"),
                zero_strategy="nan",
                feature_strategy="none",
            ),
            cv_adlari=("stratified_5fold", "stratified_10fold"),
        ),
        LiteraturDeneyi(
            ad="hossain_lightgbm",
            profil="Hossain",
            referans=hossain,
            pipeline_aciklamasi="0->NaN, median imputation, LightGBM.",
            estimator=(
                _agac_pipeline(lgbm_modeli, imputer=SimpleImputer(strategy="median"), feature_strategy="none")
                if lgbm_modeli is not None
                else None
            ),
            cv_adlari=("stratified_5fold", "stratified_10fold"),
            atlama_nedeni=lgbm_atlama,
        ),
        LiteraturDeneyi(
            ad="hossain_lightgbm_knn_soft",
            profil="Hossain",
            referans=hossain,
            pipeline_aciklamasi="LightGBM + KNN soft voting; her alt model kendi pipeline on islemesini yapar.",
            estimator=(
                VotingClassifier(
                    estimators=[
                        ("lgbm", _agac_pipeline(clone(lgbm_modeli), imputer=SimpleImputer(strategy="median"), feature_strategy="none")),
                        ("knn", _olcekli_pipeline(KNeighborsClassifier(n_neighbors=7, weights="distance"), feature_strategy="none")),
                    ],
                    voting="soft",
                    weights=[2, 1],
                    n_jobs=n_jobs,
                )
                if lgbm_modeli is not None
                else None
            ),
            cv_adlari=("stratified_5fold", "stratified_10fold"),
            atlama_nedeni=lgbm_atlama,
        ),
        LiteraturDeneyi(
            ad="hossain_lightgbm_knn_adaboost_soft",
            profil="Hossain",
            referans=hossain,
            pipeline_aciklamasi="LightGBM + KNN + AdaBoost soft voting.",
            estimator=(
                VotingClassifier(
                    estimators=[
                        ("lgbm", _agac_pipeline(clone(lgbm_modeli), imputer=SimpleImputer(strategy="median"), feature_strategy="none")),
                        ("knn", _olcekli_pipeline(KNeighborsClassifier(n_neighbors=7, weights="distance"), feature_strategy="none")),
                        ("ada", _agac_pipeline(AdaBoostClassifier(n_estimators=150, learning_rate=0.05, random_state=random_state), feature_strategy="none")),
                    ],
                    voting="soft",
                    weights=[2, 1, 1],
                    n_jobs=n_jobs,
                )
                if lgbm_modeli is not None
                else None
            ),
            cv_adlari=("stratified_5fold", "stratified_10fold"),
            atlama_nedeni=lgbm_atlama,
        ),
        LiteraturDeneyi(
            ad="amma_rf_rbfsvm_knn_soft",
            profil="Amma En-RfRsK",
            referans=amma,
            pipeline_aciklamasi="RF + RBF-SVM + KNN soft voting; class_weight, scaling ve KNNImputer kullanilir.",
            estimator=_amma_voting(random_state, n_jobs, voting="soft", imputer=KNNImputer(n_neighbors=5)),
            cv_adlari=("stratified_5fold", "stratified_10fold"),
        ),
        LiteraturDeneyi(
            ad="amma_rf_rbfsvm_knn_hard",
            profil="Amma En-RfRsK",
            referans=amma,
            pipeline_aciklamasi="RF + RBF-SVM + KNN hard voting; class_weight, scaling ve KNNImputer kullanilir.",
            estimator=_amma_voting(random_state, n_jobs, voting="hard", imputer=KNNImputer(n_neighbors=3)),
            cv_adlari=("stratified_5fold", "stratified_10fold"),
        ),
        LiteraturDeneyi(
            ad="altamimi_knnimputer_xgb_rf_etc_soft",
            profil="Altamimi controlled",
            referans=altamimi,
            pipeline_aciklamasi="KNNImputer + XGBoost + RandomForest + ExtraTrees soft voting.",
            estimator=(
                VotingClassifier(
                    estimators=[
                        ("xgb", _agac_pipeline(xgb_modeli, imputer=KNNImputer(n_neighbors=5), feature_strategy="basic")),
                        ("rf", _agac_pipeline(_rf(random_state, n_jobs), imputer=KNNImputer(n_neighbors=5), feature_strategy="basic")),
                        ("etc", _agac_pipeline(_etc(random_state, n_jobs), imputer=KNNImputer(n_neighbors=5), feature_strategy="basic")),
                    ],
                    voting="soft",
                    weights=[2, 1, 1],
                    n_jobs=n_jobs,
                )
                if xgb_modeli is not None
                else None
            ),
            cv_adlari=("stratified_5fold", "stratified_10fold"),
            atlama_nedeni=xgb_atlama,
        ),
        LiteraturDeneyi(
            ad="ansari_svm_rbf_genis_tuning",
            profil="Ansari SVM",
            referans=ansari,
            pipeline_aciklamasi="RBF-SVM, 10-fold CV, genis C/gamma aramasi; kalici feature selection yok.",
            estimator=_olcekli_pipeline(
                SVC(kernel="rbf", probability=True, random_state=random_state),
                imputer=KNNImputer(n_neighbors=5),
                zero_strategy="nan",
                feature_strategy="basic",
            ),
            cv_adlari=("stratified_10fold",),
            parametreler={
                "sifir__strategy": ["nan", "raw"],
                "ozellik__strategy": ["none", "basic", "full"],
                "imputer": [SimpleImputer(strategy="median"), KNNImputer(n_neighbors=3), KNNImputer(n_neighbors=5)],
                "scaler": [StandardScaler(), RobustScaler()],
                "model__C": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
                "model__gamma": ["scale", 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                "model__class_weight": [None, "balanced"],
            },
        ),
        LiteraturDeneyi(
            ad="ansari_svm_rbf_feature_selection",
            profil="Ansari SVM",
            referans=ansari,
            pipeline_aciklamasi="RBF-SVM + SelectKBest mutual information; feature selection sadece pipeline icinde denenir.",
            estimator=_svm_feature_selection_pipeline(random_state),
            cv_adlari=("stratified_10fold",),
            parametreler={
                "sifir__strategy": ["nan", "raw"],
                "ozellik__strategy": ["basic", "full"],
                "imputer": [SimpleImputer(strategy="median"), KNNImputer(n_neighbors=5)],
                "selector__k": [5, 6, 8, 10, 12],
                "model__C": [0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
                "model__gamma": ["scale", 0.003, 0.01, 0.03, 0.1],
                "model__class_weight": [None, "balanced"],
            },
            veri_mudahalesi_sayilir_mi=True,
            veri_mudahalesi_etiketi=(
                "Onay gerektiren feature selection deneyi; ham CSV degismez, "
                "kalici final secim kullanici onayi olmadan yapilmaz."
            ),
        ),
    ]
    return deneyler


def _manuel_cv_degerlendir(
    *,
    estimator: BaseEstimator,
    x: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> dict[str, Any]:
    fold_metrikleri: list[dict[str, Any]] = []
    for fold_no, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        model = clone(estimator)
        x_train = x.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        model.fit(x_train, y_train)
        fold_metrikleri.append(_fold_metrikleri_hesapla(model, x_test, y_test, fold_no))
    return {
        "ortalama_metrikler": _ortalama_metrikleri_al(fold_metrikleri),
        "fold_metrikleri": fold_metrikleri,
    }


def _fold_metrikleri_hesapla(
    model: BaseEstimator,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    fold_no: int,
) -> dict[str, Any]:
    y_tahmin = model.predict(x_test)
    try:
        y_skor = _pozitif_olasilik(model, x_test)
        roc_auc_kaynagi = "predict_proba_or_decision_function"
    except Exception:
        y_skor = np.asarray(y_tahmin, dtype=float)
        roc_auc_kaynagi = "predicted_label"
    try:
        roc_auc = float(roc_auc_score(y_test, y_skor))
    except ValueError:
        roc_auc = float("nan")
    return {
        "fold": fold_no,
        "accuracy": float(accuracy_score(y_test, y_tahmin)),
        "precision": float(precision_score(y_test, y_tahmin, zero_division=0)),
        "recall": float(recall_score(y_test, y_tahmin, zero_division=0)),
        "f1": float(f1_score(y_test, y_tahmin, zero_division=0)),
        "roc_auc": roc_auc,
        "roc_auc_kaynagi": roc_auc_kaynagi,
    }


def _holdout_metrikleri_hesapla(
    estimator: BaseEstimator,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    try:
        olasilik = _pozitif_olasilik(estimator, x_test)
        return _metrikleri_hesapla(y_test, olasilik, 0.5)
    except Exception:
        y_tahmin = estimator.predict(x_test)
        try:
            roc_auc = float(roc_auc_score(y_test, y_tahmin))
        except ValueError:
            roc_auc = float("nan")
        return {
            "accuracy": float(accuracy_score(y_test, y_tahmin)),
            "precision": float(precision_score(y_test, y_tahmin, zero_division=0)),
            "recall": float(recall_score(y_test, y_tahmin, zero_division=0)),
            "f1": float(f1_score(y_test, y_tahmin, zero_division=0)),
            "roc_auc": roc_auc,
            "esik": 0.5,
            "roc_auc_kaynagi": "predicted_label",
        }


def _artifact_adayini_yaz(
    *,
    proje_koku: Path,
    x_gelistirme: pd.DataFrame,
    y_gelistirme: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
    kayit: dict[str, Any],
    estimator: BaseEstimator,
) -> dict[str, Any]:
    x_train, x_val, y_train, y_val = train_test_split(
        x_gelistirme,
        y_gelistirme,
        test_size=0.25,
        random_state=random_state,
        stratify=y_gelistirme,
    )
    esik_modeli = clone(estimator)
    esik_modeli.fit(x_train, y_train)
    try:
        val_olasilik = _pozitif_olasilik(esik_modeli, x_val)
        esik = float(_en_iyi_accuracy_esigi(y_val, val_olasilik)["esik"])
    except Exception:
        esik = 0.5

    final_pipeline = clone(estimator)
    final_pipeline.fit(x_gelistirme, y_gelistirme)
    final_metrikler = _holdout_metrikleri_hesapla(final_pipeline, x_test, y_test)
    if "confusion_matrix" in final_metrikler:
        final_metrikler = _metrikleri_hesapla(
            y_test,
            _pozitif_olasilik(final_pipeline, x_test),
            esik,
        )

    _artifactleri_yaz(
        proje_koku=proje_koku,
        pipeline=final_pipeline,
        kalibrator=final_pipeline,
        esik=esik,
        metrik_ozeti=final_metrikler,
        model_metadata={
            "model_adi": kayit["deney_adi"],
            "mod": "literatur_deneyleri",
            "cv_stratejisi": kayit["cv_stratejisi"],
            "literatur_referansi": kayit["literatur_referansi"],
            "ana_metrik": "accuracy",
            "veri_politikasi": "ham veri degistirilmedi",
            "feature_selection_onay_gerektirir": bool(kayit["veri_mudahalesi_sayilir_mi"]),
        },
    )
    return {
        "yazildi": True,
        "deney_adi": kayit["deney_adi"],
        "cv_stratejisi": kayit["cv_stratejisi"],
        "esik": esik,
        "final_test_metrikleri": final_metrikler,
    }


def _temel_kayit_yap(
    deney: LiteraturDeneyi,
    cv_adi: str,
    baslangic_ozeti: dict[str, Any],
    bitis_ozeti: dict[str, Any],
) -> dict[str, Any]:
    return {
        "deney_adi": deney.ad,
        "profil": deney.profil,
        "literatur_referansi": {
            "ad": deney.referans.ad,
            "yil": deney.referans.yil,
            "kategori": deney.referans.kategori,
            "raporlanan_skor": deney.referans.raporlanan_skor,
            "url": deney.referans.url,
        },
        "pipeline": deney.pipeline_aciklamasi,
        "cv_stratejisi": cv_adi,
        "ham_csv_degisti_mi": bitis_ozeti["sha256"] != baslangic_ozeti["sha256"],
        "veri_mudahalesi_sayilir_mi": bool(deney.veri_mudahalesi_sayilir_mi),
        "veri_mudahalesi_etiketi": deney.veri_mudahalesi_etiketi,
    }


def _atlanan_kayit_yap(
    *,
    deney: LiteraturDeneyi,
    cv_adi: str,
    baslangic_ozeti: dict[str, Any],
    bitis_ozeti: dict[str, Any],
    neden: str | None = None,
) -> dict[str, Any]:
    kayit = _temel_kayit_yap(deney, cv_adi, baslangic_ozeti, bitis_ozeti)
    kayit.update(
        {
            "durum": "atlandi",
            "atlama_nedeni": neden or deney.atlama_nedeni or "Model bagimliligi yok.",
            "cv_metrikleri": {},
            "holdout_metrikleri": {},
        }
    )
    return kayit


def _olcekli_pipeline(
    model: BaseEstimator,
    *,
    imputer: BaseEstimator | None = None,
    zero_strategy: str = "nan",
    feature_strategy: str = "basic",
    scaler: BaseEstimator | None = None,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("sifir", SifirDegerDonusturucu(strategy=zero_strategy)),
            ("ozellik", KlinikOzellikUretici(strategy=feature_strategy)),
            ("imputer", imputer or SimpleImputer(strategy="median")),
            ("scaler", scaler or StandardScaler()),
            ("model", model),
        ]
    )


def _agac_pipeline(
    model: BaseEstimator,
    *,
    imputer: BaseEstimator | None = None,
    zero_strategy: str = "nan",
    feature_strategy: str = "basic",
) -> Pipeline:
    return Pipeline(
        steps=[
            ("sifir", SifirDegerDonusturucu(strategy=zero_strategy)),
            ("ozellik", KlinikOzellikUretici(strategy=feature_strategy)),
            ("imputer", imputer or SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def _amma_voting(
    random_state: int,
    n_jobs: int,
    *,
    voting: str,
    imputer: BaseEstimator,
) -> VotingClassifier:
    rf = _agac_pipeline(_rf(random_state, n_jobs), imputer=clone(imputer), feature_strategy="basic")
    svm = _olcekli_pipeline(
        SVC(
            kernel="rbf",
            probability=True,
            C=3.0,
            gamma="scale",
            class_weight="balanced",
            random_state=random_state,
        ),
        imputer=clone(imputer),
        feature_strategy="basic",
        scaler=RobustScaler(),
    )
    knn = _olcekli_pipeline(
        KNeighborsClassifier(n_neighbors=7, weights="distance"),
        imputer=clone(imputer),
        feature_strategy="basic",
        scaler=StandardScaler(),
    )
    return VotingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("knn", knn)],
        voting=voting,
        weights=[2, 2, 1],
        n_jobs=n_jobs,
    )


def _svm_feature_selection_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("sifir", SifirDegerDonusturucu(strategy="nan")),
            ("ozellik", KlinikOzellikUretici(strategy="full")),
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=_mutual_info_skoru, k=8)),
            (
                "model",
                SVC(
                    kernel="rbf",
                    probability=True,
                    C=3.0,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def _rf(random_state: int, n_jobs: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=n_jobs,
    )


def _etc(random_state: int, n_jobs: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs,
    )


def _xgboost_classifier_olustur(
    random_state: int,
    scale_pos_weight: float,
) -> tuple[BaseEstimator | None, str | None]:
    try:
        mod = importlib.import_module("xgboost")
        xgb_classifier = mod.XGBClassifier
    except Exception as hata:
        return None, f"xgboost import edilemedi: {hata}"
    return (
        xgb_classifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=220,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=3,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=1,
        ),
        None,
    )


def _lightgbm_classifier_olustur(
    random_state: int,
    scale_pos_weight: float,
) -> tuple[BaseEstimator | None, str | None]:
    try:
        mod = importlib.import_module("lightgbm")
        lgbm_classifier = mod.LGBMClassifier
    except Exception as hata:
        return None, f"lightgbm import edilemedi: {hata}"
    return (
        lgbm_classifier(
            objective="binary",
            n_estimators=220,
            num_leaves=15,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        ),
        None,
    )


def _cv_olustur(cv_adi: str, random_state: int) -> StratifiedKFold:
    if cv_adi == "stratified_5fold":
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    if cv_adi == "stratified_10fold":
        return StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    raise ValueError(f"Bilinmeyen CV stratejisi: {cv_adi}")


def _ortalama_metrikleri_al(fold_metrikleri: list[dict[str, Any]]) -> dict[str, float]:
    metrik_adlari = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ortalamalar: dict[str, float] = {}
    for metrik in metrik_adlari:
        degerler = [float(kayit[metrik]) for kayit in fold_metrikleri]
        ortalamalar[metrik] = _nanmean(degerler)
    return ortalamalar


def _nanmean(degerler: list[float]) -> float:
    temiz = [deger for deger in degerler if not math.isnan(deger)]
    if not temiz:
        return float("nan")
    return float(np.mean(temiz))


def _n_iter_sinirla(parametreler: dict[str, list[Any]], n_iter: int) -> int:
    kombinasyon_sayisi = 1
    for degerler in parametreler.values():
        kombinasyon_sayisi *= max(1, len(degerler))
    return max(1, min(int(n_iter), kombinasyon_sayisi))


def _sonuc_siralama_anahtari(kayit: dict[str, Any]) -> tuple[float, float, float, float]:
    metrikler = kayit.get("cv_metrikleri") or {}
    holdout = kayit.get("holdout_metrikleri") or {}
    return (
        float(metrikler.get("accuracy", 0.0) or 0.0),
        float(holdout.get("accuracy", 0.0) or 0.0),
        float(metrikler.get("f1", 0.0) or 0.0),
        float(metrikler.get("roc_auc", 0.0) or 0.0),
    )


def _holdout_siralama_anahtari(kayit: dict[str, Any]) -> tuple[float, float, float, float]:
    metrikler = kayit.get("cv_metrikleri") or {}
    holdout = kayit.get("holdout_metrikleri") or {}
    return (
        float(holdout.get("accuracy", 0.0) or 0.0),
        float(metrikler.get("accuracy", 0.0) or 0.0),
        float(holdout.get("f1", 0.0) or 0.0),
        float(holdout.get("roc_auc", 0.0) or 0.0),
    )


def _mutual_info_skoru(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return mutual_info_classif(x, y, random_state=RANDOM_STATE)


def _raporu_yaz(proje_koku: Path, rapor: dict[str, Any]) -> None:
    klasor = proje_koku / "makine_ogrenmesi" / "raporlar" / "degerlendirme"
    klasor.mkdir(parents=True, exist_ok=True)
    (klasor / RAPOR_DOSYA_ADI).write_text(
        json.dumps(_json_uyumlu_yap(rapor), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
