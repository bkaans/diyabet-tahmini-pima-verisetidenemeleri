"""Model egitimi icin pipeline, CV ve grid arama yapilandirmalari."""

from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import FunctionTransformer

from .on_isleme import (
    median_imputer_olustur,
    sifirlari_nan_donustur_pipeline,
    standard_scaler_olustur,
)
RANDOM_STATE = 42


def stratified_kfold_olustur() -> StratifiedKFold:
    """5-fold StratifiedKFold nesnesi dondurur."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def model_pipeline_olustur() -> dict[str, Pipeline]:
    """Her model icin leakage engelleyen pipeline yapisini kurar."""
    modeller = _model_nesneleri()
    return {
        model_adi: Pipeline(steps=_ortak_pipeline_adimlari(model))
        for model_adi, model in modeller.items()
    }


def hiperparametre_gridlerini_olustur() -> dict[str, dict[str, list]]:
    """Model bazli GridSearch parametre sozlugunu dondurur."""
    return {
        "logistic_regression": {
            "model__C": [0.1, 1, 10],
        },
        "random_forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10, None],
        },
        "xgboost": {
            "model__learning_rate": [0.1, 0.3],
            "model__max_depth": [3, 6],
            "model__n_estimators": [100, 200],
        },
    }


def grid_searchleri_olustur(
    scoring: str = "roc_auc",
    n_jobs: int = -1,
    refit: bool = True,
) -> dict[str, GridSearchCV]:
    """Her model icin GridSearchCV nesnesini hazirlar."""
    pipeline_sozlugu = model_pipeline_olustur()
    grid_sozlugu = hiperparametre_gridlerini_olustur()
    cv = stratified_kfold_olustur()

    return {
        model_adi: GridSearchCV(
            estimator=pipeline_sozlugu[model_adi],
            param_grid=grid_sozlugu[model_adi],
            scoring=scoring,
            cv=cv,
            refit=refit,
            n_jobs=n_jobs,
        )
        for model_adi in pipeline_sozlugu
    }


def _sifirlari_nan_donustur(veri: pd.DataFrame) -> pd.DataFrame:
    """Sadece belirlenen kolonlarda 0 degerini NaN'a cevirir."""
    return sifirlari_nan_donustur_pipeline(veri)


def _ortak_pipeline_adimlari(model) -> list[tuple[str, object]]:
    return [
        (
            "sifir_nan_donusumu",
            FunctionTransformer(_sifirlari_nan_donustur, validate=False),
        ),
        ("imputer", median_imputer_olustur()),
        ("scaler", standard_scaler_olustur()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", model),
    ]


def _model_nesneleri() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": _xgboost_modeli_olustur(),
    }


def _xgboost_modeli_olustur():
    try:
        from xgboost import XGBClassifier
    except Exception as hata:  # pragma: no cover - ortam bagimli
        raise ImportError(
            "XGBoost yuklenemedi. macOS icin `brew install libomp` komutunu "
            "calistirip tekrar deneyin."
        ) from hata

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
