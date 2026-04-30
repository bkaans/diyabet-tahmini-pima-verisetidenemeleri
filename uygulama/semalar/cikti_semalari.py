"""Tahmin ciktilari icin Pydantic semalari."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from .dogrulamalar import birim_aralik_dogrula, risk_kategorisi_dogrula, yon_dogrula


class FaktorSemasi(BaseModel):
    """Model aciklamasinda gosterilecek faktor bilgisi."""

    model_config = ConfigDict(extra="forbid")

    ozellik: str = Field(..., description="Ozellik adi")
    ozellik_degeri: Any = Field(..., description="Ornekteki ozellik degeri")
    shap_katkisi: float = Field(..., description="SHAP katkisi")
    yon: Literal["arttirici", "azaltici"] = Field(..., description="Katki yonu")

    @field_validator("yon")
    @classmethod
    def yonu_dogrula(cls, deger: str) -> str:
        return yon_dogrula(deger)


class TahminCiktisi(BaseModel):
    """Tahmin endpointi standart donus semasi."""

    model_config = ConfigDict(extra="forbid")

    olasilik: float = Field(..., description="Diyabet olasiligi")
    sinif: Literal[0, 1] = Field(..., description="Ikili sinif tahmini")
    risk_kategorisi: Literal["dusuk", "orta", "yuksek"] = Field(..., description="Risk seviyesi")
    top_faktorler: list[FaktorSemasi] = Field(
        default_factory=list,
        description="Tahmine en cok etki eden faktorler",
        max_length=3,
    )
    kisa_aciklama: str = Field(default="", description="Sonuc icin kisa aciklama")

    @field_validator("olasilik")
    @classmethod
    def olasiligi_dogrula(cls, deger: float, info: ValidationInfo) -> float:
        return float(birim_aralik_dogrula(info.field_name, deger))

    @field_validator("risk_kategorisi")
    @classmethod
    def riski_dogrula(cls, deger: str) -> str:
        return risk_kategorisi_dogrula(deger)
