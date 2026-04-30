"""Tahmin girdisi icin Pydantic semalari."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from .dogrulamalar import sayisal_aralik_dogrula


class TahminGirdisi(BaseModel):
    """Tahmin endpointine gelecek kullanici girdisi."""

    model_config = ConfigDict(extra="forbid")

    pregnancies: int = Field(..., description="Gebelik sayisi")
    glucose: float = Field(..., description="Glukoz degeri")
    blood_pressure: float = Field(..., description="Kan basinci")
    skin_thickness: float = Field(..., description="Cilt kalinligi")
    insulin: float = Field(..., description="Insulin degeri")
    bmi: float = Field(..., description="Vucut kitle indeksi")
    diabetes_pedigree_function: float = Field(..., description="Ailesel risk puani")
    age: int = Field(..., description="Yas")

    @field_validator(
        "pregnancies",
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
        "diabetes_pedigree_function",
        "age",
    )
    @classmethod
    def alan_araliklarini_dogrula(cls, deger: float | int, info: ValidationInfo) -> float | int:
        return sayisal_aralik_dogrula(info.field_name, deger)
