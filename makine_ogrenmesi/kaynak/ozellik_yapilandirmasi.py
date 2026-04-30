"""Modelleme katmani icin ozellik ve kolon sabitleri."""

OZELLIK_KOLONLARI = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

HEDEF_KOLONU = "Outcome"

ZORUNLU_KOLONLAR = [*OZELLIK_KOLONLARI, HEDEF_KOLONU]

SIFIRI_EKSIK_SAYILAN_KOLONLAR = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

SIFIRI_EKSIK_SAYILMAYAN_KOLONLAR = ["Pregnancies"]
