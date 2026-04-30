"""API endpoint testleri."""

from __future__ import annotations

from fastapi.testclient import TestClient

from uygulama.main import app


GECERLI_PAYLOAD = {
    "pregnancies": 2,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree_function": 0.627,
    "age": 50,
}


def test_health_endpoint_ok_donmeli() -> None:
    client = TestClient(app)

    yanit = client.get("/health")

    assert yanit.status_code == 200
    veri = yanit.json()
    assert veri["durum"] == "ok"
    assert veri["uygulama"] == "diyabet-risk-tahmini"
    assert veri["ortam"] == "gelistirme"


def test_predict_gecerli_veride_beklenen_alanlari_donmeli(monkeypatch) -> None:
    # Test hizini sabit tutmak icin SHAP hesaplamasini mockluyoruz.
    monkeypatch.setattr(
        "uygulama.servisler.tahmin_servisi._guvenli_top_faktorleri_uret",
        lambda **_: [
            {
                "ozellik": "glucose",
                "ozellik_degeri": 148.0,
                "shap_katkisi": 0.21,
                "yon": "arttirici",
            }
        ],
    )

    client = TestClient(app)
    yanit = client.post("/predict", json=GECERLI_PAYLOAD)

    assert yanit.status_code == 200
    veri = yanit.json()
    assert set(veri.keys()) == {
        "olasilik",
        "sinif",
        "risk_kategorisi",
        "top_faktorler",
        "kisa_aciklama",
    }
    assert 0 <= float(veri["olasilik"]) <= 1
    assert veri["sinif"] in (0, 1)
    assert veri["risk_kategorisi"] in {"dusuk", "orta", "yuksek"}


def test_predict_gecersiz_veride_422_donmeli() -> None:
    client = TestClient(app)
    gecersiz = {**GECERLI_PAYLOAD, "age": 10}

    yanit = client.post("/predict", json=gecersiz)

    assert yanit.status_code == 422
