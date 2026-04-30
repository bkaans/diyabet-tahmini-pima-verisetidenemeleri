"""Tahmin ve saglik endpointleri."""

from fastapi import APIRouter, HTTPException

from uygulama.cekirdek.ayarlar import ayarlari_yukle
from uygulama.semalar.cikti_semalari import TahminCiktisi
from uygulama.semalar.girdi_semalari import TahminGirdisi
from uygulama.servisler.tahmin_servisi import tek_ornek_tahmin_uret


router = APIRouter(prefix="", tags=["tahmin"])


@router.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Uygulama saglik kontrol endpointi."""
    ayarlar = ayarlari_yukle()
    return {
        "durum": "ok",
        "uygulama": ayarlar.app_adi,
        "ortam": ayarlar.app_env,
    }


@router.post("/predict", response_model=TahminCiktisi)
def predict(girdi: TahminGirdisi) -> TahminCiktisi:
    """Tek ornek icin risk tahmini endpointi."""
    try:
        sonuc = tek_ornek_tahmin_uret(girdi)
        return TahminCiktisi(**sonuc)
    except (ValueError, TypeError) as hata:
        raise HTTPException(status_code=400, detail=str(hata)) from hata
    except FileNotFoundError as hata:
        raise HTTPException(status_code=500, detail=str(hata)) from hata
    except Exception as hata:  # pragma: no cover - beklenmeyen hata
        raise HTTPException(
            status_code=500,
            detail="Tahmin islemi sirasinda beklenmeyen bir hata olustu.",
        ) from hata
