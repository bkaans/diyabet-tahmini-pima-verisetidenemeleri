"""FastAPI ana uygulama giris noktasi."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from uygulama.api import api_router
from uygulama.cekirdek.ayarlar import ayarlari_yukle


APP_KLASORU = Path(__file__).resolve().parent
STATIK_KLASORU = APP_KLASORU / "statik"


def uygulama_olustur() -> FastAPI:
    ayarlar = ayarlari_yukle()

    app = FastAPI(
        title=ayarlar.app_adi,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.mount("/statik", StaticFiles(directory=str(STATIK_KLASORU)), name="statik")
    app.include_router(api_router)

    return app


app = uygulama_olustur()
