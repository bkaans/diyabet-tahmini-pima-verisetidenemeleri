"""Uygulama ayarlari ve ortam degiskeni yukleyicisi."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv


PROJE_KOKU = Path(__file__).resolve().parents[2]
load_dotenv(PROJE_KOKU / ".env")


@dataclass(frozen=True)
class Ayarlar:
    app_adi: str
    app_env: str
    app_host: str
    app_port: int
    log_seviyesi: str


@lru_cache(maxsize=1)
def ayarlari_yukle() -> Ayarlar:
    """Ortam degiskenlerinden uygulama ayarlarini olusturur."""
    return Ayarlar(
        app_adi=os.getenv("APP_ADI", "diyabet-risk-tahmini"),
        app_env=os.getenv("APP_ENV", "gelistirme"),
        app_host=os.getenv("APP_HOST", "127.0.0.1"),
        app_port=int(os.getenv("APP_PORT", "8000")),
        log_seviyesi=os.getenv("LOG_SEVIYESI", "INFO"),
    )
