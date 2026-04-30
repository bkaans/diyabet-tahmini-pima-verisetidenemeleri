"""API router birlestirme katmani."""

from fastapi import APIRouter

from .rotalar.sayfalar import router as sayfalar_router
from .rotalar.tahmin import router as tahmin_router


api_router = APIRouter()
api_router.include_router(sayfalar_router)
api_router.include_router(tahmin_router)

__all__ = ["api_router"]
