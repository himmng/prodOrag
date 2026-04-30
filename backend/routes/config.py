from fastapi import APIRouter

from ..config.models import AppConfig
from ..config.store import load_config, save_config


router = APIRouter()


@router.get("/", response_model=AppConfig)
async def get_config() -> AppConfig:
    return load_config()


@router.put("/", response_model=AppConfig)
async def update_config(config: AppConfig) -> AppConfig:
    save_config(config)
    return config
