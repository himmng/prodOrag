from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.db import SessionLocal
from app.models.domain.configs import ConfigProfile
from app.models.schemas.config import (
    ConfigProfileCreate,
    ConfigProfileRead,
    ConfigProfileUpdate,
)

router = APIRouter(prefix="/config", tags=["config"])


async def get_session() -> AsyncSession:
    async with SessionLocal() as session:
        yield session


@router.get("/profiles", response_model=List[ConfigProfileRead])
async def list_profiles(session: AsyncSession = Depends(get_session)) -> List[ConfigProfileRead]:
    result = await session.execute(select(ConfigProfile))
    profiles = result.scalars().all()
    return [ConfigProfileRead.model_validate(p) for p in profiles]


@router.post("/profiles", response_model=ConfigProfileRead)
async def create_profile(
    payload: ConfigProfileCreate,
    session: AsyncSession = Depends(get_session),
) -> ConfigProfileRead:
    profile = ConfigProfile(**payload.model_dump())
    session.add(profile)
    await session.commit()
    await session.refresh(profile)
    return ConfigProfileRead.model_validate(profile)


@router.get("/profiles/{profile_id}", response_model=ConfigProfileRead)
async def get_profile(
    profile_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> ConfigProfileRead:
    profile = await session.get(ConfigProfile, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Config profile not found")
    return ConfigProfileRead.model_validate(profile)


@router.put("/profiles/{profile_id}", response_model=ConfigProfileRead)
async def update_profile(
    profile_id: UUID,
    payload: ConfigProfileUpdate,
    session: AsyncSession = Depends(get_session),
) -> ConfigProfileRead:
    profile = await session.get(ConfigProfile, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Config profile not found")
    for key, value in payload.model_dump().items():
        setattr(profile, key, value)
    await session.commit()
    await session.refresh(profile)
    return ConfigProfileRead.model_validate(profile)


@router.delete("/profiles/{profile_id}")
async def delete_profile(
    profile_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    profile = await session.get(ConfigProfile, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Config profile not found")
    await session.delete(profile)
    await session.commit()
