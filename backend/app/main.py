from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.config import router as config_router
from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.infrastructure.db import Base, engine

app = FastAPI()


@app.on_event("startup")
async def on_startup() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


app.include_router(health_router)
app.include_router(config_router)
app.include_router(chat_router)
app.include_router(documents_router)
