from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import config as config_routes
from .routes import chat as chat_routes
from .routes import documents as documents_routes
from .routes import static_mount


def create_app() -> FastAPI:
    app = FastAPI(title="Local RAG App", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routers
    app.include_router(config_routes.router, prefix="/api/config", tags=["config"])
    app.include_router(chat_routes.router, prefix="/api/chat", tags=["chat"])
    app.include_router(documents_routes.router, prefix="/api/documents", tags=["documents"])

    # Frontend: mount static files and index route at "/"
    static_mount.mount_static(app)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
