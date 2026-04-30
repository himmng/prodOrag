from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from backend.api.routes import router as api_router

app = FastAPI(title="protoRAG")

app.mount("/static", StaticFiles(directory="backend/static"), name="static")

templates = Jinja2Templates(directory="backend/templates")

app.include_router(api_router, prefix="/api")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")
