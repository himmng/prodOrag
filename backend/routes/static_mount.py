from fastapi import APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request


router = APIRouter()


templates = Jinja2Templates(directory="frontend/templates")


def mount_static(app):
  app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

  @app.get("/")
  async def index(request: Request):
      return templates.TemplateResponse("index.html", {"request": request})
