FROM python:3.11-slim AS base

WORKDIR /app

COPY pyproject.toml setup.cfg ./

RUN pip install --upgrade pip && pip install .

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
