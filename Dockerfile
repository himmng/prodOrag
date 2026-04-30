FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY backend ./backend
COPY frontend ./frontend

ENV DATA_DIR=/app/data

EXPOSE 8000

CMD ["python", "-m", "backend.server"]
