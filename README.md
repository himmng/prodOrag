# protoRAG

Monorepo with a FastAPI backend and Next.js 14 frontend for a basic RAG-style prototype.

## Project structure

- `backend/` – FastAPI app with SQLite + SQLAlchemy ORM
- `frontend/` – Next.js 14 + Tailwind UI
- `infra/` – Docker Compose for backend, frontend, and Qdrant
- `data/` – Local data directory (mounted into backend container)
- `qdrant_storage/` – Local storage for Qdrant vector DB

---

## Prerequisites

- Node.js 20+
- Python 3.11+
- Docker & Docker Compose (for containerized setup)
- Make (optional, if you add Makefile commands later)

All commands below are assumed to be run from the repo root.

---

## 1. Running everything via Docker Compose

This is the easiest way to run the full stack.

### 1.1. One-time setup

Ensure directories used by Docker volumes exist:

```bash
mkdir -p data qdrant_storage
```

### 1.2. Start services

From the `infra/` directory:

```bash
cd infra
docker compose up --build
```

This will start:

- Backend FastAPI at http://localhost:8000
- Frontend Next.js dev server at http://localhost:3000
- Qdrant at http://localhost:6333

Logs stream in the same terminal. Use `Ctrl+C` to stop.

### 1.3. Running in detached mode

```bash
cd infra
docker compose up --build -d
```

To stop:

```bash
cd infra
docker compose down
```

---

## 2. Running locally without Docker

### 2.1. Backend (FastAPI)

From `backend/`:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install .
```

Run the API server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at:

- API root / docs: http://localhost:8000/docs

Health endpoint:

- `GET http://localhost:8000/health`

Main functional APIs (paths may evolve, see code for exact details):

- Config profile CRUD – `backend/app/api/routes/config.py`
- Chat sessions/messages – `backend/app/api/routes/chat.py`

#### 2.1.1. Backend quick checks

Compile-time check (already passing):

```bash
cd backend
python -m compileall app
```

### 2.2. Frontend (Next.js 14)

From `frontend/`:

```bash
cd frontend
npm install
```

Run the dev server:

```bash
npm run dev
```

The frontend will be available at:

- http://localhost:3000

Available routes:

- `/` – Landing page
- `/chat` – Chat UI
- `/documents` – Documents view
- `/settings` – Settings/configuration

#### 2.2.1. Frontend lint

```bash
cd frontend
npm run lint
```

---

## 3. Backend details

### 3.1. Database

- SQLite with SQLAlchemy asyncio
- Default DB file is created automatically in the working directory as configured in `backend/app/infrastructure/db.py`
- On startup, `Base.metadata.create_all` creates all tables (see `backend/app/main.py`)

Domain models are in:

- `backend/app/models/domain/`

Pydantic schemas are in:

- `backend/app/models/schemas/`

### 3.2. API routers

Routers are registered in `backend/app/main.py`:

- `health_router` – `/health` endpoint for liveness checks
- `config_router` – Config profile CRUD endpoints (e.g., `/config/...`)
- `chat_router` – Chat sessions and messages (e.g., `/chat/...`)

Refer to:

- `backend/app/api/routes/health.py`
- `backend/app/api/routes/config.py`
- `backend/app/api/routes/chat.py`

---

## 4. Environment & configuration

Currently, configuration is minimal and largely hard-coded. As you add settings:

- Use `pydantic-settings` for typed environment-based config
- Define settings classes under `backend/app/config/` (if/when added)
- Document any required env vars here

Backend examples (not yet required, but typical):

```bash
export BACKEND_HOST=0.0.0.0
export BACKEND_PORT=8000
export DATABASE_URL="sqlite+aiosqlite:///./app.db"
```

---

## 5. Useful dev workflows

### 5.1. Typical local dev loop

1. Start backend:
   - `cd backend && source .venv/bin/activate && uvicorn app.main:app --reload`
2. Start frontend in another terminal:
   - `cd frontend && npm run dev`
3. Open http://localhost:3000 to use the app
4. Use http://localhost:8000/docs to inspect and test backend APIs

### 5.2. Rebuilding containers after code changes

If you change Python dependencies (`pyproject.toml`) or Node deps (`package.json`):

```bash
cd infra
docker compose build
```

Then restart:

```bash
cd infra
docker compose up
```

---

## 6. Troubleshooting

### Port already in use

- If port 8000 or 3000 is busy, stop other processes or adjust ports in `infra/docker-compose.yml` or CLI flags.

### Docker volume path issues

- Ensure `data/` and `qdrant_storage/` exist and are writeable
- On permission errors, fix directory permissions or run Docker Desktop with proper rights

### Dependency problems

- Backend: recreate venv and `pip install .`
- Frontend: remove `node_modules` and `package-lock.json`, then `npm install`

---

## 7. Production notes (future work)

For production you would typically:

- Build a static Next.js app (`npm run build` + `npm run start`)
- Run FastAPI with a production ASGI server behind a reverse proxy
- Externalize SQLite or switch to a managed DB
- Harden CORS, auth, logging, and observability

This README covers local and Docker-based development for the current skeleton.