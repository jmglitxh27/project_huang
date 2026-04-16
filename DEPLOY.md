# Run RNDA in the cloud (no local terminal required)

The full app (PDF ingest, embeddings, graph, novelty) is **too large for Vercel**. Use a **container host** and deploy from **GitHub** using the website only.

## 1. Put the code on GitHub

Use **GitHub Desktop** (drag-and-drop, Commit, Push) if you prefer not to use the terminal.

## 2. Deploy with Docker

This repo includes a **`Dockerfile`** that installs everything and runs:

`uvicorn rnda.web.app:app --host 0.0.0.0 --port $PORT`

### Option A — Render (browser only)

1. Sign up at [render.com](https://render.com) and connect your GitHub account.
2. **New → Blueprint** (or **Web Service**).
3. Pick this repository.
4. If using **Web Service** manually: **Environment** = **Docker**, Dockerfile path = `./Dockerfile`.
5. Add an environment variable **`OPENAI_API_KEY`** (optional but recommended for LLM features).
6. Deploy. Open the URL Render gives you (e.g. `https://rnda.onrender.com`).

**Note:** Free/small instances may be slow on first request or run out of RAM when loading PyTorch/sentence-transformers. If the service crashes, upgrade the instance or switch to Railway/Fly with more memory.

### Option B — Railway

1. [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
2. Select the repo; Railway usually detects **Dockerfile** automatically.
3. **Variables** → add **`OPENAI_API_KEY`**.
4. Deploy and open the generated domain.

### Option C — Fly.io

Usually needs the Fly CLI once for auth; if you want fully GUI-only, prefer Render or Railway.

## 3. What you get

- **Full UI** at `/` (same as local `rnda-web`).
- **Persistent disk:** default containers are **ephemeral**; runs under `data/runs` may disappear when the instance restarts unless your host mounts a **volume** (configure in Render/Railway).

## 4. Environment variables (optional)

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | LLM refinement, novelty, literature reports |
| `RNDA_GROBID_URL` | Only if you run a separate GROBID server |

## 5. Vercel

Vercel can only host the **small** `server_vercel` shell (no full pipeline). For the real app, use Docker on Render/Railway/etc., as above.
