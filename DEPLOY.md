# Run RNDA in the cloud (no local terminal required)

The full app (PDF ingest, embeddings, graph, novelty) is **too large for Vercel**. Use a **container host** and deploy from **GitHub** using the website only.

## 1. Put the code on GitHub

Use **GitHub Desktop** (drag-and-drop, Commit, Push) if you prefer not to use the terminal.

## 2. Deploy with Docker

This repo includes a **`Dockerfile`** that installs everything and runs:

`uvicorn rnda.web.app:app --host 0.0.0.0 --port $PORT`

### Faster Docker builds (Railway / Render)

- **Dependency layer:** `requirements-docker.txt` is installed in its **own** layer. PyTorch and the rest are **not** re-downloaded on every push unless you change that file or `pyproject.toml` deps.
- **App layer:** Only `pyproject.toml` + `rnda/` are copied for the quick `pip install . --no-deps` step when you change code.
- **Pip cache:** the Dockerfile uses BuildKit’s `RUN --mount=type=cache,...` so repeat builds on the same builder reuse pip’s download cache.
- **Smaller context:** `.dockerignore` excludes the whole `data/` tree so uploads stay fast.

If you add a dependency, update **`requirements-docker.txt`** to match **`pyproject.toml`**, or the image may miss the new package.

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
- **Health check:** `GET /api/health` — use this in Railway/Render so the proxy waits until the app is up (heavy deps load only when you **start a run**, not on every page view).
- **Persistent disk:** default containers are **ephemeral**; runs under `data/runs` may disappear when the instance restarts unless your host mounts a **volume** (configure in Render/Railway).

### Railway shows “Internal Server Error”

1. Open **Deployments → latest deploy → View logs** and look for **OOM**, **ModuleNotFoundError**, or **exit code 137** (out of memory).
2. **RAM:** the free/small tier may be too small when a **run** loads PyTorch / sentence-transformers. Try **at least ~2 GB** for full pipeline runs, or expect failures during ingest/gap stages.
3. Confirm the service runs **Docker** using this repo’s **`Dockerfile`** (this repo includes **`railway.toml`** to prefer that builder).
4. In the browser try **`https://YOUR_URL/api/health`** — you should see `{"status":"ok",...}`. If that works but `/` fails, check logs for template/static paths.

## 4. Environment variables (optional)

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | LLM refinement, novelty, literature reports |
| `RNDA_GROBID_URL` | Only if you run a separate GROBID server |

## 5. Vercel

Vercel can only host the **small** `server_vercel` shell (no full pipeline). For the real app, use Docker on Render/Railway/etc., as above.
