# GitHub Setup — PowerShell Commands (free public URL)

Run these from the project root in **PowerShell** to put the engine on GitHub
and the dashboard live on a free public URL via **GitHub Pages**.

## 1. One-time Git identity (skip if configured before)

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

## 2. Initialize the repo and commit

```powershell
cd path\to\dcm-syndicate-engine
git init
git add .
git commit -m "DCM syndicate pricing & bookbuilding engine: agents, pipeline, quant core, dashboard"
git branch -M main
```

## 3. Create the GitHub repo (two options)

**Option A — GitHub CLI (easiest, installs with `winget install GitHub.cli`):**

```powershell
winget install --id GitHub.cli
gh auth login
gh repo create dcm-syndicate-engine --public --source=. --push
```

**Option B — manual (no CLI):**

1. Open https://github.com/new → name it `dcm-syndicate-engine` → **Public** → *do not* initialize with README.
2. Then:

```powershell
git remote add origin https://github.com/<YOUR-USERNAME>/dcm-syndicate-engine.git
git push -u origin main
```

## 4. Turn on GitHub Pages (free public URL for the dashboard)

```powershell
# with GitHub CLI
gh api repos/<YOUR-USERNAME>/dcm-syndicate-engine/pages -X POST -f "source[branch]=main" -f "source[path]=/docs"
```

Or manually: **repo → Settings → Pages → Source: `Deploy from a branch` → Branch: `main`, Folder: `/docs` → Save.**

Within a minute your dashboard is live at:

```
https://<YOUR-USERNAME>.github.io/dcm-syndicate-engine/
```

## 5. Verify CI (lint + tests + Trivy container scan) runs on push

```powershell
gh run watch   # or open the Actions tab in the browser
```

The workflow in `.github/workflows/ci.yaml` runs automatically on every push
to `main` and every pull request: ruff, mypy, pytest, Docker build, Trivy
HIGH/CRITICAL gate, SARIF upload to the Security tab.

## 6. Later updates

```powershell
git add .
git commit -m "describe your change"
git push
```

---

### Free backend deploy (Render — one click)

The repo ships `render.yaml` (Blueprint). After the repo is on GitHub:

1. Open https://dashboard.render.com/blueprint → **New Blueprint Instance** → select your repo → **Apply**.
2. Render builds the image, trains the ML models from `data/sample_deals.csv`, and starts the FastAPI bridge.
3. You get a free public URL like `https://dcm-syndicate-api.onrender.com` — landing page at `/`, live dashboard at `/dashboard`, API at `/api/book`, WebSocket at `/ws`.

**PowerShell quick checks after deploy:**

```powershell
curl.exe https://dcm-syndicate-api.onrender.com/health
curl.exe https://dcm-syndicate-api.onrender.com/api/book
start https://dcm-syndicate-api.onrender.com/dashboard
```

Then point GitHub Pages at the same server: when served from Render, the
dashboard auto-detects the live feed (`ws://.../ws`) instead of simulating.

The dashboard also ships **standalone** on GitHub Pages (pure client-side sim)
if you don't want a backend at all.

---

### Optional: container image on GHCR (free registry)

The CI pushes images to `ghcr.io/<your-user>/dcm-syndicate-engine` on every
push to `main`. Pull it anywhere with:

```powershell
docker pull ghcr.io/<YOUR-USERNAME>/dcm-syndicate-engine:latest
```

### Run Redpanda locally for the live streaming loop

```powershell
docker run -d --name redpanda -p 9092:9092 docker.redpanda.com/redpandadata/redpanda:latest redpanda start --smp 1 --memory 1G --kafka-addr PLAINTEXT://0.0.0.0:9092 --advertise-kafka-addr PLAINTEXT://localhost:9092

# Terminal 1 — simulate institutional IOI flow
python -m dcm_engine.pipeline.producer

# Terminal 2 — aggregate the book
python -m dcm_engine.pipeline.consumer
```
