# Deploying RL-Trader

Two pieces, deployed independently:

| Piece | Host | What it is |
|---|---|---|
| **Frontend** (`docs/`) | **Vercel** | The static dashboard. Ships a baked `results.js`, so it works **with no backend at all**. |
| **Backend** (`server/`) | **Render** | An optional, featherweight API that serves the trained policy for **live inference** (pure NumPy — no PyTorch, starts instantly). |

The frontend degrades gracefully: if no backend is configured, the "Run live"
section simply stays hidden and everything else works.

---

## 1. Backend → Render

> ⚠️ **Critical:** the API lives in the **`server/`** subfolder. Render must build
> from there, or it will install the heavy root `requirements.txt` (torch + CUDA)
> and fail with `gunicorn: command not found`.

### Option A — one-click Blueprint (recommended)

The repo includes a [`render.yaml`](render.yaml) Blueprint that sets everything
(including `rootDir: server`) for you.

1. Push the repo to GitHub (already done).
2. Go to **[dashboard.render.com](https://dashboard.render.com)** → **New +** → **Blueprint**.
3. Connect the repo `RL-for-Crypto-and-stocks-`. Render reads `render.yaml` and shows a service **`rl-trader-api`** (Free plan).
4. Click **Apply**. Wait ~2–3 min (the build is small — no torch).
5. Test: `https://<your-service>.onrender.com/health` → `{"status":"ok","policies":["crypto","stock"]}`

### Option B — manual web service (if you created it by hand)

Set these in the service's **Settings**, then **Manual Deploy → Deploy latest commit**:

| Field | Value |
|---|---|
| **Root Directory** | `server` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --workers 1 --bind 0.0.0.0:$PORT --timeout 120` |

**Don't want to set Root Directory?** Then use these repo-root-relative commands instead (equivalent):

- Build: `pip install -r server/requirements.txt`
- Start: `cd server && gunicorn app:app --workers 1 --bind 0.0.0.0:$PORT --timeout 120`

> **Free-tier note:** the service sleeps after ~15 min idle, so the *first*
> request after a nap takes ~30–50 s to wake (cold start). The UI tells the user
> this. Subsequent calls are fast.

**What it serves:** `/health`, `/api/results` (the dashboard data), `/api/tickers`,
and `/api/live?market=stock&ticker=AAPL` (fetches ~2 yrs of real prices and runs
the agent live).

---

## 2. Frontend → Vercel (static, zero build)

1. Go to **[vercel.com/new](https://vercel.com/new)** → import the same GitHub repo.
2. In project settings, set **Root Directory = `docs`**. *(That's the only setting that matters — the rest auto-detects as a static site.)*
3. Click **Deploy**. Done — you get a URL like `https://rl-trader.vercel.app`.

At this point the dashboard is live and fully functional on its baked data. No
backend required.

---

## 3. Connect them (enable the lab)

The dashboard renders from baked data and needs no backend at all. The **lab** —
Playground, X-Ray, counterfactuals, distribution shift, live statistics — is what
the API powers.

1. Edit [`docs/config.js`](docs/config.js):
   ```js
   window.RL_API = "https://rl-trader-api.onrender.com";   // your Render URL
   ```
2. Commit & push. Vercel auto-redeploys in ~30 s.

The backend already allows CORS from any origin, so the connection just works.

### Deploy the two in the right order

The frontend and backend deploy independently, and the lab needs a backend that
actually serves the lab endpoints. `render.yaml` sets `autoDeploy: true` with no
`branch` key, so **Render tracks the repository's default branch** — a feature
branch's Vercel preview will still be talking to whatever is deployed from `main`.

Deploy the backend first, or at least don't be surprised by the gap:

```bash
# after merging to main, confirm the deployed API actually has the lab
curl -s https://rl-trader-api.onrender.com/health   | grep -q capabilities && echo "lab-capable"
curl -s -o /dev/null -w '%{http_code}\n' https://rl-trader-api.onrender.com/api/meta   # expect 200
```

The page handles this state honestly rather than failing confusingly: if `/health`
answers but `/api/meta` does not, the status pill reads **"Backend vX — too old for
the lab"** instead of claiming the API is live, and the run controls stay disabled.
That is a distinct state from *unreachable*, so the fix is obvious when you see it.

---

## Run it locally

```bash
# Backend
cd server
pip install -r requirements.txt
python app.py                      # http://127.0.0.1:8000/health

# Frontend (any static server)
npx serve docs                     # or just open docs/index.html
# then set window.RL_API = "http://127.0.0.1:8000" in docs/config.js
```

Verify the whole thing end to end — 77 headless-browser checks, including that an
unreachable *or outdated* backend renders nothing rather than a plausible fake:

```bash
python tools/smoke_lab.py          # needs: pip install playwright && playwright install chromium
```

## Regenerating the model + dashboard data

After retraining, refresh what gets deployed:

```bash
make verify-data                                             # fail early if the data moved
python tools/build_site_data.py --real --timesteps 200000    # -> docs/results.js
python tools/export_policy.py                                # -> server/models/*.npz
python tools/make_figures.py                                 # -> docs/assets/*.png
git add -A && git commit -m "Refresh model + results" && git push
```

Both hosts auto-deploy on push.

> **Check the data pin first.** `tools/fetch_data.py` requests a *relative* window,
> so re-fetching later slides the train/test split and the rebuilt figures will not
> match the published ones. `make verify-data` compares `data/raw` against the
> committed `data/SNAPSHOT.json` and exits non-zero on drift. Regenerating
> `server/models/*.npz` replaces the exact policies the paper describes — do it
> deliberately, not as a side effect of a refresh.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `gunicorn: command not found` / it installs torch+CUDA | Render is building from the repo root. Set **Root Directory = `server`** (or use the `server/`-prefixed commands above) and redeploy. |
| Live call hangs ~30–50 s | Render free-tier cold start — normal on the first request. |
| `data fetch failed` | Yahoo rate-limited that ticker; retry, or pick another. Results are cached 30 min. |
| Live section never appears | `window.RL_API` is empty in `docs/config.js` (that's the default — set it). |
| Status pill says **"too old for the lab"** | The API answers `/health` but not `/api/meta` — it's running a revision from before the lab. Redeploy the backend from a commit that includes `server/lab.py`. |
| Lab panels error but the dashboard works | Expected when no backend is configured: the dashboard is static, the lab is not. Nothing in the lab is simulated in the browser. |
| CORS error | Make sure `RL_API` points at the Render URL with `https://` and no trailing slash. |
