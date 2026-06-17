"""Root WSGI entrypoint.

Render's default start command is ``gunicorn app:app`` run from the repo root.
The actual application lives in ``server/app.py``; this thin shim re-exports it
so the backend deploys whether Render builds from the repo root *or* from the
``server/`` directory (the cleaner setup — see DEPLOY.md).
"""

from server.app import app  # noqa: F401  (re-exported for `gunicorn app:app`)
