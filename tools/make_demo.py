"""Record an animated tour of the lab for the README.

Nobody evaluating this project is going to clone it, install PyTorch, start a
Flask server and click around. The README's first impression is a static PNG,
which cannot show the one thing that distinguishes this site from a write-up:
that the experiments actually run.

So this drives the real page in headless Chromium against a real backend, runs
real experiments, and stitches the frames into a GIF. Every number in the
recording is one the backend computed during the capture — there is no mock-up
and no staged data, which is the same standard the site itself is held to.

Frames are captured as a slideshow rather than a video: a handful of stills with
a long dwell time keeps the file small enough to sit in a README while still
showing each panel long enough to read. Panels that need a live experiment get
their run triggered and awaited before the shutter.

Usage (with the backend already running on :8000)::

    python server/app.py                 # terminal 1
    python tools/make_demo.py            # terminal 2

Requires ``pip install playwright pillow && playwright install chromium``.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import socketserver
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(REPO, "docs")
DEFAULT_OUT = os.path.join(DOCS, "assets", "lab-demo.gif")

# Chromium's default multi-process model needs headroom this machine does not
# always have; the same lean flags the smoke test uses.
LEAN_ARGS = [
    "--single-process", "--no-zygote", "--no-sandbox",
    "--disable-dev-shm-usage", "--disable-gpu", "--disable-extensions",
]


def serve(port: int):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=DOCS)
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def capture(api: str, port: int, width: int, height: int) -> list:
    """Drive the lab and return one screenshot per scene, as PNG bytes."""
    from playwright.sync_api import sync_playwright

    shots = []
    url = f"http://127.0.0.1:{port}/index.html#lab"

    with sync_playwright() as p:
        browser = p.chromium.launch(args=LEAN_ARGS)
        page = browser.new_page(viewport={"width": width, "height": height})
        page.route(
            "**/config.js",
            lambda route: route.fulfill(
                status=200, content_type="application/javascript",
                body=f"window.RL_API = '{api}';",
            ),
        )

        def shot(label: str) -> None:
            shots.append(page.screenshot())
            print(f"  captured: {label}")

        # Home — the framing, before any panel.
        page.goto(f"http://127.0.0.1:{port}/index.html#home", wait_until="load")
        time.sleep(2.5)
        shot("home")

        page.goto(url, wait_until="load")
        page.wait_for_function(
            "() => { const el = document.getElementById('lab-api-status');"
            "return el && !/connecting/i.test(el.textContent); }",
            timeout=90000,
        )

        # Signal or Noise — answered, scored, revealed.
        page.wait_for_function(
            "() => document.querySelectorAll('.pc-card').length > 0", timeout=90000
        )
        time.sleep(1.0)
        shot("signal-or-noise (unanswered)")
        page.eval_on_selector_all(
            '.pc-seg button[data-val="1"]', "els => els.forEach(b => b.click())"
        )
        page.click("#pc-submit")
        page.wait_for_selector("#pc-result:not([hidden])", timeout=60000)
        time.sleep(1.0)
        page.eval_on_selector("#pc-result", "el => el.scrollIntoView({block:'center'})")
        time.sleep(0.6)
        shot("signal-or-noise (scored)")

        # Playground — a real episode, then its baseline table.
        page.click('.lab-tab[data-panel="playground"]')
        page.wait_for_selector("#pg-mode", state="visible", timeout=30000)
        page.click('#pg-mode button[data-val="synthetic"]')
        page.wait_for_function(
            "() => document.getElementById('pg-regime').options.length >= 4", timeout=60000
        )
        page.select_option("#pg-regime", "momentum")
        page.click("#pg-run")
        page.wait_for_selector("#pg-output:not([hidden])", timeout=120000)
        time.sleep(1.5)
        shot("playground")
        page.eval_on_selector("#pg-baselines", "el => el.scrollIntoView({block:'center'})")
        time.sleep(0.8)
        shot("baselines")

        # X-Ray, then the occlusion attribution under it.
        page.click('.lab-tab[data-panel="xray"]')
        page.wait_for_selector("#xr-body:not([hidden])", timeout=60000)
        page.wait_for_function(
            "() => document.querySelectorAll('.xr-feat').length > 0", timeout=60000
        )
        time.sleep(1.2)
        shot("x-ray")
        page.click("#attr-run")
        page.wait_for_selector("#attr-out:not([hidden])", timeout=120000)
        page.eval_on_selector("#attr-card", "el => el.scrollIntoView({block:'start'})")
        time.sleep(1.0)
        shot("attribution")

        # Walk-forward — the fold geometry is the clearest single image here.
        page.click('.lab-tab[data-panel="walkforward"]')
        page.wait_for_selector("#wf-run", state="visible", timeout=30000)
        page.click('#wf-mode button[data-val="synthetic"]')
        page.wait_for_function(
            "() => document.getElementById('wf-source').options.length >= 4", timeout=60000
        )
        page.click("#wf-run")
        page.wait_for_selector("#wf-out:not([hidden])", timeout=240000)
        time.sleep(1.2)
        shot("walk-forward")

        # The surrogate test: the project's sharpest result.
        page.click('.lab-tab[data-panel="surrogate"]')
        page.wait_for_selector("#sg-body:not([hidden])", timeout=60000)
        time.sleep(1.2)
        shot("surrogate")

        browser.close()
    return shots


def write_gif(shots: list, out: str, width: int, ms_per_frame: int) -> None:
    """Stitch the stills into a looping GIF, scaled to a README-friendly width."""
    import io as _io

    from PIL import Image

    frames = []
    for raw in shots:
        img = Image.open(_io.BytesIO(raw)).convert("RGB")
        if img.width > width:
            img = img.resize(
                (width, round(img.height * width / img.width)), Image.LANCZOS
            )
        # A 256-colour adaptive palette keeps the dark UI readable at a fraction
        # of the size of a full-colour frame.
        frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=256))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frames[0].save(
        out, save_all=True, append_images=frames[1:],
        duration=ms_per_frame, loop=0, optimize=True,
    )
    size_mb = os.path.getsize(out) / 1e6
    print(f"\nWrote {out}  ({len(frames)} frames, {size_mb:.2f} MB)")
    if size_mb > 10:
        print("  NOTE: over 10 MB — GitHub will be slow to render this. "
              "Re-run with a smaller --width.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--width", type=int, default=1000, help="Output GIF width.")
    ap.add_argument("--shot-width", type=int, default=1280)
    ap.add_argument("--shot-height", type=int, default=880)
    ap.add_argument("--ms", type=int, default=2600, help="Milliseconds per frame.")
    args = ap.parse_args()

    serve(args.port)
    print(f"capturing against {args.api} ...")
    shots = capture(args.api, args.port, args.shot_width, args.shot_height)
    write_gif(shots, args.out, args.width, args.ms)


if __name__ == "__main__":
    main()
