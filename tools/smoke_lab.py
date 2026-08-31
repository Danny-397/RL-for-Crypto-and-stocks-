"""Browser smoke test for the interactive lab.

The pytest suite covers the backend thoroughly, but the lab is only real if the
*page* works: charts have to paint, the scrubber has to move the readouts, and —
most importantly — an unreachable backend has to produce an honest error rather
than a plausible-looking fake.

This drives the actual page in headless Chromium against a locally running API.
It is intentionally **not** part of the pytest suite: it needs Playwright plus a
~150 MB browser download, which would be a heavy dependency for CI to carry for
a handful of assertions.

Usage (two terminals, from the repo root)::

    python server/app.py                        # terminal 1
    python tools/smoke_lab.py                   # terminal 2

Requires ``pip install playwright && playwright install chromium``.
Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import socketserver
import sys
import threading
import time

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    """Record a single assertion, printing it either way."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def serve(port: int):
    """Serve docs/ locally so the page runs over http rather than file://."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=DOCS)
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def run(api: str, port: int, shot: str | None) -> None:
    from playwright.sync_api import sync_playwright

    httpd = serve(port)
    url = f"http://127.0.0.1:{port}/index.html#lab"

    with sync_playwright() as p:
        browser = p.chromium.launch()

        # ── 1. the lab against a live backend ──────────────────────────────
        print("\nLive backend")
        errors: list[str] = []
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda e: errors.append(str(e)))
        # config.js loads after any init script, so intercept it to repoint the API.
        page.route(
            "**/config.js",
            lambda route: route.fulfill(
                status=200,
                content_type="application/javascript",
                body=f"window.RL_API = '{api}';",
            ),
        )
        page.goto(url, wait_until="load")
        time.sleep(1.5)

        check("lab view routes from #lab",
              page.eval_on_selector("#view-lab", "el => el.classList.contains('active')"))
        check("backend reports live",
              "LIVE" in page.inner_text("#lab-api-status").upper(),
              page.inner_text("#lab-api-status").strip())

        page.click('#pg-mode button[data-val="synthetic"]')
        time.sleep(0.4)
        check("synthetic controls appear", page.is_visible("#pg-syn-row"))
        n_regimes = page.eval_on_selector("#pg-regime", "el => el.options.length")
        check("regimes loaded from API", n_regimes >= 4, f"{n_regimes} regimes")

        page.select_option("#pg-regime", "momentum")
        page.fill("#pg-seed", "7")
        page.click("#pg-run")
        page.wait_for_selector("#pg-output:not([hidden])", timeout=60000)
        time.sleep(0.8)

        check("experiment completed", "complete" in page.inner_text("#pg-status").lower(),
              page.inner_text("#pg-status").strip())
        check("metrics rendered",
              page.eval_on_selector_all("#pg-metrics .metric", "els => els.length") >= 6)
        check("receipt rendered",
              page.eval_on_selector_all("#pg-receipt dt", "els => els.length") >= 8)

        for cid in ("chart-equity", "chart-position", "chart-reward", "chart-drawdown"):
            painted = page.eval_on_selector(
                f"#{cid}",
                "el => { const c = el.getContext('2d');"
                "const d = c.getImageData(0,0,el.width,el.height).data;"
                "let n=0; for (let i=3;i<d.length;i+=4) if (d[i]>0) n++; return n; }",
            )
            check(f"{cid} painted", painted > 500, f"{painted} px")

        before = page.inner_text("#read-equity")
        page.eval_on_selector(
            "#pg-scrub",
            "el => { el.value = Math.floor(el.max/2); el.dispatchEvent(new Event('input')); }",
        )
        time.sleep(0.4)
        check("scrubbing updates the readout", page.inner_text("#read-equity") != before)
        check("step label tracks the cursor", "/" in page.inner_text("#pg-step"))
        check("no page errors", not errors, "; ".join(errors[:2]))

        # ── Agent X-Ray ────────────────────────────────────────────────────
        print("\nAgent X-Ray")
        page.click('.lab-tab[data-panel="xray"]')
        page.wait_for_selector("#xr-body:not([hidden])", timeout=20000)
        page.wait_for_function(
            "() => document.querySelectorAll('.xr-feat').length > 0", timeout=20000
        )
        time.sleep(0.6)

        n_feat = page.eval_on_selector_all(".xr-feat", "els => els.length")
        check("every feature is listed", n_feat == 28, f"{n_feat} rows")

        groups = page.eval_on_selector_all(".xr-group h4", "els => els.map(e => e.textContent)")
        check("groups keep their semantic order", groups[0].strip().upper() == "MOMENTUM",
              " > ".join(g.strip() for g in groups[:3]))

        dims = page.inner_text("#xr-window-dims")
        check("observation is fully accounted for", "563" in dims, dims.strip())

        painted = page.eval_on_selector(
            "#xr-heatmap",
            "el => { const c = el.getContext('2d');"
            "const d = c.getImageData(0,0,el.width,el.height).data;"
            "let n=0; for (let i=3;i<d.length;i+=4) if (d[i]>0) n++; return n; }",
        )
        check("feature-window heatmap painted", painted > 5000, f"{painted} px")

        # The critic genuinely is not in the deployed archives: it must be
        # declared absent, never filled in with a plausible number.
        absent = page.eval_on_selector("#xr-value-node", "el => el.classList.contains('is-absent')")
        value_text = page.inner_text("#xr-value").strip().lower()
        check("missing critic is declared, not faked",
              absent and "not exported" in value_text, value_text)

        # Synthetic paths have no reference index, so 4 features are inert.
        inert = page.eval_on_selector_all(".xr-feat.is-inert", "els => els.length")
        check("inert cross-asset features are marked", inert == 4, f"{inert} marked")

        act_before = page.inner_text("#xr-action")
        page.eval_on_selector(
            "#xr-scrub",
            "el => { el.value = Math.floor(el.max/3); el.dispatchEvent(new Event('input')); }",
        )
        time.sleep(1.2)
        check("X-Ray scrub moves the decision", page.inner_text("#xr-action") != act_before)

        # ── Can you break the agent? ───────────────────────────────────────
        print("\nCan you break the agent?")
        page.click('.lab-tab[data-panel="generalization"]')
        page.wait_for_function(
            "() => document.querySelectorAll('.ab-card').length === 2", timeout=20000
        )
        time.sleep(0.5)

        check("both agents rendered",
              page.eval_on_selector_all(".ab-card", "els => els.length") == 2)
        dots = page.eval_on_selector_all(".seed-dot", "els => els.length")
        check("per-seed points shown", dots == 20, f"{dots} dots (2 arms x 2 splits x 5 seeds)")
        check("verdict names the failure",
              "lies" in page.inner_text("#gen-verdict").lower(),
              page.inner_text("#gen-verdict")[:70].strip())
        check("ablation is labelled precomputed",
              page.is_visible(".tag-precomputed"))
        check("regeneration command shown",
              "ablation_multiseed" in page.inner_text("#gen-receipt"))

        # Paired significance test, recomputed live.
        page.click("#gen-test")
        page.wait_for_selector("#gen-stat-out .stat-out", timeout=30000)
        time.sleep(0.5)
        stat_text = page.inner_text("#gen-stat-out")
        check("paired test ran", "p-value" in stat_text.lower())
        check("resolution floor is disclosed",
              "cannot reach significance" in stat_text.lower() or "never fall below" in stat_text.lower(),
              "floor explained")

        # Live shift test against the deployed policy.
        page.eval_on_selector("#shift-seeds",
                              "el => { el.value = 2; el.dispatchEvent(new Event('input')); }")
        page.click("#shift-run")
        page.wait_for_selector("#shift-out:not([hidden])", timeout=180000)
        time.sleep(0.5)
        n_rows = page.eval_on_selector_all(".rbar-row", "els => els.length")
        check("every regime evaluated", n_rows == 5, f"{n_rows} regimes")
        check("regimes report realised autocorrelation",
              "autocorr" in page.inner_text("#shift-bars").lower())

        page.click('.lab-tab[data-panel="playground"]')
        time.sleep(0.3)

        if shot:
            page.screenshot(path=shot)
            print(f"  screenshot -> {shot}")

        # ── 2. the lab with no backend ─────────────────────────────────────
        # The important half: nothing may be rendered from thin air.
        print("\nUnreachable backend")
        off_errors: list[str] = []
        off = browser.new_page(viewport={"width": 1280, "height": 900})
        off.on("pageerror", lambda e: off_errors.append(str(e)))
        off.route(
            "**/config.js",
            lambda route: route.fulfill(
                status=200,
                content_type="application/javascript",
                body="window.RL_API = 'http://127.0.0.1:9;';",
            ),
        )
        off.goto(url, wait_until="load")
        time.sleep(3.0)

        check("status shows unreachable",
              "UNREACHABLE" in off.inner_text("#lab-api-status").upper(),
              off.inner_text("#lab-api-status").strip())
        check("run button disabled", off.eval_on_selector("#pg-run", "el => el.disabled"))
        check("error is surfaced", off.eval_on_selector("#pg-error", "el => !el.hidden"))
        check("no results rendered", off.eval_on_selector("#pg-output", "el => el.hidden"))
        check("no page errors offline", not off_errors, "; ".join(off_errors[:2]))

        browser.close()

    httpd.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api", default="http://127.0.0.1:8000", help="Backend base URL.")
    ap.add_argument("--port", type=int, default=8777, help="Port to serve docs/ on.")
    ap.add_argument("--screenshot", default=None, help="Optional screenshot path.")
    args = ap.parse_args()

    try:
        run(args.api, args.port, args.screenshot)
    except ImportError:
        sys.exit("Playwright is required: pip install playwright && playwright install chromium")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print("All lab smoke checks passed.")


if __name__ == "__main__":
    main()
