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


# Chromium's default multi-process model needs headroom this test does not always
# have — on a loaded machine the launch fails with a bare "spawn UNKNOWN", or the
# driver dies mid-run as "connection closed". The page under test is a handful of
# canvases with no cross-origin content, so collapsing it into one process with a
# capped JS heap costs nothing and makes the run survive a busy desktop.
LEAN_ARGS = [
    "--single-process",
    "--no-zygote",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-software-rasterizer",
    "--disable-extensions",
    "--renderer-process-limit=1",
    "--js-flags=--max-old-space-size=256",
]


def serve(port: int):
    """Serve docs/ locally so the page runs over http rather than file://."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=DOCS)
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


STALE_PORT = 8099


def serve_stale_backend(port: int = STALE_PORT):
    """Stand in for the dashboard-era API: ``/health`` answers, the lab does not.

    Reproduces the state the deployed backend is in between merging the lab and
    redeploying it — the case where a naive "is anything there?" check would
    wrongly report the lab as working.
    """
    import json

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's naming
            if self.path.startswith("/health"):
                self._send(200, {"status": "ok",
                                 "policies": ["crypto", "stock"],
                                 "version": "0.1.0"})
            else:
                self._send(404, {"error": "Not Found"})

        def log_message(self, *args):  # keep the smoke output readable
            pass

    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def run(api: str, port: int, shot: str | None) -> None:
    from playwright.sync_api import sync_playwright

    httpd = serve(port)
    url = f"http://127.0.0.1:{port}/index.html#lab"

    with sync_playwright() as p:
        browser = p.chromium.launch(args=LEAN_ARGS)

        # ── 1. the lab against a live backend ──────────────────────────────
        print("\nLive backend")
        errors: list[str] = []
        page = browser.new_page(viewport={"width": 1280, "height": 900})
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

        # Wait for the handshake rather than sleeping a fixed interval: a cold
        # free-tier backend can take several seconds, and a fixed sleep turns
        # that into a flaky failure that looks like a real one.
        page.wait_for_function(
            "() => { const el = document.getElementById('lab-api-status');"
            "return el && !/connecting/i.test(el.textContent); }",
            timeout=60000,
        )

        check("lab view routes from #lab",
              page.eval_on_selector("#view-lab", "el => el.classList.contains('active')"))
        check("backend reports live",
              "LIVE" in page.inner_text("#lab-api-status").upper(),
              page.inner_text("#lab-api-status").strip())

        # ── Signal or Noise? (the default panel) ───────────────────────────
        print("\nSignal or Noise?")
        page.wait_for_function(
            "() => document.querySelectorAll('.pc-card').length > 0", timeout=60000
        )
        n_cards = page.eval_on_selector_all(".pc-card", "els => els.length")
        check("quiz charts rendered", n_cards == 12, f"{n_cards} charts")
        check("the prompt comes from the backend",
              len(page.inner_text("#pc-prompt")) > 20, page.inner_text("#pc-prompt")[:60])

        painted = page.eval_on_selector(
            "#pc-cv-0",
            "el => { const c = el.getContext('2d');"
            "const d = c.getImageData(0,0,el.width,el.height).data;"
            "let n=0; for (let i=3;i<d.length;i+=4) if (d[i]>0) n++; return n; }",
        )
        check("sparklines painted", painted > 100, f"{painted} px")
        # The answer key must never be in the page — that is the whole design.
        check("no answers in the served payload",
              "positive_class" not in page.eval_on_selector(
                  "#pc-grid", "el => el.innerHTML"))
        check("submit is blocked until every chart is called",
              page.eval_on_selector("#pc-submit", "el => el.disabled"))

        # Answer everything "trending" — a fixed strategy, so the score is
        # deterministic: exactly the n/2 charts that really are trending.
        page.eval_on_selector_all(
            '.pc-seg button[data-val="1"]', "els => els.forEach(b => b.click())"
        )
        check("progress tracks the answers",
              "12 of 12" in page.inner_text("#pc-progress"), page.inner_text("#pc-progress"))
        page.click("#pc-submit")
        page.wait_for_selector("#pc-result:not([hidden])", timeout=30000)
        time.sleep(0.4)

        score = page.inner_text(".pc-score-n")
        check("balanced classes make an all-one-answer sheet score n/2",
              score.startswith("6"), score.replace("\n", ""))
        check("a verdict is shown", len(page.inner_text(".pc-verdict")) > 40)
        check("the power table is rendered",
              page.eval_on_selector_all(".pc-power tbody tr", "els => els.length") == 3)
        check("the statistical reference is shown",
              "of 12" in page.inner_text(".pc-ref-score"), page.inner_text(".pc-ref-score"))
        check("the truth is revealed per chart",
              page.eval_on_selector_all(".pc-card.is-right, .pc-card.is-wrong",
                                        "els => els.length") == 12)
        check("the design is on the receipt",
              page.eval_on_selector_all("#pc-receipt dt", "els => els.length") >= 6)

        # ── Agent Playground ───────────────────────────────────────────────
        print("\nAgent Playground")
        page.click('.lab-tab[data-panel="playground"]')
        page.wait_for_selector("#pg-mode", state="visible", timeout=20000)
        page.click('#pg-mode button[data-val="synthetic"]')
        page.wait_for_function(
            "() => document.getElementById('pg-regime').options.length >= 4", timeout=30000
        )
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

        # ── What is it reading? (occlusion attribution) ────────────────────
        print("\nWhat is it reading?")
        page.click("#attr-run")
        page.wait_for_selector("#attr-out:not([hidden])", timeout=60000)
        time.sleep(0.5)

        n_rows = page.eval_on_selector_all("#attr-features .attr-row", "els => els.length")
        check("every feature is attributed", n_rows == 28, f"{n_rows} rows")
        check("account scalars are attributed separately",
              page.eval_on_selector_all("#attr-account .attr-row", "els => els.length") == 3)
        check("feature groups are summarised",
              page.eval_on_selector_all(".attr-chip", "els => els.length") == 8)

        # Ranked strongest-first, and the top bar must be a real effect.
        vals = page.eval_on_selector_all(
            "#attr-features .attr-val", "els => els.map(e => parseFloat(e.textContent))"
        )
        check("attribution is ranked", vals == sorted(vals, reverse=True))
        check("occlusion moves the policy", vals[0] > 0.01, f"top delta {vals[0]}")

        check("structurally inert features are named",
              "rel_return_5" in page.inner_text("#attr-dead"),
              page.inner_text("#attr-dead")[:60])
        caveats = page.inner_text("#attr-caveats").lower()
        check("the method's limits travel with the chart",
              "not causal" in caveats and "correlated" in caveats)

        # Switching scope must re-render the measurement already taken, not
        # quietly run a different one behind the same label.
        page.select_option("#attr-scope", "local")
        time.sleep(0.5)
        check("the local view is labelled as a single bar",
              "one point in" in page.inner_text("#attr-caveats"))

        # ── What if? (counterfactual) ──────────────────────────────────────
        print("\nWhat if?")
        page.click("#wi-run")
        page.wait_for_selector("#wi-out .wi-row", timeout=60000)
        time.sleep(0.5)

        n_alt = page.eval_on_selector_all("#wi-out .wi-row", "els => els.length")
        check("alternatives evaluated", n_alt >= 5, f"{n_alt} actions")
        check("the agent's own action is among them",
              page.eval_on_selector_all("#wi-out .wi-row.is-agent", "els => els.length") == 1)
        # Long and short from the same state cannot land on the same number.
        vals = page.eval_on_selector_all(
            "#wi-out .wi-row .wi-num", "els => els.map(e => e.textContent.trim())"
        )
        check("outcomes differ by action", len(set(vals)) > 2, f"{len(set(vals))} distinct")
        check("framed as counterfactual, not prediction",
              "does not imply" in page.inner_text("#wi-out").lower())

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

        # ── Real or luck? ──────────────────────────────────────────────────
        print("\nReal or luck?")
        page.click('.lab-tab[data-panel="seeds"]')
        page.wait_for_function(
            "() => document.querySelectorAll('#sd-seeds .seed-row').length > 0", timeout=30000
        )
        time.sleep(0.8)

        n_seeds = page.eval_on_selector_all("#sd-seeds .seed-row", "els => els.length")
        check("every training run listed", n_seeds == 5, f"{n_seeds} runs")
        check("one run is shown beside many",
              page.eval_on_selector_all(".vp-card", "els => els.length") == 2)

        pair = page.inner_text("#sd-pair")
        check("single-seed headline shown", "275" in pair, pair.split("\n")[1] if "\n" in pair else pair[:40])
        check("interval reported as spanning zero", "spans zero" in pair.lower())
        check("verdict states it does not survive",
              "does not survive" in page.inner_text("#sd-verdict").lower())

        painted = page.eval_on_selector(
            "#sd-hist",
            "el => { const c = el.getContext('2d');"
            "const d = c.getImageData(0,0,el.width,el.height).data;"
            "let n=0; for (let i=3;i<d.length;i+=4) if (d[i]>0) n++; return n; }",
        )
        check("bootstrap histogram painted", painted > 2000, f"{painted} px")

        # Changing the design must actually change the numbers.
        before = page.inner_text("#sd-pair")
        page.select_option("#sd-conf", "0.8")
        page.click("#sd-run")
        time.sleep(1.5)
        check("confidence level changes the interval", page.inner_text("#sd-pair") != before)

        paper = page.inner_text("#sd-paper")
        check("published ticker-axis test shown", "p-value" in paper.lower())

        # ── Research notebook ──────────────────────────────────────────────
        print("\nResearch notebook")
        page.click('.lab-tab[data-panel="notebook"]')
        page.wait_for_selector("#nb-list .nb-row", timeout=20000)
        time.sleep(0.5)

        n_before = page.eval_on_selector_all("#nb-list .nb-row", "els => els.length")
        check("earlier runs are listed", n_before >= 3, f"{n_before} experiments")
        check("storage is declared ephemeral",
              "ephemeral" in page.inner_text("#nb-storage").lower())
        check("runs with no stated question say so",
              page.eval_on_selector_all("#nb-list .nb-q.is-unstated", "els => els.length") > 0)

        # Run one with a question of our own.
        question = "Does the agent survive mean reversion?"
        page.fill("#nb-question", question)
        page.select_option("#nb-regime", "mean_reversion")
        page.click("#nb-run")
        page.wait_for_selector("#nb-detail-card:not([hidden])", timeout=90000)
        time.sleep(0.8)

        detail = page.inner_text("#nb-detail")
        # inner_text returns *rendered* text and .receipt dt is uppercased in CSS,
        # so compare case-insensitively rather than against the source casing.
        lower = detail.lower()
        check("the question is recorded verbatim", question in detail)
        check("receipt names the dataset hash", "dataset hash" in lower)
        check("receipt states critic availability", "critic" in lower)
        n_after = page.eval_on_selector_all("#nb-list .nb-row", "els => els.length")
        check("history grew by one", n_after == n_before + 1, f"{n_before} -> {n_after}")

        # Reproducing must land on the same numbers.
        first_id = page.inner_text("#nb-detail-title").replace("Experiment", "").strip()
        finding_before = page.eval_on_selector(
            f'#nb-list .nb-row[data-id="{first_id}"] .nb-find span',
            "el => el.textContent.trim()",
        )
        page.click("#nb-reproduce")
        page.wait_for_function(
            f"() => !document.getElementById('nb-detail-title').textContent.includes('{first_id}')",
            timeout=90000,
        )
        time.sleep(1.0)
        repro_id = page.inner_text("#nb-detail-title").replace("Experiment", "").strip()
        check("reproduction is a new experiment", repro_id != first_id,
              f"{first_id} -> {repro_id}")
        check("reproduction cites the original",
              first_id in page.inner_text("#nb-detail"))
        finding_after = page.eval_on_selector(
            f'#nb-list .nb-row[data-id="{repro_id}"] .nb-find span',
            "el => el.textContent.trim()",
        )
        check("reproduction matches the original result",
              finding_before == finding_after, f"{finding_before} vs {finding_after}")

        page.click('.lab-tab[data-panel="playground"]')
        time.sleep(0.3)

        if shot:
            page.screenshot(path=shot)
            print(f"  screenshot -> {shot}")

        # Finish with this browser entirely before starting the next phase, so
        # only one is ever alive — two open at once is a real memory spike on a
        # modest machine. Under --single-process, closing the last page also
        # closes the browser, so the offline phase gets a fresh launch rather
        # than a second page.
        browser.close()

        # ── 2. the lab with no backend ─────────────────────────────────────
        # The important half: nothing may be rendered from thin air.
        print("\nUnreachable backend")
        browser = p.chromium.launch(args=LEAN_ARGS)
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

        # ── 3. home page + routing, with no backend at all ─────────────────
        # The dashboard is meant to work fully static, and the home page's deep
        # links into the lab must land on the right panel without one.
        print("\nHome page and routing (static)")
        # Its own browser again. Phase 2 deliberately points the page at an
        # unreachable API, and under --single-process a renderer that goes down
        # takes the whole browser with it — so reusing it here is not safe.
        browser.close()
        browser = p.chromium.launch(args=LEAN_ARGS)
        home_errors: list[str] = []
        home = browser.new_page(viewport={"width": 1280, "height": 900})
        home.on("pageerror", lambda e: home_errors.append(str(e)))
        home.route(
            "**/config.js",
            lambda route: route.fulfill(
                status=200,
                content_type="application/javascript",
                body="window.RL_API = '';",
            ),
        )
        home.goto(f"http://127.0.0.1:{port}/index.html", wait_until="load")
        time.sleep(1.5)

        check("home view is the default",
              home.eval_on_selector("#view-home", "el => el.classList.contains('active')"))
        check("primary CTA leads to the lab",
              home.eval_on_selector(".hero-cta .btn-primary", "el => el.getAttribute('href')")
              == "#lab")
        n_cards = home.eval_on_selector_all(".lab-card", "els => els.length")
        check("every panel is advertised", n_cards == 6, f"{n_cards} cards")
        check("each card declares whether it is live",
              home.eval_on_selector_all(".lab-card-tag", "els => els.length") == 6)
        check("the training caveat is on the home page",
              "training" in home.inner_text(".lab-strip-note").lower())

        # A deep link has to move the view *and* select the panel.
        home.click('.lab-card[data-panel="seeds"]')
        time.sleep(1.0)
        check("deep link switches view",
              home.eval_on_selector("#view-lab", "el => el.classList.contains('active')"))
        check("deep link selects the panel",
              home.eval_on_selector("#panel-seeds", "el => el.classList.contains('active')"))
        check("deep link is reflected in the URL",
              "seeds" in home.evaluate("location.hash"), home.evaluate("location.hash"))
        check("selected tab is announced",
              home.eval_on_selector("#tab-seeds", "el => el.getAttribute('aria-selected')")
              == "true")

        # ARIA tabs keyboard pattern.
        home.eval_on_selector("#tab-seeds", "el => el.focus()")
        home.keyboard.press("ArrowRight")
        time.sleep(0.4)
        check("arrow keys move between tabs",
              home.eval_on_selector("#panel-notebook", "el => el.classList.contains('active')"))
        home.keyboard.press("Home")
        time.sleep(0.4)
        check("Home key jumps to the first tab",
              home.eval_on_selector("#panel-perception", "el => el.classList.contains('active')"))
        check("only the selected tab is in the tab order",
              home.eval_on_selector_all(".lab-tab", "els => els.filter(e => e.tabIndex === 0).length")
              == 1)

        home.go_back()
        time.sleep(0.8)
        check("browser back navigates", home.eval_on_selector_all(
            ".view.active", "els => els.length") == 1)
        check("no page errors on the static site", not home_errors, "; ".join(home_errors[:2]))

        # ── 4. a backend that is up but predates the lab ───────────────────
        # The dashboard-era API serves /health too. If the page trusted that it
        # would announce "API live" and then 404 on every panel — a reassuring
        # claim followed by a broken experience, which is the exact failure mode
        # this project exists to argue against.
        print("\nOutdated backend (health OK, lab endpoints missing)")
        stale = serve_stale_backend()
        try:
            browser.close()
            browser = p.chromium.launch(args=LEAN_ARGS)
            old_errors: list[str] = []
            old = browser.new_page(viewport={"width": 1280, "height": 900})
            old.on("pageerror", lambda e: old_errors.append(str(e)))
            old.route(
                "**/config.js",
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/javascript",
                    body=f"window.RL_API = 'http://127.0.0.1:{STALE_PORT}';",
                ),
            )
            old.goto(url, wait_until="load")
            old.wait_for_function(
                "() => { const el = document.getElementById('lab-api-status');"
                "return el && !/connecting/i.test(el.textContent); }",
                timeout=45000,
            )
            time.sleep(1.0)

            pill = old.inner_text("#lab-api-status").strip()
            check("does not claim the API is live", "api live" not in pill.lower(), pill)
            check("names the real problem", "too old" in pill.lower(), pill)
            check("distinguishes stale from unreachable",
                  "is-stale" in old.eval_on_selector("#lab-api-status", "el => el.className"))
            check("run button disabled", old.eval_on_selector("#pg-run", "el => el.disabled"))
            check("nothing is rendered", old.eval_on_selector("#pg-output", "el => el.hidden"))
            check("no page errors against an old backend", not old_errors,
                  "; ".join(old_errors[:2]))
        finally:
            stale.shutdown()

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
