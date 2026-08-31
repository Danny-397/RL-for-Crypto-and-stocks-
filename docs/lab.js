/* RL·Trader LAB — interactive research laboratory.
 *
 * Every number rendered here comes from the backend at window.RL_API. There is
 * no client-side simulation and no fallback data: if the API is unreachable the
 * lab says so and renders nothing, because a plausible-looking fake would be
 * indistinguishable from a real experiment to a visitor.
 *
 * Structure
 *   api        — fetch helpers + experiment polling
 *   fmt        — formatting
 *   Chart      — small canvas plotting layer with a shared scrub cursor
 *   Playground — configure, run, and inspect a single episode
 */
(function () {
  "use strict";

  const API = (window.RL_API || "").replace(/\/+$/, "");
  const $ = (id) => document.getElementById(id);

  /* ── formatting ─────────────────────────────────────────── */
  const fmt = {
    pct: (v, d = 2) => (v >= 0 ? "+" : "") + (v * 100).toFixed(d) + "%",
    num: (v, d = 2) => (v == null || !isFinite(v) ? "—" : v.toFixed(d)),
    money: (v) =>
      v == null || !isFinite(v)
        ? "—"
        : "$" + Math.round(v).toLocaleString("en-US"),
    signed: (v, d = 3) => (v >= 0 ? "+" : "") + v.toFixed(d),
    cls: (v) => (v > 0 ? "pos" : v < 0 ? "neg" : "neutral"),
  };

  /* ── API ────────────────────────────────────────────────── */
  const api = {
    ok: Boolean(API),

    async get(path) {
      const r = await fetch(API + path);
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
      return body;
    },

    async post(path, payload) {
      const r = await fetch(API + path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
      return body;
    },

    /* Create an experiment and poll it to completion.
     * onProgress receives the real server-reported progress, never a fake ramp. */
    async runExperiment(payload, onProgress, timeoutMs = 120000) {
      const created = await api.post("/api/experiments", payload);
      const id = created.id;
      const deadline = Date.now() + timeoutMs;

      while (Date.now() < deadline) {
        const body = await api.get(`/api/experiments/${id}`);
        if (onProgress) onProgress(body);
        if (body.status === "done") return body;
        if (body.status === "error") throw new Error(body.error || "experiment failed");
        await new Promise((res) => setTimeout(res, 350));
      }
      throw new Error("experiment timed out");
    },
  };

  /* ── canvas charts ──────────────────────────────────────── */
  const COLORS = {
    agent: "#d4ff3f",
    bench: "#36e0ff",
    pos: "#5df2a0",
    neg: "#ff6b6b",
    grid: "rgba(255,255,255,0.06)",
    axis: "rgba(255,255,255,0.28)",
    text: "#8a97a8",
    cursor: "rgba(255,255,255,0.45)",
  };

  /* Size a canvas for the device pixel ratio so lines stay crisp. */
  function fitCanvas(cv, cssHeight) {
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth || cv.parentElement.clientWidth || 600;
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(cssHeight * dpr);
    cv.style.height = cssHeight + "px";
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h: cssHeight };
  }

  /* A stacked chart with a cursor index shared across every chart in a group. */
  function Chart(canvas, opts) {
    const o = Object.assign(
      { height: 150, pad: { l: 54, r: 12, t: 10, b: 18 }, zero: false, kind: "line" },
      opts || {}
    );
    let series = [];
    let cursor = null;

    function bounds() {
      let lo = Infinity;
      let hi = -Infinity;
      series.forEach((s) => {
        s.values.forEach((v) => {
          if (v == null || !isFinite(v)) return;
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        });
      });
      if (!isFinite(lo)) { lo = 0; hi = 1; }
      if (o.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
      if (hi - lo < 1e-12) { hi += 1; lo -= 1; }
      const padv = (hi - lo) * 0.08;
      lo -= padv;
      hi += padv;
      // Headroom must not invent impossible values — a drawdown axis that reads
      // +4% is simply wrong, so a series with a hard bound declares it.
      if (o.clampMax != null) hi = Math.min(hi, o.clampMax);
      if (o.clampMin != null) lo = Math.max(lo, o.clampMin);
      return { lo, hi };
    }

    function draw() {
      if (!series.length) return;
      const { ctx, w, h } = fitCanvas(canvas, o.height);
      const { lo, hi } = bounds();
      const n = Math.max(...series.map((s) => s.values.length));
      const X = (i) => o.pad.l + (n <= 1 ? 0 : (i / (n - 1)) * (w - o.pad.l - o.pad.r));
      const Y = (v) => o.pad.t + (1 - (v - lo) / (hi - lo)) * (h - o.pad.t - o.pad.b);

      ctx.clearRect(0, 0, w, h);

      // horizontal gridlines + value labels
      // A sparkline (grid:false) deliberately shows no axis: in the signal-or-noise
      // test a y-axis would hand over scale information the design works to remove.
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillStyle = COLORS.text;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let g = 0; o.grid !== false && g <= 3; g++) {
        const v = lo + ((hi - lo) * g) / 3;
        const y = Y(v);
        ctx.strokeStyle = COLORS.grid;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(o.pad.l, y);
        ctx.lineTo(w - o.pad.r, y);
        ctx.stroke();
        ctx.fillText(o.fmtY ? o.fmtY(v) : v.toFixed(2), o.pad.l - 7, y);
      }

      // emphasised zero line where sign is meaningful
      if (o.zero && lo < 0 && hi > 0) {
        ctx.strokeStyle = COLORS.axis;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(o.pad.l, Y(0));
        ctx.lineTo(w - o.pad.r, Y(0));
        ctx.stroke();
      }

      series.forEach((s) => {
        if (s.kind === "area") {
          ctx.beginPath();
          s.values.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
          ctx.lineTo(X(s.values.length - 1), Y(o.zero ? 0 : lo));
          ctx.lineTo(X(0), Y(o.zero ? 0 : lo));
          ctx.closePath();
          ctx.fillStyle = s.fill || "rgba(255,107,107,0.16)";
          ctx.fill();
        }
        if (s.kind === "bar") {
          const bw = Math.max(1, (w - o.pad.l - o.pad.r) / s.values.length);
          s.values.forEach((v, i) => {
            if (v == null) return;
            ctx.fillStyle = v >= 0 ? s.color : s.negColor || COLORS.neg;
            const y0 = Y(0);
            const y1 = Y(v);
            ctx.fillRect(X(i) - bw / 2, Math.min(y0, y1), Math.max(bw * 0.86, 1), Math.abs(y1 - y0));
          });
          return;
        }
        ctx.beginPath();
        let started = false;
        s.values.forEach((v, i) => {
          if (v == null || !isFinite(v)) return;
          if (!started) { ctx.moveTo(X(i), Y(v)); started = true; }
          else ctx.lineTo(X(i), Y(v));
        });
        ctx.strokeStyle = s.color;
        ctx.lineWidth = s.width || 1.8;
        if (s.dash) ctx.setLineDash(s.dash);
        ctx.stroke();
        ctx.setLineDash([]);
      });

      // shared scrub cursor
      if (cursor != null && n > 1) {
        const x = X(Math.max(0, Math.min(n - 1, cursor)));
        ctx.strokeStyle = COLORS.cursor;
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(x, o.pad.t);
        ctx.lineTo(x, h - o.pad.b);
        ctx.stroke();
        ctx.setLineDash([]);
        series.forEach((s) => {
          const v = s.values[Math.min(cursor, s.values.length - 1)];
          if (v == null || !isFinite(v) || s.kind === "bar") return;
          ctx.beginPath();
          ctx.arc(x, Y(v), 3, 0, Math.PI * 2);
          ctx.fillStyle = s.color;
          ctx.fill();
        });
      }
    }

    function indexFromEvent(e) {
      const rect = canvas.getBoundingClientRect();
      const n = Math.max(...series.map((s) => s.values.length));
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const x = clientX - rect.left;
      const frac = (x - o.pad.l) / (rect.width - o.pad.l - o.pad.r);
      return Math.max(0, Math.min(n - 1, Math.round(frac * (n - 1))));
    }

    return {
      canvas,
      setSeries(s) { series = s; draw(); },
      setCursor(i) { cursor = i; draw(); },
      redraw: draw,
      indexFromEvent,
      get length() { return series.length ? Math.max(...series.map((s) => s.values.length)) : 0; },
    };
  }

  /* Downsample for plotting while preserving index alignment. */
  function pick(arr, n) {
    if (!arr || arr.length <= n) return arr || [];
    const out = [];
    for (let i = 0; i < n; i++) out.push(arr[Math.round((i / (n - 1)) * (arr.length - 1))]);
    return out;
  }

  /* ── shared UI helpers ──────────────────────────────────── */
  function setStatus(el, text, busy) {
    if (!el) return;
    el.innerHTML = "";
    if (busy) {
      const s = document.createElement("span");
      s.className = "spinner";
      el.appendChild(s);
    }
    el.appendChild(document.createTextNode(text || ""));
  }

  function showError(el, message) {
    if (!el) return;
    el.innerHTML = "";
    if (!message) { el.hidden = true; return; }
    el.hidden = false;
    el.className = "lab-error";
    el.textContent = message;
  }

  function metric(label, value, cls, compare) {
    return (
      `<div class="metric"><span class="k">${label}</span>` +
      `<span class="v ${cls || "neutral"}">${value}</span>` +
      (compare ? `<span class="cmp">${compare}</span>` : "") +
      `</div>`
    );
  }

  /* ── Signal or Noise? ───────────────────────────────────── */
  /* A controlled test of the visitor's own pattern detection.
   *
   * The answer key never reaches the browser: charts arrive unlabelled, and the
   * backend rescores by rebuilding the identical quiz from its seed. Nothing is
   * graded locally, so there is nothing here to read ahead in — and the verdict
   * shown is an exact binomial test computed server-side, not a canned message. */
  const Perception = (function () {
    let quiz = null;      // { params, meta, charts } as served
    let answers = [];     // 0 | 1 | null per chart — the visitor's calls
    let scored = false;
    let tickers = null;   // lazily fetched, only for the real-data condition
    const sparks = [];    // Chart instances, kept so a resize can redraw them

    const CLASS_NAMES = {
      trending: ["Random walk", "Trending"],
      real: ["Reshuffled", "Real"],
    };

    function names() {
      return CLASS_NAMES[(quiz && quiz.meta.positive_class) || "trending"];
    }

    function difficulty() { return $("pc-difficulty").dataset.value; }

    /* The source control means different things in the two conditions, so its
     * label and options are rebuilt rather than reused. */
    async function fillSources() {
      const sel = $("pc-source");
      const label = $("pc-source-label");
      sel.innerHTML = "";
      if (difficulty() === "synthetic") {
        label.textContent = "Trained-on regime";
        [["stock", "Stock agent's regime"], ["crypto", "Crypto agent's regime"]].forEach(
          ([v, t]) => sel.add(new Option(t, v))
        );
        return;
      }
      label.textContent = "Ticker";
      if (!tickers) {
        try { tickers = await api.get("/api/tickers"); }
        catch (err) { tickers = { stock: ["SPY"], crypto: ["BTC-USD"] }; }
      }
      [].concat(tickers.stock || [], tickers.crypto || []).forEach((t) =>
        sel.add(new Option(t, t))
      );
    }

    function params() {
      const n = parseInt($("pc-n").value, 10);
      const src = $("pc-source").value;
      const p = { difficulty: difficulty(), n_charts: n, seed: quiz ? quiz.params.seed : 0 };
      if (p.difficulty === "synthetic") p.market = src;
      else p.ticker = src;
      return p;
    }

    async function load() {
      const p = params();
      // A fresh seed per quiz, so a visitor can repeat the experiment rather than
      // re-take the same one and mistake memory for skill.
      p.seed = Math.floor(Math.random() * 2147483647);
      const qs = Object.keys(p).map((k) => `${k}=${encodeURIComponent(p[k])}`).join("&");

      showError($("pc-error"), null);
      $("pc-result").hidden = true;
      $("pc-actions").hidden = true;
      $("pc-grid").innerHTML = "";
      $("pc-receipt").innerHTML = "";
      setStatus($("pc-status"), "building a fresh quiz…", true);

      try {
        quiz = await api.get(`/api/perception/quiz?${qs}`);
      } catch (err) {
        setStatus($("pc-status"), "");
        showError($("pc-error"), `Could not build a quiz: ${err.message}`);
        return;
      }
      answers = new Array(quiz.charts.length).fill(null);
      scored = false;
      setStatus($("pc-status"), "");
      $("pc-prompt").textContent = quiz.meta.prompt;
      render();
      $("pc-actions").hidden = false;
      updateProgress();
    }

    function render() {
      const grid = $("pc-grid");
      grid.innerHTML = "";
      sparks.length = 0;
      const [negName, posName] = names();

      quiz.charts.forEach((chart, i) => {
        const card = document.createElement("div");
        card.className = "pc-card";
        card.id = `pc-card-${i}`;
        card.innerHTML =
          `<div class="pc-card-head"><span class="pc-n">${i + 1}</span>` +
          `<span class="pc-truth" id="pc-truth-${i}"></span></div>` +
          `<canvas class="pc-canvas" id="pc-cv-${i}"></canvas>` +
          `<div class="seg pc-seg" data-chart="${i}">` +
          `<button type="button" data-val="1">${posName}</button>` +
          `<button type="button" data-val="0">${negName}</button></div>`;
        grid.appendChild(card);

        const c = Chart($(`pc-cv-${i}`), {
          height: 96,
          grid: false,
          pad: { l: 4, r: 4, t: 8, b: 8 },
        });
        c.setSeries([{ values: chart.prices, color: COLORS.bench, width: 1.4 }]);
        sparks.push(c);
      });

      grid.querySelectorAll(".pc-seg button").forEach((btn) => {
        btn.addEventListener("click", () => {
          if (scored) return;
          const seg = btn.parentElement;
          const idx = parseInt(seg.dataset.chart, 10);
          answers[idx] = parseInt(btn.dataset.val, 10);
          seg.querySelectorAll("button").forEach((b) =>
            b.setAttribute("aria-pressed", String(b === btn))
          );
          updateProgress();
        });
      });
    }

    function updateProgress() {
      const done = answers.filter((a) => a != null).length;
      $("pc-progress").textContent = `${done} of ${answers.length} called`;
      $("pc-submit").disabled = done !== answers.length || scored;
    }

    async function submit() {
      setStatus($("pc-status"), "scoring against the answer key…", true);
      $("pc-submit").disabled = true;
      let out;
      try {
        out = await api.post("/api/perception/score", {
          params: quiz.params,
          answers: answers,
        });
      } catch (err) {
        setStatus($("pc-status"), "");
        showError($("pc-error"), `Scoring failed: ${err.message}`);
        $("pc-submit").disabled = false;
        return;
      }
      setStatus($("pc-status"), "");
      scored = true;
      reveal(out);
      renderResult(out);
      updateProgress();
    }

    /* Mark every card with the truth and the number that decided it. */
    function reveal(out) {
      const [negName, posName] = names();
      out.per_chart.forEach((r) => {
        const card = $(`pc-card-${r.index}`);
        card.classList.add(r.correct ? "is-right" : "is-wrong");
        $(`pc-truth-${r.index}`).innerHTML =
          `<b>${r.truth ? posName : negName}</b> · <span class="pc-ac">ρ₁ = ` +
          `${fmt.signed(r.autocorr_lag1, 3)}</span>`;
        card.querySelectorAll(".pc-seg button").forEach((b) => (b.disabled = true));
      });
    }

    function renderResult(out) {
      const m = out.meta;
      const ref = out.reference;
      const pw = out.power;
      const powerRows = pw.power
        .map(
          (r) =>
            `<tr><td>${Math.round(r.true_accuracy * 100)}% accurate</td>` +
            `<td class="${r.power < 0.5 ? "neg" : "pos"}">${Math.round(r.power * 100)}%</td></tr>`
        )
        .join("");

      $("pc-result").hidden = false;
      $("pc-result").innerHTML =
        `<div class="pc-score">` +
        `<div class="pc-score-main"><span class="pc-score-n">${out.correct}` +
        `<span class="pc-score-d">/${out.n}</span></span>` +
        `<span class="pc-score-lbl">chance is ${out.expected_by_chance}</span></div>` +
        metric("p-value", out.p_value.toFixed(4), out.significant_at_05 ? "pos" : "neutral",
               out.test) +
        metric("Your accuracy", (out.accuracy * 100).toFixed(0) + "%", "neutral") +
        `</div>` +
        `<p class="pc-verdict">${out.verdict}</p>` +

        `<div class="pc-two">` +
        `<div class="pc-box"><h4>What a statistic saw</h4>` +
        `<p class="xr-hint">${ref.description}</p>` +
        `<p class="pc-ref-score">${ref.correct} of ${ref.n} ` +
        `<span class="pc-ref-sub">(p = ${ref.p_value.toFixed(4)})</span></p>` +
        `<p class="xr-hint">${ref.caveat}</p></div>` +

        `<div class="pc-box"><h4>Could this test even detect skill?</h4>` +
        `<table class="pc-power"><thead><tr><th>If you were…</th>` +
        `<th>chance of proving it</th></tr></thead><tbody>${powerRows}</tbody></table>` +
        `<p class="xr-hint">${pw.explanation}</p></div>` +
        `</div>` +

        `<p class="pc-tie">The smallest p-value ${out.n} charts can produce at all is ` +
        `<b>${pw.min_attainable_p.toFixed(4)}</b> — a hard floor set by the design, not by ` +
        `the data. The project's seed-level tests hit exactly this wall: with 5 seeds a ` +
        `sign-flip test cannot go below p = 0.0625, so it can never reach significance ` +
        `however large the effect.</p>`;

      const overlap = m.realised.classes_overlap
        ? "yes — at this length the classes are not cleanly separated"
        : "no — the classes separated cleanly this time";
      $("pc-receipt").innerHTML =
        [
          ["Condition", m.difficulty === "synthetic"
            ? `synthetic · φ = ${m.signal_phi} vs φ = 0`
            : `real · ${m.ticker} vs its own reshuffled returns`],
          ["Design", m.design],
          ["Normalisation", m.normalisation],
          ["Measured ρ₁ (signal / control)",
           `${fmt.signed(m.realised.mean_autocorr_signal, 3)} / ` +
           `${fmt.signed(m.realised.mean_autocorr_control, 3)}`],
          ["Classes overlap?", overlap],
          ["Bars per chart", String(m.bars_per_chart)],
          ["Quiz seed", String(m.seed)],
        ]
          .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`)
          .join("");
    }

    function bindSeg(id, onChange) {
      const seg = $(id);
      seg.querySelectorAll("button").forEach((btn) => {
        btn.addEventListener("click", () => {
          seg.dataset.value = btn.dataset.val;
          seg.querySelectorAll("button").forEach((b) =>
            b.setAttribute("aria-pressed", String(b === btn))
          );
          onChange(btn.dataset.val);
        });
      });
    }

    async function init() {
      if (!api.ok) return;
      bindSeg("pc-difficulty", async () => { await fillSources(); load(); });
      $("pc-n").addEventListener("input", (e) => {
        $("pc-n-val").textContent = e.target.value;
      });
      $("pc-n").addEventListener("change", load);
      $("pc-source").addEventListener("change", load);
      $("pc-new").addEventListener("click", load);
      $("pc-submit").addEventListener("click", submit);
      window.addEventListener("resize", () => sparks.forEach((c) => c.redraw()));
      // A hidden canvas has zero width, so anything drawn while the panel was
      // inactive comes back blank. Redraw on the way in.
      window.addEventListener("lab:panel", (e) => {
        if (e.detail.panel === "perception") sparks.forEach((c) => c.redraw());
      });
      await fillSources();
      await load();
    }

    return { init, redraw: () => sparks.forEach((c) => c.redraw()) };
  })();

  /* ── Playground ─────────────────────────────────────────── */
  const Playground = (function () {
    let meta = null;
    let trace = null;
    let charts = {};
    let cursor = 0;

    const el = {};

    function cacheEls() {
      [
        "pg-market", "pg-mode", "pg-ticker", "pg-regime", "pg-seed",
        "pg-capital", "pg-cost", "pg-cost-val", "pg-slip", "pg-slip-val",
        "pg-reward", "pg-short", "pg-run", "pg-status", "pg-bar", "pg-error",
        "pg-empty", "pg-output", "pg-metrics", "pg-scrub", "pg-step",
        "pg-receipt", "pg-notes", "pg-hist-row", "pg-syn-row",
      ].forEach((id) => (el[id.replace(/^pg-/, "").replace(/-(.)/g, (m, c) => c.toUpperCase())] = $(id)));
    }

    function config() {
      const mode = el.mode.dataset.value || "historical";
      const cfg = {
        market: el.market.dataset.value || "stock",
        mode,
        reward: el.reward.dataset.value || "return",
        initial_balance: Number(el.capital.value) || 100000,
        transaction_cost: Number(el.cost.value) / 10000,
        slippage: Number(el.slip.value) / 10000,
        allow_short: el.short.dataset.value !== "off",
      };
      if (mode === "historical") cfg.ticker = (el.ticker.value || "AAPL").trim().toUpperCase();
      else {
        cfg.regime = el.regime.value;
        cfg.seed = Number(el.seed.value) || 0;
      }
      return cfg;
    }

    async function run() {
      showError(el.error, null);
      el.run.disabled = true;
      el.bar.firstElementChild.style.width = "0%";
      setStatus(el.status, "creating experiment…", true);

      try {
        const body = await api.runExperiment(
          { kind: "rollout", config: config() },
          (b) => {
            const pctDone = Math.round((b.progress || 0) * 100);
            el.bar.firstElementChild.style.width = pctDone + "%";
            setStatus(el.status, `${b.id} · ${b.stage} · ${pctDone}%`, true);
          }
        );
        trace = body.result;
        cursor = trace.steps.length - 1;
        render(body);
        setStatus(el.status, `${body.id} · complete · ${body.elapsed_sec}s`, false);
      } catch (err) {
        showError(el.error, String(err.message || err));
        setStatus(el.status, "", false);
      } finally {
        el.run.disabled = false;
      }
    }

    function render(body) {
      el.empty.hidden = true;
      el.output.hidden = false;

      const steps = trace.steps;
      const N = Math.min(steps.length, 480);
      const idx = (i) => Math.round((i / (N - 1)) * (steps.length - 1));

      const eq = pick(trace.equity_curve, N);
      const bench = pick(trace.bench_curve, N);
      const position = pick(steps.map((s) => s.position_after), N);
      const reward = pick(steps.map((s) => s.reward), N);
      const dd = pick(steps.map((s) => -s.drawdown), N);

      charts.equity.setSeries([
        { values: eq, color: COLORS.agent, width: 2 },
        { values: bench, color: COLORS.bench, width: 1.4, dash: [4, 3] },
      ]);
      charts.position.setSeries([
        { values: position, color: COLORS.pos, kind: "bar", negColor: COLORS.neg },
      ]);
      charts.reward.setSeries([
        { values: reward, color: COLORS.pos, kind: "bar", negColor: COLORS.neg },
      ]);
      charts.drawdown.setSeries([
        { values: dd, color: COLORS.neg, kind: "area", fill: "rgba(255,107,107,0.18)", width: 1.2 },
      ]);

      charts._map = idx;
      charts._n = N;

      el.scrub.max = String(steps.length - 1);
      el.scrub.value = String(cursor);
      renderMetrics();
      renderReceipt(body);
      setCursor(cursor);
      window.dispatchEvent(new CustomEvent("lab:trace", { detail: { body, trace } }));
    }

    function renderMetrics() {
      const m = trace.metrics;
      const b = trace.bench_metrics;
      const excess = m.total_return - b.total_return;
      el.metrics.innerHTML =
        metric("Agent return", fmt.pct(m.total_return), fmt.cls(m.total_return),
               `buy &amp; hold ${fmt.pct(b.total_return)}`) +
        metric("Excess", fmt.pct(excess), fmt.cls(excess), "vs benchmark") +
        metric("Sharpe", fmt.num(m.sharpe), fmt.cls(m.sharpe), `b&amp;h ${fmt.num(b.sharpe)}`) +
        metric("Max drawdown", fmt.pct(-m.max_drawdown, 1), "neg",
               `b&amp;h ${fmt.pct(-b.max_drawdown, 1)}`) +
        metric("Final equity", fmt.money(m.final_equity), fmt.cls(m.total_return),
               `from ${fmt.money(trace.equity_curve[0])}`) +
        metric("Bars", String(trace.n_steps), "neutral", "held-out steps");
    }

    function renderReceipt(body) {
      const meta_ = trace.meta || {};
      const r = body.receipt || {};
      const p = (r.provenance || {}).policy || {};
      const env = (r.provenance || {}).env || {};
      const tag = meta_.synthetic
        ? '<span class="tag-synthetic">Synthetic</span>'
        : '<span class="tag-real">Real market data</span>';

      const rows = [
        ["Experiment", body.id],
        ["Source", `${tag} ${meta_.ticker || meta_.label || ""}`],
        ["Bars", meta_.bars],
        ["Dataset hash", meta_.dataset_hash],
        ["Policy", `${p.name || "—"} · ${p.sha256 || "—"}`],
        ["Critic head", p.has_value_head ? "exported" : "not in archive"],
        ["Obs dim", trace.obs_dim],
        ["Cost / slippage", `${(env.transaction_cost * 1e4).toFixed(1)} bps / ${(env.slippage * 1e4).toFixed(1)} bps`],
        ["Reward", env.reward_kind],
        ["Code version", r.code_version || "unknown"],
        ["Run at", r.created_at_utc],
      ];
      el.receipt.innerHTML = rows
        .filter(([, v]) => v !== undefined && v !== null && v !== "")
        .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`)
        .join("");

      let notes = `<div class="caveat">${trace.inference_note}</div>`;
      if (trace.value_note) notes += `<div class="caveat">${trace.value_note}</div>`;
      if (meta_.synthetic && meta_.realised) {
        notes +=
          `<div class="caveat">Realised on this path: lag-1 return autocorrelation ` +
          `<b>${fmt.signed(meta_.realised.return_autocorr_lag1)}</b>, annualised volatility ` +
          `<b>${(meta_.realised.annualised_vol * 100).toFixed(0)}%</b>. Measured, not assumed.</div>`;
      }
      el.notes.innerHTML = notes;
    }

    function setCursor(i) {
      if (!trace) return;
      cursor = Math.max(0, Math.min(trace.steps.length - 1, i));
      const plotIdx = Math.round((cursor / (trace.steps.length - 1)) * (charts._n - 1));
      ["equity", "position", "reward", "drawdown"].forEach((k) => charts[k].setCursor(plotIdx));

      const s = trace.steps[cursor];
      el.step.textContent = `step ${cursor + 1} / ${trace.steps.length}` + (s.date ? ` · ${s.date}` : "");
      el.scrub.value = String(cursor);

      $("read-equity").innerHTML =
        `${fmt.money(s.equity)} <span class="${fmt.cls(s.equity - trace.equity_curve[0])}">` +
        `${fmt.pct(s.equity / trace.equity_curve[0] - 1)}</span>`;
      $("read-position").innerHTML =
        `<span class="${fmt.cls(s.position_after)}">${fmt.signed(s.position_after, 2)}</span>` +
        ` <span style="color:var(--faint)">from ${fmt.signed(s.position_before, 2)}</span>`;
      $("read-reward").innerHTML =
        `<span class="${fmt.cls(s.reward)}">${fmt.signed(s.reward, 5)}</span>`;
      $("read-drawdown").innerHTML =
        `<span class="neg">${fmt.pct(-s.drawdown, 1)}</span>`;

      window.dispatchEvent(new CustomEvent("lab:cursor", { detail: { step: cursor, record: s } }));
    }

    function bindSegment(node, onChange) {
      node.addEventListener("click", (e) => {
        const btn = e.target.closest("button[data-val]");
        if (!btn) return;
        node.dataset.value = btn.dataset.val;
        node.querySelectorAll("button").forEach((b) =>
          b.setAttribute("aria-pressed", String(b === btn))
        );
        if (onChange) onChange(btn.dataset.val);
      });
    }

    async function loadMeta() {
      try {
        meta = await api.get("/api/meta");
        const regimes = await api.get("/api/regimes");
        el.regime.innerHTML = regimes.regimes
          .map((r) => `<option value="${r.key}" title="${r.description}">${r.label}</option>`)
          .join("");
        return meta;
      } catch (err) {
        showError(el.error, `Backend unreachable: ${err.message}. The lab needs the API — nothing is simulated in the browser.`);
        el.run.disabled = true;
        return null;
      }
    }

    function syncMode(mode) {
      el.histRow.hidden = mode !== "historical";
      el.synRow.hidden = mode === "historical";
    }

    function init() {
      cacheEls();
      if (!el.run) return;

      charts.equity = Chart($("chart-equity"), { height: 210, fmtY: (v) => "$" + Math.round(v / 1000) + "k" });
      // Target exposure is bounded to [-1, 1] by the action space.
      charts.position = Chart($("chart-position"), {
        height: 92, zero: true, clampMin: -1, clampMax: 1, fmtY: (v) => v.toFixed(1),
      });
      charts.reward = Chart($("chart-reward"), { height: 92, zero: true, fmtY: (v) => v.toFixed(2) });
      // Drawdown is plotted negative and can never exceed 0.
      charts.drawdown = Chart($("chart-drawdown"), {
        height: 92, zero: true, clampMax: 0, fmtY: (v) => (v * 100).toFixed(0) + "%",
      });

      // Hovering any chart scrubs them all — one cursor, one moment in time.
      Object.keys(charts).forEach((k) => {
        const c = charts[k];
        if (!c || !c.canvas) return;
        const move = (e) => {
          if (!trace) return;
          const plotIdx = c.indexFromEvent(e);
          setCursor(Math.round((plotIdx / (charts._n - 1)) * (trace.steps.length - 1)));
        };
        c.canvas.addEventListener("mousemove", move);
        c.canvas.addEventListener("touchmove", (e) => { e.preventDefault(); move(e); }, { passive: false });
      });

      bindSegment(el.market);
      bindSegment(el.mode, syncMode);
      bindSegment(el.reward);
      bindSegment(el.short);

      el.cost.addEventListener("input", () => (el.costVal.textContent = el.cost.value + " bps"));
      el.slip.addEventListener("input", () => (el.slipVal.textContent = el.slip.value + " bps"));
      el.scrub.addEventListener("input", () => setCursor(Number(el.scrub.value)));
      el.run.addEventListener("click", run);

      window.addEventListener("resize", () => {
        Object.values(charts).forEach((c) => c && c.redraw && c.redraw());
      });

      syncMode("historical");
      loadMeta();
    }

    return { init, get trace() { return trace; }, get cursor() { return cursor; } };
  })();


  /* -- Agent X-Ray ---------------------------------------- */
  /* Shows the causal chain at one bar: what the agent saw, what it decided,
   * what the environment paid, and where that left the book.
   *
   * Policy/action/reward/position come straight from the trace already in
   * memory, so scrubbing is instant. The feature *window* is fetched from
   * /xray on a debounce -- it is 20x28 values per bar and there is no reason
   * to request it faster than a person can read it. */
  const XRay = (function () {
    let trace = null;
    let expId = null;
    let step = 0;
    let pending = null;
    let lastFetched = -1;

    const el = (id) => $(id);

    function onTrace(detail) {
      trace = detail.trace;
      expId = detail.body.id;
      lastFetched = -1;
      el("xr-empty").hidden = true;
      el("xr-body").hidden = false;
      el("xr-scrub").max = String(trace.steps.length - 1);
      setStep(trace.steps.length - 1);
    }

    function setStep(i) {
      if (!trace) return;
      step = Math.max(0, Math.min(trace.steps.length - 1, i));
      el("xr-scrub").value = String(step);
      renderChain();
      scheduleWindow();
    }

    function renderChain() {
      const s = trace.steps[step];
      el("xr-step").textContent =
        "step " + (step + 1) + " / " + trace.steps.length + (s.date ? " · " + s.date : "");
      el("xr-price").textContent = "price " + fmt.money(s.price);

      el("xr-obsdim").textContent = String(trace.obs_dim);
      el("xr-action").innerHTML =
        '<span class="' + fmt.cls(s.action) + '">' + fmt.signed(s.action, 3) + "</span>";
      el("xr-reward").innerHTML =
        '<span class="' + fmt.cls(s.reward) + '">' + fmt.signed(s.reward, 5) + "</span>";
      el("xr-position").innerHTML =
        '<span class="' + fmt.cls(s.position_after) + '">' + fmt.signed(s.position_after, 3) + "</span>";
      el("xr-position-cmp").textContent = "from " + fmt.signed(s.position_before, 3);

      // The critic is shown only when the archive actually contains one.
      const node = el("xr-value-node");
      if (trace.value_available && s.value != null) {
        node.classList.remove("is-absent");
        el("xr-value").textContent = fmt.signed(s.value, 4);
        el("xr-value-cmp").textContent = "critic estimate";
      } else {
        node.classList.add("is-absent");
        el("xr-value").textContent = "not exported";
        el("xr-value-cmp").textContent = "no critic head in archive";
      }
    }

    /* Debounced: one request per settled cursor position, not per pixel. */
    function scheduleWindow() {
      if (pending) clearTimeout(pending);
      pending = setTimeout(fetchWindow, 220);
    }

    async function fetchWindow() {
      if (!expId || step === lastFetched) return;
      const want = step;
      try {
        const body = await api.get("/api/experiments/" + expId + "/xray?step=" + want);
        if (want !== step) return; // a newer scrub already superseded this
        lastFetched = want;
        renderFeatures(body);
        renderAccount(body);
        renderHeatmap(body);
        el("xr-window-dims").textContent =
          body.window_values.length + " bars × " + body.feature_names.length +
          " features + " + body.account_names.length + " account = " + body.obs_dim;
        el("xr-notes").innerHTML =
          '<div class="caveat">' + body.scaling_note + "</div>" +
          (body.value_note ? '<div class="caveat">' + body.value_note + "</div>" : "");
      } catch (err) {
        el("xr-features").innerHTML =
          '<div class="lab-error">Could not load the observation: ' + err.message + "</div>";
      }
    }

    /* Feature rows grouped exactly as the research code groups them
     * (FEATURE_GROUPS), so the panel cannot drift from the model's input. */
    function renderFeatures(body) {
      const byName = {};
      body.feature_names.forEach((n, i) => (byName[n] = body.current[i]));
      // Scale bars against the largest magnitude on screen so they stay comparable.
      const span = Math.max(1e-6, ...body.current.map((v) => Math.abs(v)));
      // An ordered LIST, not an object: JSON object keys get sorted in transit,
      // which would scramble the semantic order (momentum first, context last).
      const groups = body.feature_groups && body.feature_groups.length
        ? body.feature_groups
        : [{ label: "Features", features: body.feature_names }];
      // Features that are structurally zero on this path (e.g. cross-asset
      // features on a synthetic series) are marked rather than left looking flat.
      const inert = new Set((body.meta && body.meta.inert_features) || []);

      el("xr-features").innerHTML = groups
        .map(function (group) {
          const rows = group.features
            .filter((n) => n in byName)
            .map(function (n) {
              const v = byName[n];
              const w = (Math.abs(v) / span) * 50;
              const bar = v >= 0
                ? '<i class="up" style="width:' + w + '%"></i>'
                : '<i class="dn" style="width:' + w + '%"></i>';
              const dead = inert.has(n);
              return '<div class="xr-feat' + (dead ? " is-inert" : "") + '">' +
                     '<span class="fname" title="' + n +
                     (dead ? " — inert on this path" : "") + '">' + n +
                     "</span>" +
                     '<span class="fbar">' + bar + "</span>" +
                     '<span class="fval ' + (dead ? "inert" : fmt.cls(v)) + '">' +
                     fmt.signed(v, 2) + "</span></div>";
            })
            .join("");
          return '<div class="xr-group"><h4>' + group.label + "</h4>" + rows + "</div>";
        })
        .join("");

      if (body.meta && body.meta.inert_features_note) {
        el("xr-features").insertAdjacentHTML(
          "beforeend",
          '<div class="caveat">' + body.meta.inert_features_note + "</div>"
        );
      }
    }

    function renderAccount(body) {
      const a = body.account;
      el("xr-account").innerHTML =
        metric("Position", fmt.signed(a.position_fraction, 3), fmt.cls(a.position_fraction), "of equity") +
        metric("Cash", fmt.signed(a.cash_fraction, 3), "neutral", "of start balance") +
        metric("Equity", fmt.num(a.equity_normalised, 3), fmt.cls(a.equity_normalised - 1), "normalised");
    }

    /* window_values is [bars][features] of z-scores; render it as a heatmap so
     * the whole market half of the observation is visible at once. */
    function renderHeatmap(body) {
      const cv = $("xr-heatmap");
      const rows = body.window_values;
      if (!rows || !rows.length) return;
      const cols = rows[0].length;
      const dpr = window.devicePixelRatio || 1;
      const w = cv.clientWidth || 420;
      const h = 260;
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
      cv.style.height = h + "px";
      const ctx = cv.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const padL = 4;
      const padB = 14;
      const cw = (w - padL) / cols;
      const ch = (h - padB) / rows.length;
      // Symmetric scale so zero is always the neutral colour.
      let span = 0;
      rows.forEach((r) => r.forEach((v) => { if (Math.abs(v) > span) span = Math.abs(v); }));
      span = Math.max(span, 1e-6);

      rows.forEach(function (row, y) {
        row.forEach(function (v, x) {
          const t = Math.max(-1, Math.min(1, v / span));
          const c = t >= 0
            ? "rgba(93,242,160," + (t * 0.85).toFixed(3) + ")"
            : "rgba(255,107,107," + (-t * 0.85).toFixed(3) + ")";
          const px = padL + x * cw;
          const py = y * ch;
          const pw = Math.max(cw - 0.5, 0.5);
          const ph = Math.max(ch - 0.5, 0.5);
          ctx.fillStyle = "#121a25";
          ctx.fillRect(px, py, pw, ph);
          ctx.fillStyle = c;
          ctx.fillRect(px, py, pw, ph);
        });
      });

      ctx.fillStyle = COLORS.text;
      ctx.font = "9px ui-monospace, monospace";
      ctx.textAlign = "left";
      ctx.fillText("oldest bar", padL, h - 4);
      ctx.textAlign = "right";
      ctx.fillText("newest bar", w, h - 4);
    }

    function init() {
      if (!$("xr-scrub")) return;
      $("xr-scrub").addEventListener("input", (e) => setStep(Number(e.target.value)));
      window.addEventListener("lab:trace", (e) => onTrace(e.detail));
      // Follow the Playground cursor so both panels stay on the same bar.
      window.addEventListener("lab:cursor", function (e) {
        if (!trace) return;
        step = e.detail.step;
        $("xr-scrub").value = String(step);
        renderChain();
        scheduleWindow();
      });
      window.addEventListener("lab:panel", function (e) {
        if (e.detail.panel === "xray" && trace) { renderChain(); scheduleWindow(); }
      });
    }

    return { init };
  })();


  /* -- Can you break the agent? ---------------------------- */
  /* Two experiments, deliberately labelled differently.
   *
   * (1) The trained ablation is REAL but PRECOMPUTED: each point is a full PPO
   *     training run, so it cannot be produced on request. The statistics over
   *     it are recomputed live, which is the honest interactive half.
   *
   * (2) The shift test is fully LIVE: the deployed policy is fixed and meets
   *     controlled synthetic distributions on demand. */
  const Generalization = (function () {
    let data = null;
    let market = "stock";

    async function load() {
      try {
        data = await api.get("/api/generalization");
        render();
      } catch (err) {
        showError($("gen-error"), "Could not load the ablation: " + err.message);
      }
    }

    function seedStrip(values) {
      return '<div class="seed-strip">' +
        values.map((v) => '<span class="seed-dot ' + (v > 0 ? "up" : "dn") +
                          '" title="' + fmt.pct(v) + '"></span>').join("") +
        "</div>";
    }

    function ci(pair) {
      if (!pair || pair.length < 2) return "";
      return '<span class="ci">95% CI [' + fmt.pct(pair[0], 0) + ", " + fmt.pct(pair[1], 0) + "]</span>";
    }

    function armCard(arm, blob, isWinner) {
      const inS = blob.in_sample;
      const out = blob.held_out;
      return (
        '<div class="ab-card' + (isWinner ? " is-winner" : "") + '">' +
        '<div class="ab-name">' + blob.label + "</div>" +
        '<div class="ab-sub">' + (arm === "single" ? "one fixed path" : "resampled every episode") + "</div>" +

        '<div class="ab-row"><span class="lbl">In-sample</span>' +
        seedStrip(inS.per_seed || []) +
        '<span class="val ' + fmt.cls(inS.mean) + '">' + fmt.pct(inS.mean, 0) + ci(inS.ci) + "</span></div>" +

        '<div class="ab-row"><span class="lbl">Held out</span>' +
        seedStrip(out.per_seed || []) +
        '<span class="val ' + fmt.cls(out.mean) + '">' + fmt.pct(out.mean, 0) + ci(out.ci) + "</span></div>" +

        '<div class="ab-gap"><span class="lbl">Generalization gap</span>' +
        '<span class="val ' + (blob.generalization_gap > 1 ? "neg" : "pos") + '">' +
        fmt.pct(blob.generalization_gap, 0) + "</span></div>" +
        "</div>"
      );
    }

    function render() {
      if (!data) return;
      const arms = data.markets[market];
      if (!arms) return;
      const single = arms.single;
      const domain = arms.domain;

      $("gen-ab").innerHTML =
        armCard("single", single, false) + armCard("domain", domain, true);

      // State the lesson in words, using this market's actual numbers.
      const ratio = single.in_sample.mean / Math.max(1e-9, Math.abs(domain.in_sample.mean));
      $("gen-verdict").innerHTML =
        "Agent <b>A</b> looks " + (ratio > 5 ? "spectacular" : "better") +
        " in training — <b>" + fmt.pct(single.in_sample.mean, 0) + "</b> on the path it " +
        "memorised — then returns <b>" + fmt.pct(single.held_out.mean, 0) + "</b> on paths it " +
        "has not seen. Agent <b>B</b> trains to a far more modest <b>" +
        fmt.pct(domain.in_sample.mean, 0) + "</b> and holds <b>" +
        fmt.pct(domain.held_out.mean, 0) + "</b> out of sample. " +
        "The in-sample number is the one that lies.";

      const rows = [
        ["Seeds", data.seeds ? data.seeds.join(", ") : "—"],
        ["Timesteps", data.timesteps ? data.timesteps.toLocaleString("en-US") : "—"],
        ["Source", data.source],
        ["Regenerate", data.generated_by],
        ["Why not live", data.why_not_live],
      ];
      $("gen-receipt").innerHTML = rows
        .filter((r) => r[1])
        .map((r) => "<dt>" + r[0] + "</dt><dd>" + r[1] + "</dd>").join("");
    }

    async function runTest() {
      const btn = $("gen-test");
      btn.disabled = true;
      setStatus($("gen-stat-status"), "recomputing…", true);
      try {
        const arms = data.markets[market];
        // Both arms were trained on the SAME seed set, so this is a genuinely
        // paired design: pair by seed index rather than testing one arm against
        // the other's scalar mean.
        const r = await api.post("/api/statistics", {
          values_a: arms.domain.held_out.per_seed,
          values_b: arms.single.held_out.per_seed,
          n_perm: Number($("gen-nperm").value),
          axis: "training_seed",
        });
        const res = r.resolution || {};
        const reachable = res.can_reach_05;
        $("gen-stat-out").innerHTML =
          '<div class="stat-out">' +
          metric("B held-out", fmt.pct(r.mean_a), fmt.cls(r.mean_a), "domain randomized") +
          metric("A held-out", fmt.pct(r.mean_b), fmt.cls(r.mean_b), "single path") +
          metric("Difference", fmt.pct(r.mean_difference), fmt.cls(r.mean_difference),
                 "95% CI [" + fmt.pct(r.difference_ci[0], 0) + ", " +
                 fmt.pct(r.difference_ci[1], 0) + "]") +
          metric("p-value", fmt.num(r.p_value, 4),
                 r.significant_at_05 ? "pos" : "neutral",
                 r.n_pairs + " paired seeds") +
          "</div>" +
          '<div class="caveat"><b>Resolution floor:</b> ' + (res.explanation || "") +
          (reachable ? "" : " With this many seeds the test cannot reach significance at " +
           "0.05 whatever the effect size, so read the confidence interval, not the p-value.") +
          "</div>" +
          '<div class="caveat">B beat A on <b>' + r.a_wins + " of " + r.n_pairs +
          "</b> seeds.</div>";
        setStatus($("gen-stat-status"),
                  r.n_perm.toLocaleString("en-US") + " permutations · " +
                  r.n_boot.toLocaleString("en-US") + " bootstrap resamples", false);
      } catch (err) {
        showError($("gen-error"), "Test failed: " + err.message);
        setStatus($("gen-stat-status"), "", false);
      } finally {
        btn.disabled = false;
      }
    }

    /* -- live distribution shift -- */
    async function runShift() {
      const btn = $("shift-run");
      btn.disabled = true;
      showError($("shift-error"), null);
      $("shift-bar").firstElementChild.style.width = "0%";
      const nSeeds = Number($("shift-seeds").value);
      const seeds = Array.from({ length: nSeeds }, (_, i) => i);

      try {
        const body = await api.runExperiment(
          {
            kind: "distribution_shift",
            seeds: seeds,
            config: {
              market: $("shift-market").dataset.value || "stock",
              mode: "synthetic",
              n_steps: 500,
            },
          },
          (b) => {
            const pctDone = Math.round((b.progress || 0) * 100);
            $("shift-bar").firstElementChild.style.width = pctDone + "%";
            setStatus($("shift-status"), b.id + " · " + b.stage, true);
          },
          180000
        );
        renderShift(body.result);
        setStatus($("shift-status"), body.id + " · complete · " + body.elapsed_sec + "s", false);
      } catch (err) {
        showError($("shift-error"), String(err.message || err));
        setStatus($("shift-status"), "", false);
      } finally {
        btn.disabled = false;
      }
    }

    function renderShift(result) {
      $("shift-out").hidden = false;
      const rows = result.regimes;
      // Scale against the widest per-path outcome, not the mean, so the plotted
      // points can never overflow the track they sit in.
      let span = 0.05;
      rows.forEach((r) => r.per_seed.forEach((s) => {
        if (Math.abs(s.excess) > span) span = Math.abs(s.excess);
      }));

      $("shift-bars").innerHTML = rows
        .map(function (r) {
          const v = r.mean_excess_return;
          const w = (Math.abs(v) / span) * 50;
          const bar = v >= 0
            ? '<i class="up" style="width:' + w + '%"></i>'
            : '<i class="dn" style="width:' + w + '%"></i>';
          const ac = r.per_seed.length
            ? r.per_seed.reduce((a, x) => a + x.realised_autocorr, 0) / r.per_seed.length
            : 0;
          // Every individual path, so a wide spread is visible rather than
          // hidden behind an average.
          const dots = r.per_seed
            .map(function (x) {
              const left = 50 + (x.excess / span) * 50;
              return '<b class="path-dot" style="left:' + Math.max(1, Math.min(99, left)) +
                     '%" title="path ' + x.seed + ": " + fmt.pct(x.excess, 1) + '"></b>';
            })
            .join("");

          return '<div class="rbar-row' + (r.is_reference ? " is-reference" : "") + '">' +
                 '<span class="rbar-name">' + r.label +
                 (r.is_reference ? ' <em class="ref-tag">in-distribution</em>' : "") +
                 "<small>lag-1 autocorr " + fmt.signed(ac, 3) +
                 " · agent " + fmt.pct(r.mean_agent_return, 0) +
                 " vs b&h " + fmt.pct(r.mean_benchmark_return, 0) + "</small></span>" +
                 '<span class="rbar-track">' + bar + dots + "</span>" +
                 '<span class="rbar-val ' + (r.excess_excludes_zero ? fmt.cls(v) : "muted-val") + '">' +
                 fmt.pct(v, 1) +
                 "<small>" + (r.excess_excludes_zero ? "" : "not distinguishable · ") +
                 "&plusmn;" + fmt.pct(r.std_excess, 0).replace("+", "") +
                 " over " + r.n_seeds + "</small></span></div>";
        })
        .join("");

      // Say plainly whether anything actually separated from the reference.
      const ref = rows.find((r) => r.is_reference);
      const separated = rows.filter((r) => !r.is_reference && r.excess_excludes_zero);
      let verdict;
      if (!ref) {
        verdict = "";
      } else if (!separated.length) {
        verdict =
          "<b>Nothing broke.</b> Across this many paths no regime's excess return " +
          "separates from zero by more than sampling noise — including the shifted " +
          "ones. That is a statement about the sample size as much as the agent: " +
          "add paths before concluding either way.";
      } else {
        verdict =
          "<b>" + separated.length + " of " + (rows.length - 1) + "</b> shifted regimes " +
          "produced an excess return distinguishable from zero at this sample size. " +
          "Compare each against the in-distribution reference rather than against zero.";
      }

      $("shift-notes").innerHTML =
        (verdict ? '<div class="gen-verdict">' + verdict + "</div>" : "") +
        '<div class="caveat">' + result.reference_note + "</div>" +
        '<div class="caveat">' + result.sampling_note + "</div>" +
        '<div class="caveat">' + result.note + "</div>" +
        '<div class="caveat">' + result.inference_note + "</div>";
    }

    function bindSeg(node, onChange) {
      if (!node) return;
      node.addEventListener("click", function (e) {
        const btn = e.target.closest("button[data-val]");
        if (!btn) return;
        node.dataset.value = btn.dataset.val;
        node.querySelectorAll("button").forEach((b) =>
          b.setAttribute("aria-pressed", String(b === btn))
        );
        if (onChange) onChange(btn.dataset.val);
      });
    }

    function init() {
      if (!$("gen-ab")) return;
      bindSeg($("gen-market"), function (v) { market = v; render(); $("gen-stat-out").innerHTML = ""; });
      bindSeg($("shift-market"));
      $("gen-test").addEventListener("click", runTest);
      $("shift-run").addEventListener("click", runShift);
      $("shift-seeds").addEventListener("input", function (e) {
        $("shift-seeds-val").textContent = e.target.value;
      });
      load();
    }

    return { init };
  })();


  /* -- Is the result real, or luck? ------------------------ */
  /* The data is precomputed and real: every point is a full PPO training run,
   * so "run 5 seeds" cannot be a live button. What IS live is the inference --
   * the bootstrap and permutation estimators the paper uses, recomputed on
   * request with caller-chosen parameters against the same real numbers. */
  const Seeds = (function () {
    let catalog = null;
    let headline = null;
    let market = "crypto";
    let last = null;

    async function load() {
      try {
        catalog = await api.get("/api/datasets");
        headline = catalog.headline_single_seed || {};
        await refresh();
        loadPaperAxis();
      } catch (err) {
        showError($("sd-error"), "Could not load the seed data: " + err.message);
      }
    }

    async function refresh() {
      showError($("sd-error"), null);
      setStatus($("sd-status"), "computing…", true);
      try {
        const body = await api.post("/api/statistics", {
          dataset: "real:" + market,
          confidence: Number($("sd-conf").value),
          n_boot: Number($("sd-boot").value),
          n_perm: Number($("sd-perm").value),
        });
        last = body;
        renderPair(body);
        renderSeeds(body);
        renderHist(body);
        renderStats(body);
        renderReceipt(body);
        setStatus($("sd-status"),
                  body.n_boot.toLocaleString("en-US") + " bootstrap resamples · " +
                  Math.round(body.confidence * 100) + "% interval", false);
      } catch (err) {
        showError($("sd-error"), "Recompute failed: " + err.message);
        setStatus($("sd-status"), "", false);
      }
    }

    /* One run beside five, at the same scale, in the same units. */
    function renderPair(body) {
      const one = headline[market];
      const ms = body.multi_seed;
      const spansZero = !ms.ci_excludes_zero;

      const claim = one
        ? '<div class="vp-card is-claim"><div class="vp-kicker">What one run says</div>' +
          '<div class="vp-num ' + fmt.cls(one.total_return) + '">' +
          fmt.pct(one.total_return, 1) + "</div>" +
          '<div class="vp-meta">seed ' + one.seed + " · " +
          (one.timesteps ? one.timesteps.toLocaleString("en-US") : "?") + " timesteps<br />" +
          (one.start_date || "") + " → " + (one.end_date || "") + "</div>" +
          '<div class="vp-flag bad">single run · no uncertainty</div></div>'
        : "";

      const truth =
        '<div class="vp-card is-truth"><div class="vp-kicker">What ' + body.n_seeds +
        " runs say</div>" +
        '<div class="vp-num ' + fmt.cls(ms.mean) + '">' + fmt.pct(ms.mean, 1) + "</div>" +
        '<div class="vp-meta">mean of ' + body.n_seeds + " independent seeds<br />" +
        Math.round(body.confidence * 100) + "% CI [" + fmt.pct(ms.ci_low, 1) + ", " +
        fmt.pct(ms.ci_high, 1) + "]</div>" +
        '<div class="vp-flag ' + (spansZero ? "bad" : "ok") + '">' +
        (spansZero ? "interval spans zero" : "interval excludes zero") + "</div></div>";

      $("sd-pair").innerHTML = claim + truth;

      const best = body.single_seed.best;
      $("sd-verdict").innerHTML = one
        ? "The headline number is <b>" + fmt.pct(one.total_return, 1) + "</b> from a single seed. " +
          "Retrain the same recipe five times and the mean is <b>" + fmt.pct(ms.mean, 1) +
          "</b>, with a " + Math.round(body.confidence * 100) + "% interval of <b>[" +
          fmt.pct(ms.ci_low, 1) + ", " + fmt.pct(ms.ci_high, 1) + "]</b>" +
          (spansZero
            ? " — which contains zero. The apparent edge does not survive reseeding."
            : " — which excludes zero.") +
          " The luckiest of those five returned <b>" + fmt.pct(best, 1) +
          "</b>; reporting that one alone would have been the same mistake."
        : "Across " + body.n_seeds + " seeds the mean is <b>" + fmt.pct(ms.mean, 1) + "</b>.";
    }

    function renderSeeds(body) {
      const vals = body.values;
      const span = Math.max(0.05, ...vals.map((v) => Math.abs(v)));
      $("sd-seeds").innerHTML = vals
        .map(function (v, i) {
          const w = (Math.abs(v) / span) * 50;
          const bar = v >= 0
            ? '<i class="up" style="width:' + w + '%"></i>'
            : '<i class="dn" style="width:' + w + '%"></i>';
          const isBest = i === body.single_seed.best_index;
          return '<div class="seed-row' + (isBest ? " is-best" : "") + '">' +
                 '<span class="sname">seed ' + (i + 1) +
                 (isBest ? " ★" : "") + "</span>" +
                 '<span class="stack">' + bar + "</span>" +
                 '<span class="sval ' + fmt.cls(v) + '">' + fmt.pct(v, 1) + "</span></div>";
        })
        .join("");
    }

    /* Histogram of the bootstrap means, with the observed mean and zero marked. */
    function renderHist(body) {
      const d = body.distribution;
      const cv = $("sd-hist");
      if (!d || !d.counts || !d.counts.length) return;
      const dpr = window.devicePixelRatio || 1;
      const w = cv.clientWidth || 600;
      const h = 150;
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
      cv.style.height = h + "px";
      const ctx = cv.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const padL = 6, padB = 20, padT = 8;
      const lo = d.edges[0];
      const hi = d.edges[d.edges.length - 1];
      const maxC = Math.max(...d.counts);
      const X = (v) => padL + ((v - lo) / (hi - lo || 1)) * (w - padL * 2);
      const bw = (w - padL * 2) / d.counts.length;

      d.counts.forEach(function (c, i) {
        const x = X(d.edges[i]);
        const bh = (c / maxC) * (h - padB - padT);
        ctx.fillStyle = d.edges[i] >= 0 ? "rgba(93,242,160,0.5)" : "rgba(255,107,107,0.5)";
        ctx.fillRect(x, h - padB - bh, Math.max(bw - 1, 1), bh);
      });

      function rule(v, color, label) {
        if (v < lo || v > hi) return;
        const x = X(v);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.6;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, h - padB);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = color;
        ctx.font = "10px ui-monospace, monospace";
        ctx.textAlign = x > w - 60 ? "right" : "left";
        ctx.fillText(label, x + (x > w - 60 ? -4 : 4), padT + 9);
      }
      rule(0, COLORS.bench, "0");
      rule(body.multi_seed.mean, COLORS.agent, "mean " + fmt.pct(body.multi_seed.mean, 1));

      ctx.fillStyle = COLORS.text;
      ctx.font = "10px ui-monospace, monospace";
      ctx.textAlign = "left";
      ctx.fillText(fmt.pct(lo, 0), padL, h - 6);
      ctx.textAlign = "right";
      ctx.fillText(fmt.pct(hi, 0), w - padL, h - 6);
    }

    function renderStats(body) {
      const ms = body.multi_seed;
      const ss = body.single_seed;
      const b = body.benchmark;
      let html =
        '<div class="stat-out">' +
        metric("Mean", fmt.pct(ms.mean), fmt.cls(ms.mean), body.n_seeds + " seeds") +
        metric("Median", fmt.pct(ms.median), fmt.cls(ms.median), "middle run") +
        metric("Std dev", fmt.pct(ms.std, 1).replace("+", ""), "neutral", "across seeds") +
        metric("Best run", fmt.pct(ss.best), fmt.cls(ss.best), "the one to not quote") +
        metric("Worst run", fmt.pct(ss.worst), fmt.cls(ss.worst), "same recipe") +
        metric("Spread", fmt.pct(ss.spread, 1), "neutral", "best minus worst") +
        "</div>";

      if (b) {
        const res = b.resolution || {};
        html +=
          '<div class="stat-out" style="margin-top:12px">' +
          metric("Benchmark", fmt.pct(b.value), "neutral", "buy &amp; hold") +
          metric("Difference", fmt.pct(b.mean_difference), fmt.cls(b.mean_difference),
                 "agent minus b&amp;h") +
          metric("p-value", fmt.num(b.p_value, 4), "neutral",
                 b.n_perm.toLocaleString("en-US") + " permutations") +
          metric("Beat benchmark", b.seeds_beating_benchmark + " / " + body.n_seeds,
                 b.seeds_beating_benchmark > body.n_seeds / 2 ? "pos" : "neg", "seeds") +
          "</div>" +
          '<div class="caveat">' + b.verdict + "</div>" +
          '<div class="caveat"><b>Axis:</b> ' + b.axis_note + "</div>";
      }
      $("sd-stats").innerHTML = html;
    }

    function renderReceipt(body) {
      const rows = [
        ["Dataset", body.dataset],
        ["Label", body.label],
        ["Source", body.source],
        ["Regenerate", body.generated_by],
        ["Published mean", body.published ? fmt.pct(body.published.mean, 2) : null],
        ["Published CI", body.published
          ? "[" + fmt.pct(body.published.ci_low, 2) + ", " + fmt.pct(body.published.ci_high, 2) + "]"
          : null],
      ];
      $("sd-receipt").innerHTML = rows
        .filter((r) => r[1])
        .map((r) => "<dt>" + r[0] + "</dt><dd>" + r[1] + "</dd>").join("");
    }

    /* The paper's own axis: paired across held-out tickers. */
    async function loadPaperAxis() {
      setStatus($("sd-paper-status"), "computing…", true);
      try {
        const body = await api.post("/api/statistics", {
          dataset: "assets:" + market,
          n_perm: Number($("sd-perm").value),
        });
        const res = body.resolution || {};
        $("sd-paper").innerHTML =
          '<div class="stat-out">' +
          metric("Agent mean", fmt.pct(body.mean_a), fmt.cls(body.mean_a),
                 body.n_pairs + " tickers") +
          metric("Buy &amp; hold", fmt.pct(body.mean_b), fmt.cls(body.mean_b), "same tickers") +
          metric("Difference", fmt.pct(body.mean_difference), fmt.cls(body.mean_difference),
                 "95% CI [" + fmt.pct(body.difference_ci[0], 0) + ", " +
                 fmt.pct(body.difference_ci[1], 0) + "]") +
          metric("p-value", fmt.num(body.p_value, 4),
                 body.significant_at_05 ? "pos" : "neutral",
                 "floor " + fmt.num(res.min_attainable_p, 4)) +
          "</div>" +
          '<div class="caveat">Agent beat buy &amp; hold on <b>' + body.a_wins + " of " +
          body.n_pairs + "</b> held-out tickers.</div>" +
          '<div class="caveat"><b>Note:</b> ' + body.caveat + "</div>";
        setStatus($("sd-paper-status"),
                  body.n_perm.toLocaleString("en-US") + " permutations · paired by ticker", false);
      } catch (err) {
        $("sd-paper").innerHTML =
          '<div class="lab-error">Could not run the published test: ' + err.message + "</div>";
        setStatus($("sd-paper-status"), "", false);
      }
    }

    function init() {
      if (!$("sd-pair")) return;
      const seg = $("sd-market");
      seg.addEventListener("click", function (e) {
        const btn = e.target.closest("button[data-val]");
        if (!btn) return;
        seg.dataset.value = btn.dataset.val;
        seg.querySelectorAll("button").forEach((b) =>
          b.setAttribute("aria-pressed", String(b === btn))
        );
        market = btn.dataset.val;
        refresh();
        loadPaperAxis();
      });
      $("sd-run").addEventListener("click", refresh);
      window.addEventListener("resize", function () { if (last) renderHist(last); });
      window.addEventListener("lab:panel", function (e) {
        if (e.detail.panel === "seeds" && last) renderHist(last);
      });
      load();
    }

    return { init };
  })();


  /* -- Walk-forward ---------------------------------------- */
  /* Disjoint chronological folds, each with its own scaler, plus a direct
   * measurement of what fitting that scaler on the test block costs.
   *
   * The one thing this panel must never imply is that it retrained per fold. It
   * cannot — the backend has no torch — so the same fixed policy is evaluated on
   * every fold, and the backend's own note saying so is rendered with the
   * results rather than tucked into a tooltip. */
  const WalkForward = (function () {
    let tickers = null;
    let last = null;

    function mode() { return $("wf-mode").dataset.value; }
    function market() { return $("wf-market").dataset.value; }

    async function fillSources() {
      const sel = $("wf-source");
      sel.innerHTML = "";
      if (mode() === "historical") {
        $("wf-source-label").textContent = "Ticker";
        if (!tickers) {
          try { tickers = await api.get("/api/tickers"); }
          catch (err) { tickers = { stock: ["SPY"], crypto: ["BTC-USD"] }; }
        }
        (tickers[market()] || []).forEach((t) => sel.add(new Option(t, t)));
        return;
      }
      $("wf-source-label").textContent = "Regime";
      try {
        const body = await api.get("/api/regimes");
        body.regimes.forEach((r) => sel.add(new Option(r.label, r.key)));
        sel.value = "momentum";
      } catch (err) { sel.add(new Option("Momentum", "momentum")); }
    }

    function config() {
      const cfg = { market: market(), mode: mode() };
      if (mode() === "historical") cfg.ticker = $("wf-source").value;
      else {
        cfg.regime = $("wf-source").value;
        cfg.seed = 3;
        // Folds need history: a 650-bar default would leave blocks too short to
        // step through once the feature warm-up is taken off the front.
        cfg.n_steps = 1600;
      }
      return cfg;
    }

    function timeline(res) {
      const total = res.n_rows || 1;
      return res.plan
        .map((f) => {
          const l = (f.train_start / total) * 100;
          const w = ((f.train_end - f.train_start) / total) * 100;
          const tl = (f.test_start / total) * 100;
          const tw = ((f.test_end - f.test_start) / total) * 100;
          const range = f.test_from ? `${f.test_from} → ${f.test_to}` :
            `bars ${f.test_start}–${f.test_end}`;
          return (
            `<div class="wf-row"><span class="wf-row-k">Fold ${f.fold + 1}</span>` +
            `<span class="wf-track">` +
            `<i class="wf-train" style="left:${l}%;width:${w}%"></i>` +
            `<i class="wf-test" style="left:${tl}%;width:${tw}%"></i></span>` +
            `<span class="wf-row-v">${range}</span></div>`
          );
        })
        .join("");
    }

    function table(res) {
      const head =
        "<thead><tr><th>Fold</th><th>Train bars</th><th>Test bars</th>" +
        "<th>Agent</th><th>Buy &amp; hold</th><th>Excess</th></tr></thead>";
      const body = res.folds
        .map(
          (f) =>
            `<tr><td>${f.fold + 1}</td><td>${f.train_bars}</td><td>${f.test_bars}</td>` +
            `<td class="${fmt.cls(f.agent_return)}">${fmt.pct(f.agent_return, 1)}</td>` +
            `<td class="${fmt.cls(f.benchmark_return)}">${fmt.pct(f.benchmark_return, 1)}</td>` +
            `<td class="${fmt.cls(f.excess_return)}"><b>${fmt.pct(f.excess_return, 1)}</b></td></tr>`
        )
        .join("");
      return head + "<tbody>" + body + "</tbody>";
    }

    function render(body) {
      const res = body.result;
      last = res;
      $("wf-timeline").innerHTML = timeline(res);
      $("wf-table").innerHTML = table(res);

      const s = res.summary;
      $("wf-summary").innerHTML =
        `<div class="pc-score">` +
        metric("Mean excess", fmt.pct(s.mean_excess_return, 1), fmt.cls(s.mean_excess_return),
               "across " + s.n_folds + " folds") +
        metric("Worst fold", fmt.pct(s.worst_fold_excess, 1), "neg") +
        metric("Best fold", fmt.pct(s.best_fold_excess, 1), "pos") +
        metric("Folds beaten", s.folds_beaten + " / " + s.n_folds, "neutral") +
        `</div><p class="pc-verdict">${s.spread_note}</p>`;

      const lk = res.leakage;
      $("wf-leakage").innerHTML = lk
        ? `<h4 class="attr-h">What fitting the scaler on the test block costs</h4>` +
          `<div class="wf-table-wrap"><table class="wf-table"><thead><tr><th>Fold</th>` +
          `<th>Scaler on train rows</th><th>Scaler on everything</th><th>Difference</th>` +
          `</tr></thead><tbody>` +
          lk.per_fold
            .map(
              (r) =>
                `<tr><td>${r.fold + 1}</td>` +
                `<td>${fmt.pct(r.train_only_return, 1)}</td>` +
                `<td>${fmt.pct(r.full_sample_return, 1)}</td>` +
                `<td class="${fmt.cls(r.delta)}"><b>${fmt.pct(r.delta, 1)}</b></td></tr>`
            )
            .join("") +
          `</tbody></table></div>` +
          `<p class="xr-hint">Largest single-fold difference: <b>${fmt.pct(lk.max_abs_delta, 1)}</b>. ` +
          `${lk.note}</p>`
        : "";

      $("wf-caveat").innerHTML =
        `<p class="attr-method">${res.fixed_policy_note}</p>` +
        `<ul><li>${res.scheme_note}</li><li>${res.inference_note}</li></ul>`;

      const m = res.meta;
      $("wf-receipt").innerHTML = [
        ["Experiment", body.id],
        ["Data", m.synthetic ? `synthetic · ${m.regime} · seed ${m.seed}`
                             : `${m.ticker} · ${m.provider}`],
        ["Dataset hash", m.dataset_hash],
        ["Rows after feature warm-up", String(res.n_rows)],
        ["Scheme", `${res.scheme} · first training window ${Math.round(res.train_min_frac * 100)}%`],
        ["Scaler", "fit per fold on training rows only"],
        ["Policy", "the single deployed policy, unchanged across folds"],
      ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");

      $("wf-out").hidden = false;
    }

    async function run() {
      const btn = $("wf-run");
      btn.disabled = true;
      showError($("wf-error"), null);
      $("wf-out").hidden = true;
      setStatus($("wf-status"), "splitting and evaluating…", true);
      const fill = $("wf-bar").firstElementChild;
      fill.style.width = "0%";
      try {
        const body = await api.runExperiment(
          {
            kind: "walk_forward",
            n_folds: Number($("wf-folds").value),
            scheme: $("wf-scheme").value,
            compare_leakage: true,
            config: config(),
          },
          (b) => { fill.style.width = Math.round((b.progress || 0) * 100) + "%"; }
        );
        fill.style.width = "100%";
        render(body);
        setStatus($("wf-status"), `${body.id} · ${body.elapsed_sec}s`, false);
      } catch (err) {
        showError($("wf-error"), String(err.message || err));
        setStatus($("wf-status"), "", false);
      } finally {
        btn.disabled = false;
      }
    }

    function bindSeg(id, onChange) {
      const seg = $(id);
      seg.addEventListener("click", function (e) {
        const btn = e.target.closest("button[data-val]");
        if (!btn) return;
        seg.dataset.value = btn.dataset.val;
        seg.querySelectorAll("button").forEach((b) =>
          b.setAttribute("aria-pressed", String(b === btn))
        );
        onChange(btn.dataset.val);
      });
    }

    async function showSchemeNote() {
      try {
        const meta = await api.get("/api/meta");
        const key = $("wf-scheme").value;
        const row = (meta.walk_forward.schemes || []).find((s) => s.key === key);
        $("wf-scheme-note").textContent = row ? row.description : "";
      } catch (err) { /* the note is nice to have, not load-bearing */ }
    }

    function init() {
      if (!$("wf-run")) return;
      bindSeg("wf-market", fillSources);
      bindSeg("wf-mode", fillSources);
      $("wf-folds").addEventListener("input", (e) => {
        $("wf-folds-val").textContent = e.target.value;
      });
      $("wf-scheme").addEventListener("change", showSchemeNote);
      $("wf-run").addEventListener("click", run);
      if (!api.ok) return;
      fillSources();
      showSchemeNote();
    }

    return { init, get last() { return last; } };
  })();


  /* -- What is it reading? (occlusion attribution) --------- */
  /* Ranks the agent's inputs by how far the target position moves when each one
   * is removed. Every bar here is a real forward pass through the deployed
   * policy — and every limit of the method travels with it, because a ranked
   * bar chart is exactly the kind of output a reader will take for causation. */
  const Attribution = (function () {
    let expId = null;
    let step = 0;
    let last = null;

    /* Bars are drawn relative to the strongest effect measured, so the chart
     * says "relative to the largest", never "share of the decision". */
    function bars(rows, valueKey, max) {
      return rows
        .map((r) => {
          const v = r[valueKey];
          const w = max > 1e-12 ? Math.max(1, (v / max) * 100) : 0;
          return (
            `<div class="attr-row"><span class="attr-name">${r.name}</span>` +
            `<span class="attr-track"><i style="width:${w.toFixed(1)}%"></i></span>` +
            `<span class="attr-val">${v.toFixed(3)}</span></div>`
          );
        })
        .join("");
    }

    function render(out) {
      last = out;
      const scope = $("attr-scope").value;
      const episode = scope === "episode" && out.episode;
      const rows = episode ? out.episode.features : out.local.market;
      const key = episode ? "mean_abs_delta" : "abs_delta";
      const acctKey = episode ? "mean_abs_delta" : "abs_delta";
      const acct = episode ? out.episode.account : out.local.account;
      const max = Math.max.apply(null, rows.map((r) => r[key]).concat([0]));

      $("attr-groups").innerHTML = out.groups
        .slice()
        .sort((a, b) => b.share - a.share)
        .map(
          (g) =>
            `<div class="attr-chip"><span class="attr-chip-k">${g.label}</span>` +
            `<span class="attr-chip-v">${(g.share * 100).toFixed(0)}%</span></div>`
        )
        .join("");

      $("attr-features").innerHTML = bars(rows, key, max);
      // Account scalars share the market block's scale, so the comparison
      // between "what the market says" and "what my book says" is honest.
      $("attr-account").innerHTML = bars(acct, acctKey, max);

      const dead = episode ? out.episode.dead_features : [];
      $("attr-dead").innerHTML = dead.length
        ? `<p class="xr-hint attr-dead">Exactly zero everywhere sampled: ` +
          `<b>${dead.join(", ")}</b>. ${out.inert_note || ""}</p>`
        : "";

      $("attr-caveats").innerHTML =
        `<p class="attr-method">${out.method}</p><ul>` +
        out.caveats.map((c) => `<li>${c}</li>`).join("") +
        `</ul>` +
        (episode
          ? `<p class="xr-hint">Averaged over ${out.episode.bars_sampled} bars ` +
            `sampled across the ${out.episode.bars_total}-bar episode ` +
            `(every ${out.episode.stride}). Magnitudes are averaged, not signed: ` +
            `a feature the agent leans on in both directions would otherwise ` +
            `cancel itself out to zero.</p>`
          : `<p class="xr-hint">Measured at bar ${out.step} only — one point in ` +
            `input space. The episode view is the more stable ranking.</p>`);

      $("attr-out").hidden = false;
    }

    async function run() {
      if (!expId) return;
      const btn = $("attr-run");
      btn.disabled = true;
      showError($("attr-error"), null);
      setStatus($("attr-status"), "occluding each input…", true);
      try {
        const out = await api.get(
          `/api/experiments/${expId}/attribution?step=${step}&bars=60`
        );
        render(out);
        setStatus($("attr-status"), `${out.episode.bars_sampled} bars measured`, false);
      } catch (err) {
        showError($("attr-error"), String(err.message || err));
        setStatus($("attr-status"), "", false);
      } finally {
        btn.disabled = false;
      }
    }

    function init() {
      if (!$("attr-run")) return;
      $("attr-run").addEventListener("click", run);
      // Switching scope re-renders what was already measured — it does not
      // silently re-run a different experiment behind the label.
      $("attr-scope").addEventListener("change", function () {
        if (last) render(last);
      });
      window.addEventListener("lab:trace", function (e) {
        expId = e.detail.body.id;
        last = null;
        $("attr-out").hidden = true;
        setStatus($("attr-status"), "", false);
      });
      window.addEventListener("lab:cursor", function (e) {
        if (step === e.detail.step) return;
        step = e.detail.step;
        // The local view belongs to a bar; keep it honest by clearing it.
        if (last && $("attr-scope").value === "local") {
          last = null;
          $("attr-out").hidden = true;
        }
      });
    }

    return { init };
  })();


  /* -- What if? (environment counterfactual) --------------- */
  /* Replays one bar under alternative actions from an identical environment
   * state. The alternatives are scored on price movement that already
   * occurred, which is exactly what makes this a counterfactual rather than a
   * forecast -- the copy says so, and the backend repeats it on every reply. */
  const WhatIf = (function () {
    let expId = null;
    let config = null;
    let step = 0;

    const CANDIDATES = [
      { a: 1.0, name: "Fully long", sub: "+1.00" },
      { a: 0.5, name: "Half long", sub: "+0.50" },
      { a: 0.0, name: "Flat", sub: "0.00" },
      { a: -0.5, name: "Half short", sub: "-0.50" },
      { a: -1.0, name: "Fully short", sub: "-1.00" },
    ];

    async function run() {
      if (!expId || !config) return;
      const btn = $("wi-run");
      btn.disabled = true;
      showError($("wi-error"), null);
      setStatus($("wi-status"), "replaying alternatives…", true);
      try {
        // Include the agent's own action so it is measured on the same footing
        // as the alternatives rather than quoted from elsewhere.
        const agentAction = Playground.trace
          ? Playground.trace.steps[step].action
          : 0;
        const actions = CANDIDATES.map((c) => c.a);
        if (!actions.some((a) => Math.abs(a - agentAction) < 1e-6)) actions.push(agentAction);

        const body = await api.runExperiment({
          kind: "counterfactual",
          step: step,
          actions: actions,
          horizon: Number($("wi-horizon").value),
          config: config,
        });
        render(body.result);
        setStatus($("wi-status"), body.id + " · " + body.elapsed_sec + "s", false);
      } catch (err) {
        showError($("wi-error"), String(err.message || err));
        setStatus($("wi-status"), "", false);
      } finally {
        btn.disabled = false;
      }
    }

    function label(a, isAgent) {
      if (isAgent) return { name: "Agent's choice", sub: fmt.signed(a, 3) };
      const known = CANDIDATES.find((c) => Math.abs(c.a - a) < 1e-6);
      return known || { name: fmt.signed(a, 2), sub: "custom" };
    }

    function render(result) {
      const rows = result.candidates.slice().sort((x, y) => y.action - x.action);
      const span = Math.max(1e-6, ...rows.map((r) => Math.abs(r.return)));

      const head =
        '<div class="wi-head-row"><span>Action</span><span>Return over ' +
        result.horizon + (result.horizon === 1 ? " bar" : " bars") +
        "</span><span>End equity</span><span>Reward</span></div>";

      const body = rows
        .map(function (r) {
          const l = label(r.action, r.is_agent_action);
          const w = (Math.abs(r.return) / span) * 50;
          const bar = r.return >= 0
            ? '<i class="up" style="width:' + w + '%"></i>'
            : '<i class="dn" style="width:' + w + '%"></i>';
          return '<div class="wi-row' + (r.is_agent_action ? " is-agent" : "") + '">' +
                 '<span class="wi-name">' + l.name + "<small>" + l.sub +
                 (r.is_agent_action ? " · taken" : "") + "</small></span>" +
                 '<span class="wi-track">' + bar + "</span>" +
                 '<span class="wi-num ' + fmt.cls(r.return) + '">' + fmt.pct(r.return, 2) +
                 "<small>" + fmt.money(r.end_equity) + "</small></span>" +
                 '<span class="wi-num ' + fmt.cls(r.reward) + '">' + fmt.signed(r.reward, 4) +
                 "</span></div>";
        })
        .join("");

      // Was the agent's choice the best one available at this bar? Answering it
      // plainly is the point of the panel -- and it is often "no", which is fine:
      // one bar of hindsight is not a verdict on a policy.
      const best = rows.reduce((a, b) => (b.return > a.return ? b : a));
      const agent = rows.find((r) => r.is_agent_action);
      let verdict = "";
      if (agent && best) {
        verdict = agent.is_agent_action && Math.abs(best.action - agent.action) < 1e-6
          ? "At this bar the agent's action was the best of those tried."
          : "At this bar <b>" + label(best.action, false).name.toLowerCase() +
            "</b> would have returned <b>" + fmt.pct(best.return, 2) + "</b> against the " +
            "agent's <b>" + fmt.pct(agent.return, 2) + "</b>. Hindsight on one bar is not " +
            "evidence about the policy — it is the distribution over many bars that matters.";
      }

      $("wi-out").innerHTML =
        head + body +
        (verdict ? '<div class="caveat">' + verdict + "</div>" : "") +
        '<div class="caveat">' + result.note + "</div>";
    }

    function init() {
      if (!$("wi-run")) return;
      $("wi-run").addEventListener("click", run);
      $("wi-horizon").addEventListener("change", function () {
        if ($("wi-out").innerHTML) run();
      });
      window.addEventListener("lab:trace", function (e) {
        expId = e.detail.body.id;
        config = e.detail.body.config;
        $("wi-out").innerHTML = "";
        setStatus($("wi-status"), "", false);
      });
      window.addEventListener("lab:cursor", function (e) {
        // Moving the cursor invalidates the previous bar's answer.
        if (step !== e.detail.step) {
          step = e.detail.step;
          $("wi-out").innerHTML = "";
        }
      });
    }

    return { init };
  })();


  /* -- Research notebook ----------------------------------- */
  /* Every experiment this session has run, with the config that produced it.
   *
   * Two things are deliberately NOT done here. The notebook never writes a
   * hypothesis on the visitor's behalf — an invented research question is the
   * notebook's version of a fabricated result — so an experiment run without one
   * is shown as "no question stated". And the "finding" line is derived only
   * from numbers the backend actually returned; a failed or pending experiment
   * gets no finding at all. */
  const Notebook = (function () {
    let rows = [];
    let openId = null;

    async function loadRegimes() {
      try {
        const body = await api.get("/api/regimes");
        $("nb-regime").innerHTML = body.regimes
          .map((r) => '<option value="' + r.key + '">' + r.label + "</option>")
          .join("");
      } catch (err) {
        /* the selector stays empty; running will surface the real error */
      }
    }

    function kindLabel(kind) {
      if (kind === "rollout") return "Rollout";
      if (kind === "distribution_shift") return "Shift sweep";
      if (kind === "counterfactual") return "Counterfactual";
      return kind;
    }

    function configLine(row) {
      const c = row.config || {};
      const bits = [c.market];
      if (c.mode === "historical") bits.push(c.ticker);
      else bits.push(c.regime || "synthetic", "seed " + c.seed);
      bits.push(fmt.money(c.initial_balance));
      bits.push(Math.round((c.transaction_cost || 0) * 10000) + " bps");
      bits.push(c.reward === "dsr" ? "diff. sharpe" : "return");
      return bits.filter(Boolean).join(" · ");
    }

    /* Only ever derived from numbers the backend returned. */
    function finding(row) {
      if (row.status === "error") {
        return '<span class="neg">failed</span><small>' +
               (row.error || "").slice(0, 40) + "</small>";
      }
      if (row.status !== "done") {
        return '<span class="muted-val">' + Math.round((row.progress || 0) * 100) +
               "%</span><small>" + (row.stage || row.status) + "</small>";
      }
      const r = row._result;
      if (!r) return '<span class="muted-val">—</span><small>open to load</small>';
      if (row.kind === "rollout" && r.metrics) {
        const a = r.metrics.total_return;
        const b = r.bench_metrics ? r.bench_metrics.total_return : null;
        return '<span class="' + fmt.cls(a) + '">' + fmt.pct(a, 1) + "</span><small>" +
               (b === null ? "" : "b&h " + fmt.pct(b, 1)) + "</small>";
      }
      if (row.kind === "distribution_shift" && r.regimes) {
        const sep = r.regimes.filter((x) => x.excess_excludes_zero).length;
        return '<span class="muted-val">' + r.regimes.length + " regimes</span><small>" +
               sep + " separated from zero</small>";
      }
      if (row.kind === "counterfactual" && r.candidates) {
        return '<span class="muted-val">' + r.candidates.length +
               " actions</span><small>horizon " + r.horizon + "</small>";
      }
      return '<span class="muted-val">done</span><small>' + row.elapsed_sec + "s</small>";
    }

    function render() {
      const list = $("nb-list");
      if (!rows.length) {
        list.innerHTML =
          '<div class="nb-empty">No experiments yet. Every run from any panel in ' +
          "this session appears here.</div>";
        $("nb-count").textContent = "";
        return;
      }
      $("nb-count").textContent =
        rows.length + (rows.length === 1 ? " experiment" : " experiments") + " this session";

      list.innerHTML = rows
        .map(function (row) {
          const when = new Date(row.created_at * 1000).toLocaleTimeString("en-US", {
            hour: "2-digit", minute: "2-digit", second: "2-digit",
          });
          const q = row.question
            ? '<div class="nb-q">' + escapeHtml(row.question) + "</div>"
            : '<div class="nb-q is-unstated">no question stated</div>';
          return '<div class="nb-row" data-id="' + row.id + '">' +
                 '<span class="nb-id">' + row.id + "<small>" + kindLabel(row.kind) +
                 "</small></span>" +
                 '<span class="nb-main">' + q +
                 '<div class="nb-cfg">' + configLine(row) + " · " + when + "</div></span>" +
                 '<span class="nb-find">' + finding(row) + "</span>" +
                 "</div>";
        })
        .join("");
    }

    function escapeHtml(str) {
      const d = document.createElement("div");
      d.textContent = str;
      return d.innerHTML;
    }

    async function refresh() {
      try {
        const body = await api.get("/api/experiments?limit=50");
        $("nb-storage").textContent = body.storage;
        const previous = {};
        rows.forEach((r) => { if (r._result) previous[r.id] = r._result; });
        rows = body.experiments.map(function (r) {
          r._result = previous[r.id] || null;
          return r;
        });
        render();
      } catch (err) {
        showError($("nb-error"), "Could not load history: " + err.message);
      }
    }

    async function open(id) {
      openId = id;
      const card = $("nb-detail-card");
      card.hidden = false;
      $("nb-detail-title").textContent = "Experiment " + id;
      $("nb-detail").innerHTML = '<div class="lab-status">loading…</div>';
      try {
        const body = await api.get("/api/experiments/" + id);
        const row = rows.find((r) => r.id === id);
        if (row) { row._result = body.result || null; render(); }

        const receipt = body.receipt || {};
        const prov = receipt.provenance || {};
        const cfg = body.config || {};

        const entries = [
          ["Experiment", body.id],
          ["Kind", kindLabel(body.kind)],
          ["Question", body.question || "— not stated —"],
          ["Status", body.status + (body.error ? " · " + body.error : "")],
          ["Started", receipt.created_at_utc],
          ["Elapsed", body.elapsed_sec + "s"],
          ["Code version", receipt.code_version || "unknown"],
          ["Market", cfg.market],
          ["Mode", cfg.mode === "historical" ? "real data · " + cfg.ticker
                                             : "synthetic · " + cfg.regime + " · seed " + cfg.seed],
          ["Starting capital", fmt.money(cfg.initial_balance)],
          ["Transaction cost", Math.round(cfg.transaction_cost * 10000) + " bps"],
          ["Slippage", Math.round(cfg.slippage * 10000) + " bps"],
          ["Reward", cfg.reward === "dsr" ? "differential Sharpe" : "risk-aware return"],
          ["Shorting", cfg.allow_short ? "allowed" : "long only"],
          ["Dataset hash", prov.dataset_hash],
          ["Bars", prov.bars],
          ["Policy", prov.policy ? prov.policy.name + " · " + prov.policy.sha256 : null],
          ["Critic", prov.policy
            ? (prov.policy.has_value_head ? "exported" : "not in this archive")
            : null],
          ["Storage", receipt.storage],
        ];

        $("nb-detail").innerHTML =
          '<dl class="receipt">' +
          entries.filter((e) => e[1] !== undefined && e[1] !== null)
                 .map((e) => "<dt>" + e[0] + "</dt><dd>" + escapeHtml(String(e[1])) + "</dd>")
                 .join("") +
          "</dl>" +
          '<div class="caveat">This config round-trips exactly: replaying it rebuilds ' +
          "an identical environment and reproduces these numbers.</div>";
      } catch (err) {
        $("nb-detail").innerHTML =
          '<div class="lab-error">Could not load ' + id + ": " + err.message + "</div>";
      }
    }

    async function reproduce() {
      if (!openId) return;
      const btn = $("nb-reproduce");
      btn.disabled = true;
      setStatus($("nb-status"), "reproducing " + openId + "…", true);
      try {
        const cfgBody = await api.get("/api/experiments/" + openId + "/config");
        const original = rows.find((r) => r.id === openId);
        const body = await api.runExperiment({
          kind: cfgBody.kind,
          config: cfgBody.config,
          question: (original && original.question)
            ? "Reproduction of " + openId + ": " + original.question
            : "Reproduction of " + openId,
        }, null, 180000);
        await refresh();
        await open(body.id);
        setStatus($("nb-status"), body.id + " reproduced " + openId, false);
      } catch (err) {
        showError($("nb-error"), "Reproduction failed: " + err.message);
        setStatus($("nb-status"), "", false);
      } finally {
        btn.disabled = false;
      }
    }

    async function run() {
      const btn = $("nb-run");
      btn.disabled = true;
      showError($("nb-error"), null);
      $("nb-bar").firstElementChild.style.width = "0%";
      const kind = $("nb-kind").value;
      const question = $("nb-question").value.trim();

      const payload = {
        kind: kind,
        question: question || undefined,
        config: {
          market: $("nb-market").dataset.value || "stock",
          mode: "synthetic",
          regime: $("nb-regime").value || "random_walk",
          seed: Number($("nb-seed").value) || 0,
          n_steps: 650,
        },
      };
      if (kind === "distribution_shift") payload.seeds = [0, 1, 2, 3, 4];

      try {
        const body = await api.runExperiment(payload, function (b) {
          $("nb-bar").firstElementChild.style.width =
            Math.round((b.progress || 0) * 100) + "%";
          setStatus($("nb-status"), b.id + " · " + b.stage, true);
        }, 180000);
        await refresh();
        await open(body.id);
        setStatus($("nb-status"), body.id + " · complete · " + body.elapsed_sec + "s", false);
      } catch (err) {
        showError($("nb-error"), String(err.message || err));
        setStatus($("nb-status"), "", false);
        await refresh();
      } finally {
        btn.disabled = false;
      }
    }

    function syncKind() {
      const isShift = $("nb-kind").value === "distribution_shift";
      // A sweep chooses its own regimes, so the single-regime picker is irrelevant.
      $("nb-regime-ctl").style.display = isShift ? "none" : "";
    }

    function init() {
      if (!$("nb-list")) return;
      const seg = $("nb-market");
      seg.addEventListener("click", function (e) {
        const b = e.target.closest("button[data-val]");
        if (!b) return;
        seg.dataset.value = b.dataset.val;
        seg.querySelectorAll("button").forEach((x) =>
          x.setAttribute("aria-pressed", String(x === b))
        );
      });
      $("nb-kind").addEventListener("change", syncKind);
      $("nb-run").addEventListener("click", run);
      $("nb-refresh").addEventListener("click", refresh);
      $("nb-reproduce").addEventListener("click", reproduce);
      $("nb-close").addEventListener("click", function () {
        $("nb-detail-card").hidden = true;
        openId = null;
      });
      $("nb-list").addEventListener("click", function (e) {
        const row = e.target.closest(".nb-row[data-id]");
        if (row) open(row.dataset.id);
      });
      // Any panel's run shows up here, so refresh whenever this tab is opened.
      window.addEventListener("lab:panel", function (e) {
        if (e.detail.panel === "notebook") refresh();
      });
      syncKind();
      loadRegimes();
      refresh();
    }

    return { init, refresh };
  })();

  /* ── backend status ─────────────────────────────────────── */
  async function initStatus() {
    const pill = $("lab-api-status");
    if (!pill) return;
    if (!api.ok) {
      pill.className = "api-pill is-down";
      pill.innerHTML = '<i class="dot-led"></i> No backend configured';
      return;
    }
    pill.innerHTML = '<i class="dot-led"></i> connecting…';
    let health;
    try {
      health = await api.get("/health");
    } catch (err) {
      pill.className = "api-pill is-down";
      pill.innerHTML = '<i class="dot-led"></i> API unreachable';
      return;
    }

    // /health alone is not enough. The dashboard-era backend serves it too, so
    // reporting "live" off that would claim the lab works and then 404 on every
    // panel — the one failure mode this site must not have. Confirm the lab API
    // itself answers before saying anything reassuring.
    try {
      await api.get("/api/meta");
    } catch (err) {
      pill.className = "api-pill is-stale";
      pill.innerHTML =
        `<i class="dot-led"></i> Backend v${health.version || "?"} — too old for the lab`;
      pill.title =
        "The API responds but does not serve the lab endpoints. Redeploy the " +
        "backend from a revision that includes server/lab.py.";
      return;
    }

    pill.className = "api-pill is-live";
    pill.innerHTML =
      `<i class="dot-led"></i> API live · ${health.policies.join(" + ")} · v${health.version}`;
  }

  /* ── lab sub-tabs ───────────────────────────────────────── */
  /* Panels are addressable as #lab/<panel>, so a link can point at a specific
   * experiment rather than dropping the reader on the first tab and asking them
   * to find it. Keyboard behaviour follows the ARIA tabs pattern: arrows move
   * between tabs, Home/End jump to the ends, and a roving tabindex keeps a
   * single stop in the page's tab order. */
  const PANELS = ["perception", "playground", "xray", "generalization",
                  "walkforward", "seeds", "notebook"];

  function showPanel(name, updateHash) {
    if (PANELS.indexOf(name) === -1) name = PANELS[0];
    document.querySelectorAll(".lab-tab").forEach(function (t) {
      const on = t.dataset.panel === name;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
    });
    document.querySelectorAll(".lab-panel").forEach(function (p) {
      p.classList.toggle("active", p.id === "panel-" + name);
    });
    if (updateHash && location.hash.indexOf("#lab") === 0) {
      history.replaceState(null, "", "#lab/" + name);
    }
    window.dispatchEvent(new CustomEvent("lab:panel", { detail: { panel: name } }));
  }

  function panelFromHash() {
    const m = /^#lab\/([a-z]+)/.exec(location.hash || "");
    return m && PANELS.indexOf(m[1]) !== -1 ? m[1] : null;
  }

  function initTabs() {
    const tabs = Array.prototype.slice.call(document.querySelectorAll(".lab-tab"));
    if (!tabs.length) return;

    tabs.forEach(function (t, i) {
      t.addEventListener("click", function () {
        showPanel(t.dataset.panel, true);
      });
      t.addEventListener("keydown", function (e) {
        const keys = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: 1, ArrowUp: -1 };
        let next = null;
        if (keys[e.key]) next = (i + keys[e.key] + tabs.length) % tabs.length;
        else if (e.key === "Home") next = 0;
        else if (e.key === "End") next = tabs.length - 1;
        if (next === null) return;
        e.preventDefault();
        tabs[next].focus();
        showPanel(tabs[next].dataset.panel, true);
      });
    });

    // Honour a deep link on arrival, and keep up with back/forward navigation.
    const initial = panelFromHash();
    if (initial) showPanel(initial, false);
    window.addEventListener("hashchange", function () {
      const p = panelFromHash();
      if (p) showPanel(p, false);
    });
  }

  function boot() {
    if (!$("view-lab")) return;
    initTabs();
    initStatus();
    Perception.init();
    Playground.init();
    XRay.init();
    Attribution.init();
    Generalization.init();
    WalkForward.init();
    Seeds.init();
    WhatIf.init();
    Notebook.init();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  // Shared with the other lab panels (X-Ray, generalization, multi-seed).
  window.RLLab = { api, fmt, Chart, COLORS, pick, setStatus, showError, metric, showPanel,
                   Perception, Playground, XRay, Attribution, Generalization,
                   WalkForward, Seeds, WhatIf, Notebook };
})();
