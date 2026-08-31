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
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillStyle = COLORS.text;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let g = 0; g <= 3; g++) {
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
    try {
      const h = await api.get("/health");
      pill.className = "api-pill is-live";
      pill.innerHTML = `<i class="dot-led"></i> API live · ${h.policies.join(" + ")} · v${h.version}`;
    } catch (err) {
      pill.className = "api-pill is-down";
      pill.innerHTML = '<i class="dot-led"></i> API unreachable';
    }
  }

  /* ── lab sub-tabs ───────────────────────────────────────── */
  function initTabs() {
    const tabs = document.querySelectorAll(".lab-tab");
    if (!tabs.length) return;
    tabs.forEach((t) => {
      t.addEventListener("click", () => {
        tabs.forEach((x) => x.setAttribute("aria-selected", String(x === t)));
        document.querySelectorAll(".lab-panel").forEach((p) => {
          p.classList.toggle("active", p.id === "panel-" + t.dataset.panel);
        });
        window.dispatchEvent(new CustomEvent("lab:panel", { detail: { panel: t.dataset.panel } }));
      });
    });
  }

  function boot() {
    if (!$("view-lab")) return;
    initTabs();
    initStatus();
    Playground.init();
    XRay.init();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  // Shared with the other lab panels (X-Ray, generalization, multi-seed).
  window.RLLab = { api, fmt, Chart, COLORS, pick, setStatus, showError, metric, Playground, XRay };
})();
