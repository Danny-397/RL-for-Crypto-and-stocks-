/* RL·Trader prototype — lightweight, dependency-free interactions.
   1) An animated "live agent" equity curve (agent vs. buy & hold).
   2) Count-up animation for the stats strip.
   All purely illustrative — it visualises the concept, not real model output. */

(() => {
  "use strict";

  // ────────────────────────────────────────────────────────────
  // 1. Equity-curve canvas
  // ────────────────────────────────────────────────────────────
  const canvas = document.getElementById("equityChart");
  if (canvas && canvas.getContext) {
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    const PAD = { l: 12, r: 12, t: 16, b: 16 };

    // Seeded pseudo-random walk so the curve looks plausible and is stable.
    function seededRandom(seed) {
      let s = seed >>> 0;
      return () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 4294967296;
      };
    }

    function buildSeries(n, drift, vol, seed) {
      const rnd = seededRandom(seed);
      const out = [1];
      for (let i = 1; i < n; i++) {
        const shock = drift + (rnd() - 0.5) * vol;
        out.push(Math.max(0.5, out[i - 1] * (1 + shock)));
      }
      return out;
    }

    const N = 120;
    // Agent ends higher and smoother; benchmark is choppier and lags.
    const agent = buildSeries(N, 0.0016, 0.018, 7);
    const bench = buildSeries(N, 0.0007, 0.022, 23);

    const allVals = agent.concat(bench);
    const min = Math.min(...allVals) * 0.98;
    const max = Math.max(...allVals) * 1.02;

    const x = (i) => PAD.l + (i / (N - 1)) * (W - PAD.l - PAD.r);
    const y = (v) => PAD.t + (1 - (v - min) / (max - min)) * (H - PAD.t - PAD.b);

    function drawGrid() {
      ctx.strokeStyle = "rgba(255,255,255,0.05)";
      ctx.lineWidth = 1;
      for (let g = 0; g <= 4; g++) {
        const gy = PAD.t + (g / 4) * (H - PAD.t - PAD.b);
        ctx.beginPath();
        ctx.moveTo(PAD.l, gy);
        ctx.lineTo(W - PAD.r, gy);
        ctx.stroke();
      }
    }

    function drawLine(series, count, color, width, glow, fill) {
      ctx.save();
      if (fill) {
        const grad = ctx.createLinearGradient(0, PAD.t, 0, H);
        grad.addColorStop(0, "rgba(212,255,63,0.18)");
        grad.addColorStop(1, "rgba(212,255,63,0)");
        ctx.beginPath();
        ctx.moveTo(x(0), y(series[0]));
        for (let i = 1; i < count; i++) ctx.lineTo(x(i), y(series[i]));
        ctx.lineTo(x(count - 1), H - PAD.b);
        ctx.lineTo(x(0), H - PAD.b);
        ctx.closePath();
        ctx.fillStyle = grad;
        ctx.fill();
      }
      ctx.beginPath();
      ctx.moveTo(x(0), y(series[0]));
      for (let i = 1; i < count; i++) ctx.lineTo(x(i), y(series[i]));
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineJoin = "round";
      if (glow) {
        ctx.shadowColor = color;
        ctx.shadowBlur = 12;
      }
      ctx.stroke();
      ctx.restore();
    }

    function drawHead(series, count, color) {
      const cx = x(count - 1);
      const cy = y(series[count - 1]);
      ctx.save();
      ctx.shadowColor = color;
      ctx.shadowBlur = 14;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Animated reveal, then a gentle live "breathing" loop.
    let progress = 0;
    let revealed = false;

    function frame() {
      ctx.clearRect(0, 0, W, H);
      drawGrid();

      const count = revealed ? N : Math.max(2, Math.floor(progress * N));
      drawLine(bench, Math.min(count, N), "rgba(120,132,148,0.7)", 2, false, false);
      drawLine(agent, Math.min(count, N), "#d4ff3f", 2.6, true, true);
      drawHead(agent, Math.min(count, N), "#d4ff3f");

      if (!revealed) {
        progress += 0.018;
        if (progress >= 1) {
          revealed = true;
          animateMetrics();
        }
        requestAnimationFrame(frame);
      }
    }

    // Reveal the headline metrics in sync with the curve finishing.
    function animateMetrics() {
      const totalRet = (agent[N - 1] - 1) * 100;
      countTo("m-return", 0, totalRet, 900, (v) => (v >= 0 ? "+" : "") + v.toFixed(1) + "%");
      countTo("m-sharpe", 0, 1.12, 900, (v) => v.toFixed(2));
      // Max drawdown of the agent series.
      let peak = agent[0], dd = 0;
      for (const v of agent) { peak = Math.max(peak, v); dd = Math.max(dd, (peak - v) / peak); }
      countTo("m-dd", 0, dd * 100, 900, (v) => v.toFixed(1) + "%");
    }

    // Kick off once the card scrolls into view (or immediately if already visible).
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting) { frame(); io.disconnect(); }
      }, { threshold: 0.3 });
      io.observe(canvas);
    } else {
      frame();
    }
  }

  // ────────────────────────────────────────────────────────────
  // 2. Generic count-up helper + stats strip
  // ────────────────────────────────────────────────────────────
  function countTo(id, from, to, dur, fmt) {
    const el = document.getElementById(id);
    if (!el) return;
    const start = performance.now();
    function tick(now) {
      const p = Math.min(1, (now - start) / dur);
      const eased = 1 - Math.pow(1 - p, 3); // easeOutCubic
      el.textContent = fmt(from + (to - from) * eased);
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  const statNums = document.querySelectorAll(".stat-cell .num");
  if (statNums.length && "IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries, obs) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        const el = e.target;
        const target = parseInt(el.dataset.count, 10);
        const suffix = el.querySelector(".pct");
        let i = 0;
        const step = Math.max(1, Math.round(target / 28));
        const timer = setInterval(() => {
          i = Math.min(target, i + step);
          el.firstChild ? (el.childNodes[0].nodeValue = String(i)) : (el.textContent = i);
          if (suffix) el.appendChild(suffix);
          if (i >= target) clearInterval(timer);
        }, 28);
        obs.unobserve(el);
      });
    }, { threshold: 0.5 });
    statNums.forEach((n) => io.observe(n));
  }
})();
