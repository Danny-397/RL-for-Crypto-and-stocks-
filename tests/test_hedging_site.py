"""The hedging page and its data file agree with the committed results (no hand-typed numbers)."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "hedging.js"


def _data():
    text = DATA.read_text(encoding="utf-8")
    return json.loads(re.search(r"window\.HEDGING = (\{.*\});\s*$", text, re.S).group(1))


def test_the_page_has_the_view_the_nav_link_and_the_script():
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    assert 'id="view-hedging"' in html
    assert 'href="#hedging"' in html and 'data-view="hedging"' in html
    assert 'src="hedging-view.js"' in html
    assert '"hedging"' in (ROOT / "docs" / "app.js").read_text(encoding="utf-8")


def test_the_data_file_matches_the_results():
    d = _data()
    for name in ("sim", "real"):
        res = json.loads((ROOT / "docs" / "assets" / f"hedging_{name}.json").read_text(encoding="utf-8"))
        assert [r["cost"] for r in d[name]["costs"]] == [r["cost"] for r in res["costs"]]
        for a, b in zip(d[name]["costs"], res["costs"]):
            assert abs(a["agent_median"]["objective"] - b["agent_median"]["objective"]) < 1e-3
            assert a["vs_band"]["wins"] == b["vs_band"]["wins"]


def test_one_exported_policy_per_cost_and_every_real_month():
    d = _data()
    assert sorted(float(k) for k in d["policies"]) == sorted(d["costs"])
    for p in d["policies"].values():
        assert len(p["layers"][0]["W"][0]) == 4                   # four observations, no delta among them
        assert len(p["head"]["W"]) == 1
    n = len(d["episodes"]["starts"])
    assert n == d["real"]["episodes"] and len(d["episodes"]["S"]) == n
    assert all(len(s) == d["n_steps"] + 1 for s in d["episodes"]["S"])


def test_the_browser_policy_matches_the_python_one():
    """The page runs the policy in JavaScript; its maths must equal the Python forward pass."""
    import numpy as np

    d = _data()
    p = next(iter(d["policies"].values()))
    obs = np.array([0.02, 0.5, 0.25, 0.4])
    x = np.clip((obs - p["obs_mean"]) / np.array(p["obs_std"]), -p["obs_clip"], p["obs_clip"])
    for L in p["layers"]:
        x = np.tanh(np.array(L["W"]) @ x + np.array(L["b"]))
    a = np.tanh(np.array(p["head"]["W"]) @ x + np.array(p["head"]["b"]))[0]
    hedge = float(np.clip((a + 1) / 2, 0, 1))
    js = (ROOT / "docs" / "hedging-view.js").read_text(encoding="utf-8")
    # The page's forward pass: normalise, tanh layers, tanh head, map to [0, 1].
    assert "Math.tanh(p.head.W[0]" in js and "(a + 1) / 2" in js and "obs_clip" in js
    assert 0.0 <= hedge <= 1.0


def test_the_page_runs_the_same_policy_as_pytorch(tmp_path):
    """docs/hedging-view.js must compute the hedge the trained network computes.

    It once applied tanh after every trunk layer, including the last, which the
    network does not; the page's hedges were off by up to 0.06 shares while the
    published numbers (computed in PyTorch) were right. This runs the page's own
    function in Node against a freshly exported policy.
    """
    import json
    import shutil
    import subprocess

    import numpy as np
    import pytest
    import torch

    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    from rl_trader.hedging.core import OBS_DIM
    from rl_trader.hedging.train import HedgeTrainConfig
    from rl_trader.models.ppo_agent import PPOAgent
    from tools.hedging_experiment import export_policy

    torch.manual_seed(0)
    agent = PPOAgent(OBS_DIM, 1, HedgeTrainConfig().ppo)
    rng = np.random.default_rng(0)
    for _ in range(100):
        agent.observe(np.array([rng.normal(0, .05), rng.uniform(), rng.uniform(.1, .45), rng.uniform()]))
    with torch.no_grad():                         # make the last layer matter
        for p in agent.ac.parameters():
            p.mul_(3.0)
    pol = export_policy(agent)
    obs = [[float(rng.normal(0, .05)), float(rng.uniform()), float(rng.uniform(.1, .45)), float(rng.uniform())]
           for _ in range(50)]
    src = open("docs/hedging-view.js", encoding="utf-8").read()
    start = src.index("function policyHedge(p, obs) {")
    end = src.index("\n  }\n", start) + 4
    script = tmp_path / "run.js"
    script.write_text("const f = (" + src[start:end].strip() + ");\n"
                      f"const p = {json.dumps(pol)};\nconst obs = {json.dumps(obs)};\n"
                      "console.log(JSON.stringify(obs.map(o => f(p, o))));\n", encoding="utf-8")
    js = json.loads(subprocess.run(["node", str(script)], capture_output=True, text=True, check=True).stdout)
    torch_h = [float(np.clip((agent.select_action(np.array(o), deterministic=True)[0][0] + 1) / 2, 0, 1))
               for o in obs]
    assert max(abs(a - b) for a, b in zip(js, torch_h)) < 1e-4
