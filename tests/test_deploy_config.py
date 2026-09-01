"""The deployment config must describe the deployment that exists.

`render.yaml` named the service `rl-trader-api` and `DEPLOY.md` told you to
verify it with `curl https://rl-trader-api.onrender.com/health`. That host has
never existed — the live backend is `rl-for-crypto-and-stocks.onrender.com`,
which is what `docs/config.js` calls. So the documented way to check a healthy
deployment returned 404, and the documented way to redeploy would have created a
second service on a URL the site does not call.

The hostname on Render follows the service name, so these three files have to
agree. Nothing checked that they did.
"""

from __future__ import annotations

import io
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HOST_RE = re.compile(r"https://([a-z0-9-]+)\.onrender\.com")


def _read(*parts: str) -> str:
    with io.open(os.path.join(REPO, *parts), encoding="utf-8") as fh:
        return fh.read().replace("\r\n", "\n")


def _service_name() -> str:
    """The `name:` under services: in render.yaml, ignoring comments."""
    for line in _read("render.yaml").split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("name:"):
            return stripped.split(":", 1)[1].strip()
    raise AssertionError("render.yaml declares no service name")


def _configured_host() -> str:
    """The host the site actually calls.

    Read from the assignment at column 0, not from the commented example above
    it — those are two different strings, and it was the *example* that was
    stale. Splitting on "window.RL_API" finds the comment first and would have
    checked the wrong one.
    """
    for line in _read("docs", "config.js").split("\n"):
        if line.startswith("window.RL_API"):
            match = HOST_RE.search(line)
            assert match, f"no Render URL in: {line.strip()}"
            return match.group(1)
    raise AssertionError("docs/config.js never assigns window.RL_API")


def test_the_site_calls_the_service_the_blueprint_provisions():
    """Render serves the blueprint's service at <name>.onrender.com."""
    live, provisioned = _configured_host(), _service_name()
    assert live == provisioned, (
        f"docs/config.js calls '{live}.onrender.com' but render.yaml provisions "
        f"'{provisioned}'. A Blueprint deploy would produce a service the site "
        "never calls.")


def test_every_documented_hostname_is_the_real_one():
    """A verification command that 404s teaches the reader the wrong lesson."""
    name = _service_name()
    for doc in ("DEPLOY.md", "README.md"):
        path = os.path.join(REPO, doc)
        if not os.path.exists(path):
            continue
        hosts = set(HOST_RE.findall(_read(doc)))
        wrong = {h for h in hosts if h != name}
        assert not wrong, (
            f"{doc} references Render hosts that are not the provisioned "
            f"service '{name}': {sorted(wrong)}")


def test_the_config_example_matches_the_configured_value():
    """The commented example is what people copy; it must not point elsewhere."""
    config = _read("docs", "config.js")
    hosts = set(HOST_RE.findall(config))
    assert len(hosts) == 1, (
        f"docs/config.js mentions more than one backend host: {sorted(hosts)}")


def test_the_free_plan_sleep_is_documented():
    """The behaviour that makes a healthy backend look dead has to be written
    down, or the next person debugs a service that was only asleep."""
    deploy = _read("DEPLOY.md").lower()
    assert "sleep" in deploy or "wake" in deploy
    assert "curl -m" in deploy, "no long-timeout curl is suggested"


@pytest.mark.parametrize("field", ["healthCheckPath", "startCommand", "rootDir"])
def test_the_blueprint_still_declares_what_render_needs(field: str):
    assert field in _read("render.yaml"), f"render.yaml lost {field}"


def test_the_lab_retries_a_sleeping_backend_before_calling_it_unreachable():
    """One failed request against a free-tier service means very little."""
    lab = _read("docs", "lab.js")
    i = lab.index("async function initStatus()")
    block = lab[i: i + 2600]
    assert "waking the backend" in block, "the pill no longer explains a cold start"
    assert "for (let attempt" in block, "the pill no longer retries"
    assert "API unreachable" in block, "the give-up state has gone missing"
