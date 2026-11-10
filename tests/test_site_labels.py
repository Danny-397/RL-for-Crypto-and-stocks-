"""Every form control on the site has an accessible name (a <label for>, aria-label or aria-labelledby).

The live-ticker box had only a placeholder, which screen readers don't treat as a label.
"""
from html.parser import HTMLParser
from pathlib import Path

INDEX = Path(__file__).resolve().parent.parent / "docs" / "index.html"


class _Controls(HTMLParser):
    def __init__(self):
        super().__init__()
        self.controls, self.label_for, self.in_label = [], set(), 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "label":
            self.in_label += 1
            if a.get("for"):
                self.label_for.add(a["for"])
        elif tag in ("input", "select", "textarea") and a.get("type") not in ("hidden", "submit", "button"):
            self.controls.append((a, self.in_label > 0))

    def handle_endtag(self, tag):
        if tag == "label" and self.in_label:
            self.in_label -= 1


def test_every_form_control_has_an_accessible_name():
    p = _Controls()
    p.feed(INDEX.read_text(encoding="utf-8"))
    assert p.controls
    unnamed = [a.get("id") or str(a) for a, wrapped in p.controls
               if not (wrapped or a.get("aria-label") or a.get("aria-labelledby") or a.get("id") in p.label_for)]
    assert unnamed == []
