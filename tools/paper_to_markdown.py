#!/usr/bin/env python3
"""Convert paper/rl_trader.tex to Markdown for the web build.

WHY A PREPROCESSOR AND NOT JUST PANDOC
Running pandoc straight at the .tex loses the things a reader most needs:

* the abstract vanishes entirely (it hangs off \\maketitle, which pandoc drops);
* all 19 \\cite{} commands disappear, leaving dangling spaces before the
  punctuation that followed them -- "evaluation choices ." -- so the paper
  silently stops attributing Henderson, PPO, GAE and the rest;
* \\begin{thebibliography} collapses into one run-on paragraph that opens with
  a stray "9" (the column-width argument), rather than a numbered list;
* \\ref{} becomes an <a href="#sec:ablation"> pointing at a LaTeX label that
  does not exist in the generated HTML, so every cross-reference is a dead link;
* \\includegraphics{fig_ablation} has no extension or directory, so the figure
  404s.

So the .tex is rewritten into a form pandoc renders faithfully, then converted.
The source paper is never modified -- it stays the authoritative artifact and
is still what builds the PDF.

USAGE
    python tools/paper_to_markdown.py            # -> paper/rl_trader.md
    python tools/paper_to_markdown.py --check    # report, write nothing

Requires pandoc, via either the system binary or `pip install pypandoc_binary`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "rl_trader.tex"
OUT = ROOT / "paper" / "rl_trader.md"

# Where the figure lives relative to the published report page.
FIGURE_SRC = "figures/fig_ablation.png"


def bib_numbers(tex: str) -> dict[str, int]:
    """\\bibitem order defines the citation numbers, exactly as LaTeX would."""
    return {k: i for i, k in enumerate(re.findall(r"\\bibitem\{([^}]+)\}", tex), 1)}


def label_numbers(tex: str) -> dict[str, str]:
    """Map every \\label to the number LaTeX would print for it.

    Sections and subsections are counted as 1, 1.1, 1.2, 2 ...; tables and
    figures each get their own running count. A label belongs to whichever
    numbered thing most recently opened above it.
    """
    out: dict[str, str] = {}
    sec = sub = tab = fig = 0
    current = ""
    for m in re.finditer(
        r"\\section\{|\\subsection\{|\\begin\{table\}|\\begin\{figure\}|\\label\{([^}]+)\}",
        tex,
    ):
        token = m.group(0)
        if token.startswith("\\section"):
            sec, sub = sec + 1, 0
            current = str(sec)
        elif token.startswith("\\subsection"):
            sub += 1
            current = f"{sec}.{sub}"
        elif token.startswith("\\begin{table}"):
            tab += 1
            current = str(tab)
        elif token.startswith("\\begin{figure}"):
            fig += 1
            current = str(fig)
        else:
            out[m.group(1)] = current
    return out


def preprocess(tex: str) -> tuple[str, dict]:
    """Rewrite the source into something pandoc renders without losses."""
    stats = {}
    bib = bib_numbers(tex)
    labels = label_numbers(tex)
    stats["bibitems"] = len(bib)
    stats["labels"] = len(labels)

    # 1. The abstract becomes a real section so it survives.
    tex, n = re.subn(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        lambda m: "\\section*{Abstract}" + m.group(1),
        tex,
        flags=re.S,
    )
    stats["abstract"] = n

    # 2. Citations become bracketed numbers matching the bibliography order.
    def cite(m: re.Match) -> str:
        keys = [k.strip() for k in m.group(1).split(",")]
        nums = [str(bib[k]) for k in keys if k in bib]
        return f"[{', '.join(nums)}]" if nums else ""

    tex, n = re.subn(r"\\cite\{([^}]+)\}", cite, tex)
    stats["cites"] = n

    # 3. Cross-references become the plain number LaTeX would have printed.
    #    A link would be worse: it would point at a label the HTML lacks.
    tex, n = re.subn(
        r"\\ref\{([^}]+)\}", lambda m: labels.get(m.group(1), "?"), tex
    )
    stats["refs"] = n

    # 4. thebibliography -> a real enumerated list.
    def bibliography(m: re.Match) -> str:
        body = m.group(1)
        items = re.split(r"\\bibitem\{[^}]+\}", body)[1:]
        rows = "\n".join(f"\\item {i.strip()}" for i in items)
        return "\\section*{References}\n\\begin{enumerate}\n" + rows + "\n\\end{enumerate}"

    tex, n = re.subn(
        r"\\begin\{thebibliography\}\{[^}]*\}(.*?)\\end\{thebibliography\}",
        bibliography,
        tex,
        flags=re.S,
    )
    stats["bibliography"] = n

    # 5. Give the figure a path that resolves from the published page.
    tex, n = re.subn(
        r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}",
        lambda m: "\\includegraphics{" + FIGURE_SRC + "}",
        tex,
    )
    stats["figures"] = n

    # 6. The one custom macro. Its DEFINITION has to go before its uses are
    #    expanded, or \newcommand{\pc}{\%} rewrites into \newcommand{\%}{\%}
    #    and pandoc dies with "Loop encountered in expanding macro %".
    tex = re.sub(r"\\newcommand\{\\pc\}\{[^}]*\}[^\n]*", "", tex)
    tex = re.sub(r"\\pc(?![a-zA-Z])", r"\\%", tex)

    return tex, stats


def convert(tex_body: str) -> str:
    try:
        import pypandoc
    except ImportError:
        sys.exit("pandoc is needed:  pip install pypandoc_binary")
    return pypandoc.convert_text(
        tex_body, "gfm", format="latex", extra_args=["--wrap=none"]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report and write nothing")
    args = ap.parse_args()

    if not TEX.exists():
        print(f"missing source: {TEX}", file=sys.stderr)
        return 2

    tex = TEX.read_text(encoding="utf-8")
    processed, stats = preprocess(tex)
    md = convert(processed)

    # Pandoc leaves the odd empty bracket where a dropped command used to be.
    md = re.sub(r" +([.,;])", r"\1", md)

    # Pandoc emits inline math as GFM's $`...`$, which the site's CommonMark
    # renderer does not understand -- it would print the delimiters literally.
    # The paper has no equation environments, so every one of these is short
    # notation ($`+275\%`$, $`[-1,1]`$); render them as plain code instead.
    math = len(re.findall(r"\$`[^`]*`\$", md))
    md = re.sub(r"\$`([^`]*)`\$", lambda m: "`" + m.group(1).replace("\\%", "%").replace("\\", "") + "`", md)
    print("  math spans flattened:", math)

    print("  abstract sections   :", stats["abstract"])
    print("  citations rewritten :", stats["cites"], f"({stats['bibitems']} references)")
    print("  cross-refs resolved :", stats["refs"], f"({stats['labels']} labels)")
    print("  bibliography lists  :", stats["bibliography"])
    print("  figures repathed    :", stats["figures"])
    print("  output              :", len(md), "chars")

    if args.check:
        print("  (--check: nothing written)")
        return 0

    OUT.write_text(md, encoding="utf-8", newline="")
    print(f"  wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
