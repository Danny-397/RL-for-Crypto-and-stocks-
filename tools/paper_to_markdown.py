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


# Inline-math commands the paper uses, mapped to the character a reader
# expects. Order matters: wrappers that take an argument come first.
MATH_REPLACEMENTS = [
    # Wrappers -- keep the contents, drop the command.
    (r"\\mathbf\{([^{}]*)\}", r"\1"),
    (r"\\mathrm\{([^{}]*)\}", r"\1"),
    (r"\\mathit\{([^{}]*)\}", r"\1"),
    (r"\\text\{([^{}]*)\}", r"\1"),
    (r"\\textbf\{([^{}]*)\}", r"\1"),
    # Relations and operators.
    (r"\\leq(?![a-zA-Z])", "\u2264"),
    (r"\\le(?![a-zA-Z])", "\u2264"),
    (r"\\geq(?![a-zA-Z])", "\u2265"),
    (r"\\ge(?![a-zA-Z])", "\u2265"),
    (r"\\neq(?![a-zA-Z])", "\u2260"),
    (r"\\gg(?![a-zA-Z])", "\u226b"),
    (r"\\ll(?![a-zA-Z])", "\u226a"),
    (r"\\approx(?![a-zA-Z])", "\u2248"),
    # Greek, as used for hyperparameters and deltas.
    (r"\\Delta(?![a-zA-Z])", "\u0394"),
    (r"\\alpha(?![a-zA-Z])", "\u03b1"),
    (r"\\beta(?![a-zA-Z])", "\u03b2"),
    (r"\\gamma(?![a-zA-Z])", "\u03b3"),
    (r"\\delta(?![a-zA-Z])", "\u03b4"),
    (r"\\epsilon(?![a-zA-Z])", "\u03b5"),
    (r"\\theta(?![a-zA-Z])", "\u03b8"),
    (r"\\lambda(?![a-zA-Z])", "\u03bb"),
    (r"\\mu(?![a-zA-Z])", "\u03bc"),
    (r"\\pi(?![a-zA-Z])", "\u03c0"),
    (r"\\sigma(?![a-zA-Z])", "\u03c3"),
    (r"\\times(?![a-zA-Z])", "\u00d7"),
    (r"\\cdot(?![a-zA-Z])", "\u00b7"),
    (r"\\pm(?![a-zA-Z])", "\u00b1"),
    (r"\\sim(?![a-zA-Z])", "~"),
    (r"\\to(?![a-zA-Z])", "\u2192"),
    (r"\\infty(?![a-zA-Z])", "\u221e"),
    # Escapes and spacing.
    (r"\\%", "%"),
    (r"\\\$", "$"),
    (r"\\&", "&"),
    (r"\\_", "_"),
    (r"\\,", " "),
    (r"\\;", " "),
    (r"\\:", " "),
    (r"\\!", ""),
    (r"\\ ", " "),
]


def flatten_math(md: str) -> tuple[str, set[str]]:
    """Turn pandoc's $`...`$ inline math into plain inline code.

    Returns the rewritten text and the set of LaTeX commands it did not know
    how to render. Stripping backslashes wholesale is not good enough: it
    turned `\\le` into the bare word "le" and `p \\le 0.05` was published as
    "p{le}0.05". Grouping braces have to go too, since `p{=}0.787` reached the
    live page looking exactly like that.
    """
    unmapped: set[str] = set()

    def one(match: re.Match) -> str:
        body = match.group(1)
        for pattern, repl in MATH_REPLACEMENTS:
            body = re.sub(pattern, repl, body)
        # Anything still carrying a backslash is a command with no mapping.
        for name in re.findall(r"\\([a-zA-Z]+)", body):
            unmapped.add("\\" + name)
        body = re.sub(r"\\([a-zA-Z]+)", r"\1", body)
        # Grouping braces carry no meaning once the math is plain text.
        body = body.replace("{", "").replace("}", "")
        return "`" + body + "`"

    return re.sub(r"\$`([^`]*)`\$", one, md), unmapped


def convert(tex_body: str) -> str:
    try:
        import pypandoc
    except ImportError:
        sys.exit("pandoc is needed:  pip install pypandoc_binary")
    # --shift-heading-level-by=1 because LaTeX \section maps to level 1, which
    # would give the page seven <h1>s. The site's template supplies the only
    # <h1> and builds its table of contents and anchors from <h2>, so sections
    # have to land there.
    return pypandoc.convert_text(
        tex_body,
        "gfm",
        format="latex",
        extra_args=["--wrap=none", "--shift-heading-level-by=1"],
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
    md, unmapped = flatten_math(md)
    print("  math spans flattened:", math)
    if unmapped:
        print("  UNMAPPED MATH COMMANDS:", ", ".join(sorted(unmapped)),
              file=sys.stderr)
        print("  -> add them to MATH_REPLACEMENTS; they would otherwise render "
              "as bare words", file=sys.stderr)
        return 1

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
