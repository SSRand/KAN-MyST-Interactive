"""Convert the KAN paper's LaTeX source into MyST markdown.

One-off script tailored to `arXiv-2404.19756v5/kan.tex`. Reads that file,
walks the LaTeX patterns the paper actually uses (sections, figures,
equations, citations, lists, inline formatting), and writes paper.md and
the converted figures into this project.

Run from the repo root:
    uv run python scripts/convert_paper.py

Idempotent — re-running overwrites paper.md and re-syncs content/figs/.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path("/Users/hanchangjiang.rand/Works/Evidence/arXiv-2404.19756v5")
SOURCE_TEX = SOURCE_DIR / "kan.tex"
SOURCE_BIB = SOURCE_DIR / "ref.bib"
SOURCE_FIGS = SOURCE_DIR / "figs"

PAPER_MD = ROOT / "paper.md"
PAPER_BIB = ROOT / "paper.bib"
FIGS_DIR = ROOT / "content" / "figs"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def strip_comments(text: str) -> str:
    """Drop % comments (preserving the backslash-escaped `\\%`)."""
    text = re.sub(r"\\begin\{comment\}.*?\\end\{comment\}", "", text, flags=re.DOTALL)
    out_lines = []
    for line in text.splitlines():
        cut = None
        for i, ch in enumerate(line):
            if ch == "%" and (i == 0 or line[i - 1] != "\\"):
                cut = i
                break
        out_lines.append(line if cut is None else line[:cut])
    return "\n".join(out_lines)


def extract_body(text: str) -> str:
    """Keep only the content between `\\begin{document}` and `\\end{document}`."""
    start = text.find("\\begin{document}")
    end = text.find("\\end{document}")
    if start < 0 or end < 0:
        return text
    return text[start + len("\\begin{document}") : end]


def strip_title_block(text: str) -> str:
    """Remove `\\maketitle` and surrounding orphan macros — the title comes from
    MyST frontmatter.

    The abstract is rendered as an inline `**Abstract.** …` paragraph rather
    than as `## Abstract`, so it doesn't consume heading number 1 and shift
    every section down by one.
    """
    text = re.sub(r"\\maketitle", "", text)
    text = re.sub(r"\\appendix", "\n## Appendices\n", text)
    text = re.sub(r"\\bibliographystyle\{[^}]+\}", "", text)
    text = re.sub(r"\\bibliography\{[^}]+\}", "", text)
    text = re.sub(r"\\thanks\{[^{}]*\}", "", text)
    text = re.sub(r"\\footnote\{([^{}]*)\}", r" (\1)", text)
    # LaTeX TOC plumbing — no equivalent in MyST.
    text = re.sub(r"\\addtocontents\{[^{}]*\}\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "", text)
    text = re.sub(r"\\setcounter\{[^{}]*\}\{[^{}]*\}", "", text)
    text = re.sub(r"\\tableofcontents\b", "", text)
    # Drop the `\small` size modifier that hugs `\begin{abstract}` and emit an
    # inline bold prefix instead of a heading so numbering starts at 1.
    text = re.sub(r"\\begin\{abstract\}\s*\\?small\b\s*", "\n**Abstract.** ", text)
    text = re.sub(r"\\begin\{abstract\}\s*", "\n**Abstract.** ", text)
    text = re.sub(r"\\end\{abstract\}", "", text)
    return text


# ---------------------------------------------------------------------------
# Math macro expansion
# ---------------------------------------------------------------------------

# Standard LaTeX math function names. `{\rm X}` for any of these becomes the
# proper `\X` operator; anything else becomes `\mathrm{X}` so braces stay
# balanced and superscripts / subscripts group correctly.
MATH_FUNCTIONS = {
    "exp", "sin", "cos", "tan", "log", "ln",
    "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan", "arctanh",
    "sec", "csc", "cot",
    "min", "max", "sup", "inf", "lim", "argmin", "argmax",
    "det", "tr", "Re", "Im", "Pr", "deg", "dim", "ker",
    "mod",
}


def _convert_roman(match: re.Match) -> str:
    body = match.group(1).strip()
    if body in MATH_FUNCTIONS:
        return f"\\{body}"
    return f"\\mathrm{{{body}}}"


def expand_macros(text: str) -> str:
    """Expand custom KAN-paper macros and the deprecated `{\\rm X}` form.

    Source preamble declares:
        \\newcommand{\\mat}[1]{\\mathbf{#1}}
        \\newcommand{\\x}{\\mat{x}}
        \\newcommand{\\lag}{\\mathcal{L}}
    so `\\mat{...}`, `\\x`, `\\lag` survive into the body and would render
    literally without expansion. We also convert `{\\rm X}` everywhere
    (inside or outside math) into either a proper `\\X` operator or
    `\\mathrm{X}` to preserve grouping for superscripts.
    """
    # Expand the three custom macros from the preamble.
    text = re.sub(r"\\mat\{([^{}]+)\}", r"\\mathbf{\1}", text)
    text = re.sub(r"\\lag\b", r"\\mathcal{L}", text)
    text = re.sub(r"\\x\b", r"\\mathbf{x}", text)

    # `{\rm X}` → `\X` (when X is a math function) or `\mathrm{X}` otherwise.
    text = re.sub(r"\{\\rm\s+([^{}]+)\}", _convert_roman, text)

    # The deprecated `{\style X}` scoping form, in both spaced (`{\mathbf X}`)
    # and unspaced (`{\mathbf\Phi}`) variants — convert to proper
    # `\style{X}` arguments so MathJax renders the font shift correctly.
    for style in ("mathbf", "mathcal", "mathbb", "mathit", "mathsf", "mathtt"):
        text = re.sub(
            rf"\{{\\{style}\s+([^{{}}]+)\}}", rf"\\{style}{{\1}}", text
        )
        text = re.sub(
            rf"\{{\\{style}(\\[A-Za-z]+)\}}", rf"\\{style}{{\1}}", text
        )

    return text


def _sanitise_label(label: str) -> str:
    """Replace whitespace with hyphens so anchors stay valid markdown."""
    return re.sub(r"\s+", "-", label.strip())


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def sync_figures() -> dict[str, str]:
    """Copy raster figures, convert PDFs to PNGs via qlmanage. Returns a map
    from the LaTeX-style include path to the relative path used in paper.md.
    """
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    qlmanage = shutil.which("qlmanage")
    mapping: dict[str, str] = {}

    for path in sorted(SOURCE_FIGS.iterdir()):
        suffix = path.suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg"):
            target = FIGS_DIR / path.name
            shutil.copyfile(path, target)
            mapping[path.name] = f"content/figs/{path.name}"
        elif suffix == ".pdf":
            png_name = f"{path.stem}.png"
            target = FIGS_DIR / png_name
            if qlmanage:
                try:
                    subprocess.run(
                        [qlmanage, "-t", "-s", "1800", "-o", str(FIGS_DIR), str(path)],
                        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    rendered = FIGS_DIR / f"{path.name}.png"
                    if rendered.exists():
                        rendered.rename(target)
                except Exception:
                    pass
            if target.exists():
                mapping[path.name] = f"content/figs/{png_name}"
            else:
                mapping[path.name] = f"content/figs/{path.name}"  # placeholder
    return mapping


def figure_block(content: str, fig_map: dict[str, str], counter: list[int]) -> str:
    """Convert a `\\begin{figure}…\\end{figure}` body to a MyST :::{figure} directive."""
    counter[0] += 1
    includes = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", content)
    caption_match = re.search(r"\\caption\{(.+?)\}", content, flags=re.DOTALL)
    label_match = re.search(r"\\label\{([^{}]+)\}", content)
    caption_text = caption_match.group(1).strip() if caption_match else ""
    label = _sanitise_label(label_match.group(1)) if label_match else f"fig-{counter[0]}"
    width_match = re.search(r"width\s*=\s*([0-9.]+)\\linewidth", content)
    width_pct = f"{int(float(width_match.group(1)) * 100)}%" if width_match else "100%"

    if not includes:
        return f"<!-- figure {label} ({counter[0]}): no \\includegraphics found -->\n"

    src = includes[0].strip().lstrip("./")
    src_name = Path(src).name
    rel_path = fig_map.get(src_name, f"content/figs/{src_name}")

    caption = process_inline(caption_text) if caption_text else f"Figure {counter[0]}"
    return (
        f":::{{figure}} {rel_path}\n"
        f":label: {label}\n"
        f":width: {width_pct}\n\n"
        f"{caption}\n"
        f":::\n"
    )


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


def math_block(body: str, env: str) -> str:
    """Convert `\\begin{equation|align|...}` bodies to a MyST math directive."""
    inner = body
    label_match = re.search(r"\\label\{([^{}]+)\}", inner)
    inner = re.sub(r"\\label\{[^{}]+\}", "", inner)
    inner = re.sub(r"\\tag\{[^{}]+\}", "", inner)
    inner = re.sub(rf"\\begin\{{{re.escape(env)}\*?\}}", "", inner, count=1)
    inner = re.sub(rf"\\end\{{{re.escape(env)}\*?\}}", "", inner, count=1).strip()

    label_line = f":label: {_sanitise_label(label_match.group(1))}\n" if label_match else ""

    if env in ("align", "align*", "aligned"):
        return f":::{{math}}\n{label_line}\\begin{{aligned}}\n{inner}\n\\end{{aligned}}\n:::\n"
    return f":::{{math}}\n{label_line}{inner}\n:::\n"


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------


def list_block(body: str, env: str) -> str:
    """Convert `\\begin{itemize|enumerate}` to markdown lists."""
    inner = re.sub(rf"\\begin\{{{env}\}}(?:\[[^\]]*\])?", "", body, count=1)
    inner = re.sub(rf"\\end\{{{env}\}}", "", inner, count=1)
    items = re.split(r"\\item\s*", inner)
    items = [it.strip() for it in items if it.strip()]
    bullet = "1." if env == "enumerate" else "-"
    lines = []
    for it in items:
        body_md = process_inline(it).strip()
        first, *rest = body_md.split("\n")
        lines.append(f"{bullet} {first}")
        for line in rest:
            lines.append(f"   {line}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Inline formatting
# ---------------------------------------------------------------------------


def process_inline(text: str) -> str:
    """Convert inline LaTeX markup that survives outside math blocks."""
    # Stash inline math so we don't touch its contents.
    placeholders: list[str] = []

    def stash(content: str) -> str:
        placeholders.append(content)
        return f"@@MATH{len(placeholders) - 1}@@"

    text = re.sub(r"\$([^$]+)\$", lambda m: stash(f"${m.group(1)}$"), text)

    # Citations: \cite{a,b}, \citep{a}, \citet{a}.
    def cite_replace(match):
        keys = [k.strip() for k in match.group(1).split(",")]
        formatted = "; ".join(f"@{k}" for k in keys)
        return f"[{formatted}]"

    text = re.sub(r"\\cite[ptpaal]*\{([^{}]+)\}", cite_replace, text)

    # Cross-references → MyST [](#label). Labels can carry whitespace in the
    # source (`\label{approx thm}`) so we sanitise the target.
    text = re.sub(
        r"\\(?:eqref|autoref|Cref|cref|ref)\{([^{}]+)\}",
        lambda m: f"[](#{_sanitise_label(m.group(1))})",
        text,
    )

    # URLs.
    text = re.sub(r"\\url\{([^{}]+)\}", r"<\1>", text)
    text = re.sub(r"\\href\{([^{}]+)\}\{([^{}]+)\}", r"[\2](\1)", text)

    # Inline formatting.
    text = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", text)
    text = re.sub(r"\\textit\{([^{}]*)\}", r"*\1*", text)
    text = re.sub(r"\\emph\{([^{}]*)\}", r"*\1*", text)
    text = re.sub(r"\\texttt\{([^{}]*)\}", r"`\1`", text)
    # The deprecated `{\bf X}` and `{\it X}` scoping forms still appear in
    # academic LaTeX. Convert before the catch-all stripper eats `\bf`.
    # ({\rm X} is already handled by expand_macros so math is preserved.)
    text = re.sub(r"\{\\bf\s+([^{}]+)\}", r"**\1**", text)
    text = re.sub(r"\{\\it\s+([^{}]+)\}", r"*\1*", text)

    # Strip orphan labels and stale formatting macros.
    text = re.sub(r"\\label\{[^{}]+\}", "", text)
    for macro in [r"\\small", r"\\large", r"\\centering", r"\\noindent",
                  r"\\quad", r"\\qquad", r"\\,", r"\\;", r"\\:", r"\\!",
                  r"\\AND", r"\\And", r"\\par"]:
        text = re.sub(macro, " ", text)

    # Quote-style replacements.
    text = text.replace("``", '"').replace("''", '"')
    text = re.sub(r"(?<![A-Za-z])---(?![A-Za-z])", "—", text)
    text = re.sub(r"(?<![A-Za-z])--(?![A-Za-z])", "–", text)
    text = text.replace("~", " ")

    # Restore inline math.
    for i, content in enumerate(placeholders):
        text = text.replace(f"@@MATH{i}@@", content)

    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Environments — top-level walker
# ---------------------------------------------------------------------------


def find_matching_end(text: str, env: str, start: int) -> int:
    """Find the index just past the matching `\\end{env}` for `\\begin{env}` at start."""
    pattern_begin = re.compile(rf"\\begin\{{{re.escape(env)}\*?\}}")
    pattern_end = re.compile(rf"\\end\{{{re.escape(env)}\*?\}}")
    depth = 1
    cursor = start
    while depth and cursor < len(text):
        b = pattern_begin.search(text, cursor + 1)
        e = pattern_end.search(text, cursor + 1)
        if e is None:
            return -1
        if b is not None and b.start() < e.start():
            depth += 1
            cursor = b.end()
        else:
            depth -= 1
            cursor = e.end()
    return cursor


ENV_BLOCKS = {
    "figure": "figure",
    "figure*": "figure",
    "equation": "equation",
    "equation*": "equation",
    "align": "align",
    "align*": "align",
    "itemize": "itemize",
    "enumerate": "enumerate",
    "table": "table",
    "table*": "table",
}


def walk_body(text: str, fig_map: dict[str, str]) -> str:
    """Replace tracked environments with their MyST equivalents and process the
    remaining prose."""
    fig_counter = [0]
    out: list[str] = []
    cursor = 0
    begin_re = re.compile(r"\\begin\{([a-zA-Z*]+)\}")

    while cursor < len(text):
        match = begin_re.search(text, cursor)
        if match is None:
            out.append(text[cursor:])
            break

        env_raw = match.group(1)
        env = ENV_BLOCKS.get(env_raw)
        if env is None:
            out.append(text[cursor : match.end()])
            cursor = match.end()
            continue

        out.append(text[cursor : match.start()])
        end_idx = find_matching_end(text, env_raw, match.start())
        if end_idx < 0:
            out.append(text[match.start() :])
            break

        block = text[match.start() : end_idx]

        if env == "figure":
            out.append("\n" + figure_block(block, fig_map, fig_counter) + "\n")
        elif env in ("equation", "align"):
            out.append("\n" + math_block(block, env_raw) + "\n")
        elif env in ("itemize", "enumerate"):
            out.append("\n" + list_block(block, env_raw) + "\n")
        elif env == "table":
            label = re.search(r"\\label\{([^{}]+)\}", block)
            cap = re.search(r"\\caption\{(.+?)\}", block, flags=re.DOTALL)
            cap_txt = process_inline(cap.group(1)) if cap else "Table"
            anchor = f"({label.group(1)})=\n" if label else ""
            out.append(
                f"\n{anchor}> **Table — {cap_txt}**\n>\n> *Table omitted from this conversion;"
                f" see the [arXiv source](https://arxiv.org/abs/2404.19756) for the original.*\n"
            )
        cursor = end_idx

    return "".join(out)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


SECTION_DEPTH = {
    "section": "##",
    "subsection": "###",
    "subsubsection": "####",
    "paragraph": "#####",
}


def replace_sections(text: str) -> str:
    """Convert \\section{X}[\\label{Y}] to markdown heading with anchor."""
    def repl(match):
        depth = SECTION_DEPTH[match.group(1)]
        title = match.group(2).strip()
        label_match = re.search(r"\\label\{([^{}]+)\}", match.group(0))
        anchor = f"({_sanitise_label(label_match.group(1))})=\n" if label_match else ""
        return f"\n{anchor}{depth} {title}\n"

    return re.sub(
        r"\\(section|subsection|subsubsection|paragraph)\{([^{}]+)\}(?:\\label\{[^{}]+\})?",
        repl,
        text,
    )


# ---------------------------------------------------------------------------
# Final cleanup
# ---------------------------------------------------------------------------


def cleanup(text: str) -> str:
    """Post-process — strip remaining LaTeX macros, collapse whitespace.

    Math content (`:::{math}…:::` blocks, `$…$` inline, `$$…$$` display) is
    stashed first so the catch-all command stripper doesn't eat LaTeX inside
    formulas.
    """
    placeholders: list[str] = []

    def stash(match):
        placeholders.append(match.group(0))
        return f"@@PROTECT{len(placeholders) - 1}@@"

    text = re.sub(r":::\{math\}.*?:::", stash, text, flags=re.DOTALL)
    text = re.sub(r"\$\$[^$]+\$\$", stash, text)
    text = re.sub(r"\$[^$]+\$", stash, text)

    # Fallback: catch any \ref / \cite that process_inline missed before the
    # generic command stripper eats them.
    text = re.sub(
        r"\\(?:eqref|autoref|Cref|cref|ref)\{([^{}]+)\}",
        lambda m: f"[](#{_sanitise_label(m.group(1))})",
        text,
    )
    text = re.sub(r"\\(?:textbf|textit|emph)\{([^{}]+)\}", lambda m: f"**{m.group(1)}**" if m.group(0).startswith(r"\textbf") else f"*{m.group(1)}*", text)
    text = re.sub(r"\\texttt\{([^{}]+)\}", r"`\1`", text)

    def _cite_fallback(match):
        keys = [k.strip() for k in match.group(1).split(",")]
        return "[" + "; ".join(f"@{k}" for k in keys) + "]"

    text = re.sub(r"\\cite[ptpaal]*\{([^{}]+)\}", _cite_fallback, text)
    text = re.sub(r"\{\\bf\s+([^{}]+)\}", r"**\1**", text)
    text = re.sub(r"\{\\it\s+([^{}]+)\}", r"*\1*", text)
    text = text.replace("~", " ")

    text = re.sub(r"\\(?:large|small|noindent|centering|par|hfill|vfill|hspace|vspace)\{[^{}]*\}", "", text)
    text = re.sub(r"\\(?:large|small|noindent|centering|par|hfill|vfill)\b", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\{\s*\})?", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    for i, content in enumerate(placeholders):
        text = text.replace(f"@@PROTECT{i}@@", content)

    return text.strip() + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


FRONTMATTER = """---
title: 'KAN: Kolmogorov–Arnold Networks'
subtitle: Interactive companion to arXiv:2404.19756
numbering:
  headings: true
  figure:
    template: Fig. %s
  equation:
    template: Eq. (%s)
---

"""


# ---------------------------------------------------------------------------
# Iframe insertion
# ---------------------------------------------------------------------------


def _iframe(route: str, label: str, height: int, caption: str) -> str:
    return (
        f"\n:::{{iframe}} http://localhost:8050/{route}\n"
        f":label: panel-{label}\n"
        f":width: 100%\n"
        f":height: {height}px\n\n"
        f"{caption}\n"
        f":::\n"
    )


# Each anchor is a (unique tail-of-paragraph snippet, iframe block) pair.
# We replace the snippet with `snippet + iframe` so the prose still reads
# normally and the panel sits exactly under its semantic section.
IFRAME_ANCHORS = [
    (
        "we expect the interpolation threshold to be $G=1000/15\\approx 67$, "
        "which roughly agrees with our experimentally observed value $G\\sim 50$.",
        _iframe(
            "coarse",
            "coarse",
            760,
            "Interactive — fit a `[2, 5, 1]` KAN to a target you choose. "
            "Change the grid and step count to see how the diagram and loss curve respond.",
        )
        + _iframe(
            "refine",
            "refine",
            760,
            "Interactive — train a coarse KAN, then call `model.refine()` to upsample "
            "the spline grid and continue training. The loss curve marks the refinement step.",
        ),
    ),
    (
        "where $\\mu_1,\\mu_2$ are relative magnitudes usually set to $\\mu_1=\\mu_2=1$, "
        "and $\\lambda$ controls overall regularization magnitude.",
        _iframe(
            "sparsify",
            "sparsify",
            760,
            "Interactive — continue training under L1 ($\\lambda$) and entropy ($H$) "
            "penalties. Weak edges fade out of the KAN diagram.",
        ),
    ),
    (
        "and consider a node to be important if both incoming and outgoing scores are "
        "greater than a threshold hyperparameter $\\theta=10^{-2}$ by default. All "
        "unimportant neurons are pruned.",
        _iframe(
            "prune",
            "prune",
            820,
            "Interactive — sparsify, then call `model.prune()` to remove low-magnitude "
            "nodes structurally, then refit on the surviving topology.",
        ),
    ),
    (
        "$y\\approx cf(ax+b)+d$. The fitting is done by iterative grid search of $a, b$ "
        "and linear regression.",
        _iframe(
            "symbolic",
            "symbolic",
            880,
            "Interactive — run the full pipeline (coarse → sparsify → prune → refit) and "
            "let pykan snap surviving edges to closed-form expressions. The recovered "
            "formula appears above the diagram.",
        ),
    ),
]


def insert_iframes(text: str) -> str:
    """Embed iframe directives at semantic anchor points in the converted text."""
    for anchor, iframe in IFRAME_ANCHORS:
        if anchor not in text:
            print(f"[convert] WARN: anchor missing — {anchor[:60]}…", file=sys.stderr)
            continue
        text = text.replace(anchor, anchor + iframe, 1)
    return text


def main() -> None:
    if not SOURCE_TEX.exists():
        print(f"Source not found: {SOURCE_TEX}", file=sys.stderr)
        sys.exit(1)

    print(f"[convert] reading {SOURCE_TEX}", file=sys.stderr)
    raw = SOURCE_TEX.read_text(encoding="utf-8")

    print("[convert] syncing figures…", file=sys.stderr)
    fig_map = sync_figures()
    print(f"[convert] mapped {len(fig_map)} figures", file=sys.stderr)

    text = strip_comments(raw)
    text = extract_body(text)
    text = strip_title_block(text)
    text = expand_macros(text)
    text = walk_body(text, fig_map)
    text = replace_sections(text)
    text = process_inline(text)
    text = cleanup(text)
    text = insert_iframes(text)

    body = FRONTMATTER + text
    PAPER_MD.write_text(body, encoding="utf-8")
    print(f"[convert] wrote {PAPER_MD}", file=sys.stderr)

    shutil.copyfile(SOURCE_BIB, PAPER_BIB)
    print(f"[convert] copied bibliography → {PAPER_BIB}", file=sys.stderr)


if __name__ == "__main__":
    main()
