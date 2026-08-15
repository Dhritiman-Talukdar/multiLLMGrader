#!/usr/bin/env python3
"""Build shareable versions of manuscript.md with its figures embedded.

The manuscript carries figures as text placeholders:

    *[Figure 1: `path/to/fig.png` — caption text]*

which render as nothing but italic text wherever you send the .md. This script
resolves those placeholders against the real PNGs and emits three artifacts:

    build/manuscript.html  self-contained (base64 figures), for a link or email
    build/manuscript.md    same text, real ![](...) image syntax, relative paths
    build/manuscript.pdf   via pandoc + pdflatex, for co-authors and reviewers

Run from anywhere:  python3 docs/paper/build_shareable.py
"""

import base64
import html
import mimetypes
import re
import shutil
import subprocess
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent.parent
SOURCE = PAPER_DIR / "manuscript.md"
BUILD = PAPER_DIR / "build"

# *[Figure N: `path` — optional caption spanning lines]*
PLACEHOLDER = re.compile(r"\*\[Figure\s+(\d+)\s*:\s*`([^`]+)`\s*(?:—\s*)?(.*?)\]\*", re.S)


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().rstrip(".")


def resolve(rel_path: str) -> Path:
    path = (REPO_ROOT / rel_path).resolve()
    if not path.is_file():
        sys.exit(f"missing figure: {rel_path} (looked in {path})")
    return path


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def caption_html(number: str, caption: str) -> str:
    label = f'<span class="fig-num">Figure {number}</span>'
    if not caption:
        return label
    return f"{label} {html.escape(caption)}."


def substitute(source: str, render) -> str:
    def swap(match):
        return render(match.group(1), match.group(2), collapse(match.group(3)))

    return PLACEHOLDER.sub(swap, source)


# --- HTML -------------------------------------------------------------------

def html_figure(number, rel_path, caption):
    path = resolve(rel_path)
    return (
        '<figure class="plate">'
        f'<img src="{data_uri(path)}" alt="Figure {number}. {html.escape(caption)}" '
        'loading="lazy" decoding="async">'
        f'<figcaption>{caption_html(number, caption)}</figcaption>'
        "</figure>"
    )


def pandoc(markdown: str, *args: str) -> str:
    result = subprocess.run(
        ["pandoc", "-f", "gfm", *args],
        input=markdown, capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"pandoc failed:\n{result.stderr}")
    return result.stdout


def build_html(source: str) -> Path:
    body = pandoc(substitute(source, html_figure), "-t", "html")

    # Tables and code get their own horizontal scroll so the page never does.
    body = re.sub(r"<table>", '<div class="scroll-x"><table>', body)
    body = re.sub(r"</table>", "</table></div>", body)

    # Unfilled placeholders and pending citations are real state in a draft --
    # mark them so co-authors can see what is still open.
    body = re.sub(
        r"<code>\[CITE:\s*(.*?)</code>",
        lambda m: f'<code class="pending cite">cite {m.group(1)}</code>',
        body, flags=re.S,
    )
    body = re.sub(
        r"\[(AUTHOR LIST|AFFILIATION|EMAIL|ACKNOWLEDGEMENTS)\]",
        lambda m: f'<code class="pending">{m.group(1).lower()}</code>',
        body,
    )

    toc = "\n".join(
        f'<li><a href="#{anchor}">{html.escape(text)}</a></li>'
        for anchor, text in re.findall(r'<h2 id="([^"]+)"[^>]*>(.*?)</h2>', body)
    )

    page = TEMPLATE.replace("{{TOC}}", toc).replace("{{BODY}}", body)
    out = BUILD / "manuscript.html"
    out.write_text(page)
    return out


# --- Markdown + PDF ---------------------------------------------------------

def md_figure(number, rel_path, caption):
    resolve(rel_path)
    alt = f"Figure {number}. {caption}." if caption else f"Figure {number}"
    text = f"**Figure {number}.** {caption}." if caption else f"**Figure {number}.**"
    return f"![{alt}]({rel_path})\n\n{text}"


def build_markdown(source: str) -> Path:
    out = BUILD / "manuscript.md"
    out.write_text(substitute(source, md_figure))
    return out


def build_pdf(markdown_path: Path) -> Path | None:
    # xelatex, not pdflatex: the manuscript uses real Unicode throughout
    # (U+2212 minus in every negative delta, en dashes in Bland-Altman), which
    # pdflatex refuses without per-character declarations.
    engine = shutil.which("xelatex") or shutil.which("lualatex")
    if not engine:
        return None
    out = BUILD / "manuscript.pdf"
    header = BUILD / "_pdf-header.tex"
    header.write_text(
        r"\usepackage{graphicx}"
        "\n"
        r"\setkeys{Gin}{width=\linewidth,keepaspectratio}"
        "\n"
    )
    result = subprocess.run(
        [
            "pandoc", str(markdown_path), "-f", "gfm", "-o", str(out),
            f"--pdf-engine={Path(engine).name}",
            f"--resource-path={REPO_ROOT}",
            f"--include-in-header={header}",
            "-V", "geometry:margin=1in",
            "-V", "fontsize=11pt",
            "-V", "colorlinks=true",
            "-V", "linkcolor=RoyalBlue",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    header.unlink()
    if result.returncode != 0:
        print(f"  pdf skipped -- pandoc/pdflatex error:\n{result.stderr.strip()[:800]}")
        return None
    return out


TEMPLATE = r"""<title>Human Agreement Is the Ceiling</title>
<style>
:root {
  color-scheme: light;
  --paper:    #F4F6F8;
  --surface:  #FFFFFF;
  --ink:      #171A21;
  --muted:    #59616F;
  --faint:    #838C9B;
  --rule:     #D8DEE7;
  --rule-soft:#E7EBF1;
  --accent:   #1D4E89;
  --ceiling:  #8C2438;
  --plate-bg: #FFFFFF;
  --shadow:   0 1px 2px rgba(23, 26, 33, .05), 0 8px 24px -16px rgba(23, 26, 33, .28);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --paper:    #10131A;
    --surface:  #171B24;
    --ink:      #E4E9F2;
    --muted:    #9AA4B4;
    --faint:    #737E90;
    --rule:     #2A313D;
    --rule-soft:#20262F;
    --accent:   #7FADE4;
    --ceiling:  #E0899A;
    --plate-bg: #E9ECF1;
    --shadow:   0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px -16px rgba(0, 0, 0, .8);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --paper:    #10131A;
  --surface:  #171B24;
  --ink:      #E4E9F2;
  --muted:    #9AA4B4;
  --faint:    #737E90;
  --rule:     #2A313D;
  --rule-soft:#20262F;
  --accent:   #7FADE4;
  --ceiling:  #E0899A;
  --plate-bg: #E9ECF1;
  --shadow:   0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px -16px rgba(0, 0, 0, .8);
}

:root {
  --serif: Charter, "Bitstream Charter", "Iowan Old Style", "Source Serif 4",
           "Source Serif Pro", Palatino, Georgia, serif;
  --mono: ui-monospace, "SF Mono", "JetBrains Mono", "IBM Plex Mono", Menlo,
          Consolas, monospace;
  --measure: 68ch;
  --plate: min(100%, 62rem);
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: var(--serif);
  font-size: 1.0625rem;
  line-height: 1.68;
  -webkit-font-smoothing: antialiased;
}

/* ---- shell ---- */
.shell {
  display: grid;
  grid-template-columns: 1fr;
  gap: 0;
  max-width: 96rem;
  margin: 0 auto;
  padding: 0 1.5rem 6rem;
}
@media (min-width: 62rem) {
  .shell {
    grid-template-columns: 15rem minmax(0, 1fr);
    gap: 3.5rem;
    padding-inline: 2.5rem;
  }
}

/* ---- masthead ---- */
.masthead {
  grid-column: 1 / -1;
  border-bottom: 1px solid var(--rule);
  padding: 4.5rem 0 2.75rem;
  margin-bottom: 3rem;
}
.eyebrow {
  font-family: var(--mono);
  font-size: .6875rem;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--faint);
  margin: 0 0 1.5rem;
}
.masthead h1 {
  font-size: clamp(1.9rem, 1.2rem + 2.4vw, 3rem);
  line-height: 1.12;
  font-weight: 600;
  letter-spacing: -.018em;
  text-wrap: balance;
  max-width: 22ch;
  margin: 0;
}
.byline {
  font-family: var(--mono);
  font-size: .8125rem;
  line-height: 2.1;
  color: var(--muted);
  margin: 1.75rem 0 0;
  max-width: var(--measure);
}

/* The one flourish, and it states the paper's thesis: every model sits
   under a line the humans define. */
.ceiling-mark {
  margin-top: 2.75rem;
  max-width: 34rem;
}
.ceiling-mark .line {
  border-top: 2px solid var(--ceiling);
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding-top: .5rem;
  font-family: var(--mono);
  font-size: .6875rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--ceiling);
}
.ceiling-mark .bars {
  display: flex;
  align-items: flex-end;
  gap: .3rem;
  height: 2.75rem;
  margin-top: .5rem;
}
.ceiling-mark .bars i {
  flex: 1;
  background: var(--accent);
  opacity: .5;
  border-radius: 1px 1px 0 0;
}

/* ---- rail ---- */
.rail { display: none; }
@media (min-width: 62rem) {
  .rail {
    display: block;
    grid-column: 1;
    align-self: start;
    position: sticky;
    top: 2.5rem;
    max-height: calc(100vh - 5rem);
    overflow-y: auto;
    padding-bottom: 2rem;
  }
  .rail h2 {
    font-family: var(--mono);
    font-size: .6875rem;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--faint);
    font-weight: 400;
    margin: 0 0 1rem;
  }
  .rail ol {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: .55rem;
  }
  .rail a {
    color: var(--muted);
    text-decoration: none;
    font-size: .875rem;
    line-height: 1.35;
    display: block;
    border-left: 2px solid var(--rule-soft);
    padding-left: .75rem;
  }
  .rail a:hover, .rail a:focus-visible {
    color: var(--accent);
    border-left-color: var(--accent);
  }
}

/* ---- article ---- */
article { grid-column: -2 / -1; min-width: 0; }
article > * { max-width: var(--measure); }
article > h1 { display: none; }        /* title lives in the masthead */

h2, h3 {
  font-family: var(--serif);
  font-weight: 600;
  letter-spacing: -.012em;
  text-wrap: balance;
  scroll-margin-top: 2rem;
}
h2 {
  font-size: 1.6rem;
  line-height: 1.25;
  margin: 4rem 0 1.25rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--rule);
}
h3 {
  font-size: 1.16rem;
  line-height: 1.35;
  margin: 2.5rem 0 .75rem;
  color: var(--ink);
}
p { margin: 0 0 1.15rem; }
strong { font-weight: 600; }
em { font-style: italic; }

a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: .16em; }
a:focus-visible, .rail a:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 3px;
  border-radius: 2px;
}

ul, ol { margin: 0 0 1.15rem; padding-left: 1.3rem; }
li { margin-bottom: .4rem; }

hr {
  border: 0;
  border-top: 1px solid var(--rule);
  margin: 3rem 0;
  max-width: var(--measure);
}

blockquote {
  margin: 1.5rem 0;
  padding: .9rem 1.25rem;
  border-left: 2px solid var(--rule);
  color: var(--muted);
  font-size: .96em;
  background: var(--surface);
  border-radius: 0 3px 3px 0;
}
blockquote p:last-child { margin-bottom: 0; }

/* ---- data ---- */
code {
  font-family: var(--mono);
  font-size: .84em;
  background: var(--surface);
  border: 1px solid var(--rule-soft);
  border-radius: 3px;
  padding: .1em .35em;
  overflow-wrap: anywhere;
}
code.pending {
  background: transparent;
  border: 1px dashed var(--ceiling);
  color: var(--ceiling);
  font-size: .78em;
  letter-spacing: .02em;
  text-transform: uppercase;
  padding: .12em .45em;
}
code.cite { text-transform: none; letter-spacing: 0; }

.scroll-x {
  overflow-x: auto;
  max-width: var(--plate);
  margin: 1.75rem 0 2rem;
  border: 1px solid var(--rule);
  border-radius: 4px;
  background: var(--surface);
}
table {
  border-collapse: collapse;
  width: 100%;
  font-family: var(--mono);
  font-size: .8125rem;
  font-variant-numeric: tabular-nums;
  line-height: 1.5;
}
th, td {
  text-align: left;
  padding: .6rem .9rem;
  border-bottom: 1px solid var(--rule-soft);
  white-space: nowrap;
}
th {
  font-weight: 600;
  color: var(--muted);
  font-size: .6875rem;
  letter-spacing: .07em;
  text-transform: uppercase;
  border-bottom: 1px solid var(--rule);
  background: var(--paper);
  position: sticky;
  top: 0;
}
tbody tr:last-child td { border-bottom: 0; }
td:not(:first-child), th:not(:first-child) { text-align: right; }

/* ---- figures ---- */
.plate {
  margin: 2.5rem 0 2.75rem;
  max-width: var(--plate);
  display: flex;
  flex-direction: column;
  gap: .85rem;
}
.plate img {
  display: block;
  width: 100%;
  height: auto;
  background: var(--plate-bg);
  border: 1px solid var(--rule);
  border-radius: 4px;
  box-shadow: var(--shadow);
}
.plate figcaption {
  font-family: var(--mono);
  font-size: .78rem;
  line-height: 1.6;
  color: var(--muted);
  max-width: var(--measure);
}
.fig-num {
  color: var(--ink);
  font-weight: 600;
  letter-spacing: .06em;
  text-transform: uppercase;
  font-size: .72rem;
  margin-right: .4rem;
}

@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
</style>

<div class="shell">
  <header class="masthead">
    <p class="eyebrow">Working manuscript &middot; draft with figures</p>
    <h1>Human Agreement Is the Ceiling</h1>
    <p class="byline">
      Evaluating seven frontier LLMs as rubric graders across two engineering courses<br>
      945 model gradings &middot; 755 human judgements &middot; 2 courses &middot; 3 graders each
    </p>
    <div class="ceiling-mark" aria-hidden="true">
      <div class="line"><span>Human ceiling</span><span>ICC 0.956</span></div>
      <div class="bars">
        <i style="height:36%"></i><i style="height:31%"></i><i style="height:44%"></i>
        <i style="height:28%"></i><i style="height:39%"></i><i style="height:24%"></i>
        <i style="height:33%"></i>
      </div>
    </div>
  </header>

  <nav class="rail" aria-label="Sections">
    <h2>Contents</h2>
    <ol>{{TOC}}</ol>
  </nav>

  <article>{{BODY}}</article>
</div>
"""


def main() -> None:
    BUILD.mkdir(exist_ok=True)
    source = SOURCE.read_text()

    found = len(PLACEHOLDER.findall(source))
    print(f"manuscript.md: {found} figure placeholders")

    page = build_html(source)
    print(f"  {page.relative_to(REPO_ROOT)}  ({page.stat().st_size / 1e6:.1f} MB)")

    markdown = build_markdown(source)
    print(f"  {markdown.relative_to(REPO_ROOT)}")

    pdf = build_pdf(markdown)
    if pdf:
        print(f"  {pdf.relative_to(REPO_ROOT)}  ({pdf.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
