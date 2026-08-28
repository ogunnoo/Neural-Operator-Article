#!/usr/bin/env bash
# Render neural_operator_experiment.md to a print-ready PDF.
#
# Requirements:
#   brew install pandoc          (markdown -> self-contained HTML, KaTeX math)
#   Google Chrome                (HTML -> PDF; no LaTeX install needed)
#   .venv with pypdf + reportlab (page-number/footer stamping)
#
# Network is needed only on the FIRST run, so pandoc can fetch and inline the
# KaTeX CSS/JS/fonts. The resulting HTML is fully self-contained.
set -euo pipefail
cd "$(dirname "$0")"

SRC="neural_operator_experiment.md"
OUT="Neural Operator Experiment - Lid-Driven Cavity DeepONet.pdf"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
PY="${PY:-./.venv/bin/python}"

mkdir -p build

echo "[1/3] markdown -> self-contained HTML"
pandoc "$SRC" \
  --from markdown+tex_math_dollars+pipe_tables+implicit_figures \
  --to html5 --standalone --katex --embed-resources \
  --metadata pagetitle="Building a Simple Neural Operator Experiment" \
  --metadata lang=en \
  --css build/print.css \
  --output build/article.html

echo "[2/3] HTML -> PDF (headless Chrome)"
"$CHROME" --headless=new --disable-gpu --no-sandbox --no-pdf-header-footer \
  --run-all-compositor-stages-before-draw --virtual-time-budget=60000 \
  --print-to-pdf="$PWD/build/article_raw.pdf" \
  "file://$PWD/build/article.html" 2>/dev/null

echo "[3/3] stamping footer + page numbers + metadata"
"$PY" build/stamp.py build/article_raw.pdf "$OUT"

echo "Done -> $OUT"
