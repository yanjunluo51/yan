#!/bin/bash
# Rebuild the flattened .tex, the PDF, and the Word version of the revised main text.
set -e
cd "$(dirname "$0")/.."
SCR=/tmp/claude-0/-home-user-yan/5364adbd-c49d-5b0b-88c9-e1c4fd24fd61/scratchpad/docx
mkdir -p "$SCR"
PANDOC=$(python3 -c "import pypandoc; print(pypandoc.get_pandoc_path())")
python3 tools/flatten.py
pdflatex -interaction=nonstopmode -halt-on-error CAIE_main_revised.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error CAIE_main_revised.tex >/dev/null
python3 tools/prep.py
"$PANDOC" "$SCR/pandoc_in.tex" -f latex -t docx --reference-doc=tools/ref_tnr.docx --number-sections -o "$SCR/raw.docx"
python3 tools/post.py "$SCR/raw.docx" CAIE_main_revised.docx
rm -f *.aux *.out *.log *.spl
echo built
