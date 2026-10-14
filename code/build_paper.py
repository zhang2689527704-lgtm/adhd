"""Build the final manuscript artifacts:
  1. regenerate numbers.tex from the registry
  2. copy figures into manuscript/Fig
  3. compile the clean PDF
  4. compile the highlighted PDF (yellow highlights on \\chg{} spans)
  5. emit the audit report
Run from the repository root with the conda python.
"""

import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MANU = os.path.join(HERE, '..', 'manuscript')
FIG = os.path.join(HERE, '..', 'figures')
EXP = os.path.join(HERE, '..', 'experiments')

PY = sys.executable
FIG_MAP = {
    'fig1_demographics.pdf': 'fig1_demographics.pdf',
    'fig2_architecture.pdf': 'fig2_architecture.pdf',
    'fig4_main_results.pdf': 'fig4_main_results.pdf',
    'fig5_ablation_mask.pdf': 'fig5_ablation_mask.pdf',
    'fig6_attention.pdf': 'fig6_attention.pdf',
    'fig7_channels.pdf': 'fig7_channels.pdf',
    'fig8_learned_structure.pdf': 'fig8_learned_structure.pdf',
    'fig9_lagged.pdf': 'fig9_lagged.pdf',
    'fig11_bands.pdf': 'fig11_bands.pdf',
    'fig12_permutation.pdf': 'fig12_permutation.pdf',
    'fig13_participants.pdf': 'fig13_participants.pdf',
}


def run(args, cwd=None, log=None):
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if log:
        open(log, 'w', encoding='utf-8').write(r.stdout + r.stderr)
    return r


def main():
    # 1. numbers
    run([PY, os.path.join(HERE, 'emit_numbers.py')])
    print('numbers regenerated')

    # 2. figures
    figdir = os.path.join(MANU, 'Fig')
    os.makedirs(figdir, exist_ok=True)
    missing = []
    for src in FIG_MAP:
        s = os.path.join(FIG, src)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(figdir, src))
        else:
            missing.append(src)
    print('figures copied; missing:', missing or 'none')

    # 3. clean build: pdflatex -> bibtex -> pdflatex x2
    run(['pdflatex', '-interaction=nonstopmode', 'paper_revised.tex'],
        cwd=MANU, log=os.path.join(EXP, 'compile_clean.log'))
    run(['bibtex', 'paper_revised'], cwd=MANU,
        log=os.path.join(EXP, 'bibtex.log'))
    for i in range(2):
        r = run(['pdflatex', '-interaction=nonstopmode', 'paper_revised.tex'],
                cwd=MANU, log=os.path.join(EXP, 'compile_clean.log'))
    pdf = os.path.join(MANU, 'paper_revised.pdf')
    shutil.copy(pdf, os.path.join(MANU, '..', 'paper_clean.pdf'))
    print('clean build done ->', os.path.join(MANU, '..', 'paper_clean.pdf'))

    # 4. highlighted PDF: yellow annotations for changed lines (PyMuPDF)
    r = run([PY, os.path.join(HERE, 'make_highlighted_pdf.py')])
    print(r.stdout.strip())


if __name__ == '__main__':
    main()
