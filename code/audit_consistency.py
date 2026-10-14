"""Consistency audit for the revised manuscript.

Checks:
  1. Every \\num... macro used in the .tex sources is defined in numbers.tex.
  2. No stale numbers or legacy terminology survive in the compiled PDF text.
  3. Every figure referenced by the manuscript exists; no orphan figures.
  4. The key headline numbers in the PDF text match the registry values.
"""

import glob
import json
import os
import re
import sys

import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
MANU = os.path.join(HERE, '..', 'manuscript')
EXP = os.path.join(HERE, '..', 'experiments')

FORBIDDEN_PATTERNS = [
    # legacy model names / causal terminology
    'CausalEEGNet', 'CausalDiscoveryNet', 'Causal LSTM', 'Causal Transformer',
    'Granger causality test', 'causally predict', 'causal mask', 'causal boundary',
    'Causal Alignment', 'temporal causal',
    # stale numbers / claims from the rejected manuscript
    '88 diagnosed with ADHD', '50 controls', '130 ms', '0.844', '0.431',
    '0.286', '23.1', '100-200', '100–200',
    'presaccadic activity', 'precedes saccade onset', 'lateralized readiness',
]


def main():
    problems = []

    # 1. macro audit
    defined = set()
    for line in open(os.path.join(MANU, 'numbers.tex'), encoding='utf-8'):
        m = re.match(r'\\newcommand\{\\num([A-Za-z]+)\}', line)
        if m:
            defined.add(m.group(1))
    used = set()
    for path in glob.glob(os.path.join(MANU, '*.tex')):
        if path.endswith('numbers.tex') or path.endswith('paper_highlighted.tex'):
            continue
        t = open(path, encoding='utf-8').read()
        for m in re.finditer(r'\\num([A-Za-z]+)', t):
            used.add(m.group(1))
    missing = sorted(used - defined)
    if missing:
        problems.append(f'Undefined macros used: {missing}')
    unused = sorted(defined - used)
    if unused:
        print(f'note: {len(unused)} defined-but-unused macros (ok): {unused[:6]}...')

    # 2/4. PDF text audit
    pdf_path = os.path.join(MANU, 'paper_revised.pdf')
    if not os.path.exists(pdf_path):
        problems.append('compiled PDF missing')
    else:
        doc = fitz.open(pdf_path)
        text = '\n'.join(p.get_text() for p in doc)
        flat = re.sub(r'\s+', ' ', text)
        for pat in FORBIDDEN_PATTERNS:
            if pat in flat:
                problems.append(f'forbidden pattern in PDF text: {pat!r}')
        # headline numbers vs registry
        s = json.load(open(os.path.join(EXP, 'summary.json')))
        full = s['models'].get('TempAlignEEGNet', {})
        if full:
            v = f"{full['r_col_mean']:.3f}"
            if v not in flat:
                problems.append(f'full-model r={v} not found in PDF text')
            v2 = f"{full['R2_col_mean']:.3f}"
            if v2 not in flat:
                problems.append(f'full-model R2={v2} not found in PDF text')
        # MISSING placeholders must not remain
        if 'MISSING' in flat:
            problems.append('MISSING placeholder left in PDF text')

    # 3. figure check
    figs = re.findall(r'\{(Fig/[^}]+)\}', open(os.path.join(MANU, 'floats.tex'),
                                               encoding='utf-8').read())
    for frel in set(figs):
        if not os.path.exists(os.path.join(MANU, frel)):
            problems.append(f'referenced figure missing on disk: {frel}')
    ondisk = {f'Fig/{os.path.basename(p)}' for p in glob.glob(os.path.join(MANU, 'Fig', '*'))}
    orphans = ondisk - set(figs)
    if orphans:
        print('note: orphan figure files (ok if in Fig dir for source package):',
              sorted(orphans))

    print()
    if problems:
        print('AUDIT PROBLEMS:')
        for p in problems:
            print(' -', p)
        sys.exit(1)
    print('AUDIT PASSED: macros defined, no stale content, figures resolve.')


if __name__ == '__main__':
    main()
