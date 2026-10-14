"""Generate every manuscript figure from the experiment JSON/registry.

Usage: python make_figures.py [figure names...]
Figures map to manuscript numbers as documented in build_manuscript.py.
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, '..', 'experiments')
FIG = os.path.join(HERE, '..', 'figures')
CACHE = os.path.join(HERE, '..', 'cache')
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, GREEN, RED, GRAY = '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8C8C8C'


def load_summary():
    with open(os.path.join(EXP, 'summary.json')) as f:
        return json.load(f)


def seed_avg_preds(tag, name):
    files = sorted(glob.glob(os.path.join(EXP, f'preds_{tag}_{name}_f*_s*.npz')))
    if not files:
        return None
    per_seed = {}
    for f in files:
        seed = int(f.split('_s')[1].split('.')[0])
        per_seed.setdefault(seed, []).append(f)
    seeds = sorted(per_seed)
    merged = {}
    for s in seeds:
        arrs = [np.load(f) for f in sorted(per_seed[s])]
        merged[s] = {k: np.concatenate([a[k] for a in arrs])
                     for k in ('pids', 'targets', 'preds')}
    ref = merged[seeds[0]]
    preds = np.mean([merged[s]['preds'] for s in seeds], axis=0)
    return {'pids': ref['pids'], 'targets': ref['targets'], 'preds': preds}


# ---------------------------------------------------------------- Fig 1
def fig1_demographics():
    meta = json.load(open(os.path.join(CACHE, 'meta.json')))
    import pandas as pd
    df = pd.DataFrame(meta['participants'])
    groups = [('ADHD', 'yes', ORANGE), ('Control', 'no', BLUE), ('Undetermined', 'undetermined', GRAY)]
    counts = [int((df['diagnosed'] == g).sum()) for _, g, _ in groups]
    ages = [df[df['diagnosed'] == g]['age'].values for _, g, _ in groups]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    axes[0].bar([g for g, _, _ in groups], counts,
                color=[c for _, _, c in groups])
    for i, c in enumerate(counts):
        axes[0].text(i, c + 1, str(c), ha='center', fontsize=9)
    axes[0].set_ylabel('Participants')
    axes[0].set_title('(a) Diagnostic group composition (n=126)', fontsize=9)

    bp = axes[1].boxplot(ages, labels=[g for g, _, _ in groups], patch_artist=True, widths=0.5)
    for patch, (_, _, c) in zip(bp['boxes'], groups):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    axes[1].set_ylabel('Age (years)')
    axes[1].set_title('(b) Age distribution by group', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig1_demographics.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4
ORDER_MAIN = ['TempAlignEEGNet', 'GazeOnly', 'EEGOnly', 'LSTM', 'Transformer',
              'LTC', 'WaveletTransformer', 'RidgeGaze', 'PersistenceTrend',
              'PersistenceZero', 'MeanPredictor']
LABELS = {'TempAlignEEGNet': 'TempAlign\n(full)', 'GazeOnly': 'Gaze-only',
          'EEGOnly': 'EEG-only', 'LSTM': 'LSTM', 'Transformer': 'Transformer',
          'LTC': 'LTC', 'WaveletTransformer': 'Wavelet\nTransf.',
          'RidgeGaze': 'Ridge\n(gaze)', 'PersistenceTrend': 'Persist.\n(trend)',
          'PersistenceZero': 'Persist.\n(no change)', 'MeanPredictor': 'Mean\npredictor'}


def fig4_main():
    s = load_summary()
    names = [n for n in ORDER_MAIN if n in s['models']]
    r = [s['models'][n]['r_col_mean'] for n in names]
    ci = [s['models'][n]['r_col_ci95'] for n in names]
    cols = [GREEN if n == 'TempAlignEEGNet' else
            (BLUE if n in ('GazeOnly', 'EEGOnly') else GRAY) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={'width_ratios': [1.3, 1]})
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x, r, yerr=ci, color=cols, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n] for n in names], fontsize=6.5)
    ax.set_ylabel(r'$r_{\Delta col}$ (held-out)')
    ax.axhline(0, color='k', lw=0.6)
    # significance vs full model
    paired = s.get('paired', {})
    for i, n in enumerate(names):
        if n in paired:
            p = paired[n]['r_col']['wilcoxon_p']
            star = '**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.')
            top = r[i] + ci[i] + 0.012
            ax.text(i, top, star, ha='center', fontsize=6.5)
    ax.set_title('(a) Model comparison (participant-level 5-fold CV)', fontsize=9)

    sa = seed_avg_preds('main', 'TempAlignEEGNet')
    ax2 = axes[1]
    if sa is not None:
        t, p = sa['targets'], sa['preds']
        lim = [min(t[:, 0].min(), p[:, 0].min()), max(t[:, 0].max(), p[:, 0].max())]
        ax2.scatter(t[:, 0], p[:, 0], s=2, alpha=0.15, color=BLUE)
        q = np.percentile(np.abs(t[:, 0]), [0.5, 99.5])
        ax2.set_xlim(q[0] * 1.1, q[1] * 1.1)
        ax2.set_ylim(q[0] * 1.1, q[1] * 1.1)
        pooled = s['models']['TempAlignEEGNet']['pooled']
        ax2.set_xlabel(r'Actual $\Delta$col (grid units)')
        ax2.set_ylabel(r'Predicted $\Delta$col')
        ax2.set_title(f"(b) Held-out predictions\nr={pooled['r_col']:.3f}, "
                      f"$R^2$={pooled['R2_col']:.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig4_main_results.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 5
def fig5_ablation():
    s = load_summary()
    names = [n for n in ['TempAlignEEGNet', 'NoMask', 'FrontalExcluded', 'TempStructNet']
             if n in s['models']]
    labels = {'TempAlignEEGNet': 'Full model', 'NoMask': 'No temporal\nmask',
              'FrontalExcluded': 'Frontal channels\nexcluded',
              'TempStructNet': 'Learned mask'}
    r = [s['models'][n]['r_col_mean'] for n in names]
    ci = [s['models'][n]['r_col_ci95'] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={'width_ratios': [1, 1.2]})
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x, r, yerr=ci, color=[GREEN] + [BLUE] * (len(names) - 1), capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[n] for n in names], fontsize=7)
    ax.set_ylabel(r'$r_{\Delta col}$')
    ax.set_title('(a) Ablations under identical CV folds', fontsize=9)

    ax = axes[1]
    T, S = 256, 50
    i = np.arange(T)[:, None]
    j = np.arange(S)[None, :]
    allowed = (j <= np.floor(i * S / T)).astype(float)
    ax.imshow(allowed, aspect='auto', origin='lower', cmap='Blues',
              extent=[0, S, 0, T], vmin=0, vmax=1.2)
    ax.plot(np.arange(S), np.arange(S) * T / S, 'r--', lw=1,
            label='boundary $j=\\lfloor iS/T\\rfloor$')
    ax.set_xlabel('Gaze position $j$')
    ax.set_ylabel('EEG position $i$')
    ax.set_title('(b) Fixed temporal mask $M$ (blue = allowed,\nwhite = $-10^9$ penalty)', fontsize=9)
    ax.legend(fontsize=6.5, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig5_ablation_mask.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 6 (attention)
def fig6_attention():
    path = os.path.join(EXP, 'mask_verification.json')
    if not os.path.exists(path):
        return
    ver = json.load(open(path))
    attn = np.array(ver['fixed_mask']['mean_attention'])  # (256, 50)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7),
                             gridspec_kw={'width_ratios': [1.4, 1]})
    ax = axes[0]
    im = ax.imshow(attn.T, aspect='auto', origin='lower', cmap='viridis',
                   extent=[0, 256, 0, 50])
    T, S = 256, 50
    j = np.arange(S)
    ax.plot(j * T / S, j, 'r--', lw=1, label='temporal boundary $j=iS/T$')
    ax.set_xlabel('EEG position $i$ (time)')
    ax.set_ylabel('Gaze position $j$ (time)')
    ax.set_title('(a) Mean cross-attention weights', fontsize=9)
    ax.legend(fontsize=6.5, loc='upper left')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    prof = attn.max(axis=1)  # per EEG row
    ax.plot(np.arange(256) / 256 * 2.0, prof, color=BLUE, lw=1)
    ax.set_xlabel('Time within window (s)')
    ax.set_ylabel('Max attention weight')
    ax.set_title('(b) Attention concentration over time', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig6_attention.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 7
def fig7_channels():
    path = os.path.join(EXP, 'channel_zeroout.json')
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    ch = d['channels']
    mean = np.array(d['mean_delta_mse'])
    std = np.array(d['std_delta_mse'])
    order = np.argsort(-mean)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    ax = axes[0]
    ax.barh(range(len(ch)), mean[order][::-1], xerr=std[order][::-1],
            color=BLUE, capsize=2)
    ax.set_yticks(range(len(ch)))
    ax.set_yticklabels([ch[i] for i in order][::-1], fontsize=7)
    ax.set_xlabel('MSE increase when channel zeroed (z-score$^2$)')
    ax.set_title('(a) Per-channel contribution (gaze changes)', fontsize=9)

    ax = axes[1]
    pos = {'AF3': (-0.35, 0.9), 'AF4': (0.35, 0.9), 'F7': (-0.7, 0.6), 'F8': (0.7, 0.6),
           'F3': (-0.3, 0.65), 'F4': (0.3, 0.65), 'FC5': (-0.6, 0.3), 'FC6': (0.6, 0.3),
           'T7': (-0.85, 0.0), 'T8': (0.85, 0.0), 'P7': (-0.6, -0.5), 'P8': (0.6, -0.5),
           'O1': (-0.3, -0.85), 'O2': (0.3, -0.85)}
    vals = np.array(d['mean_delta_mse'])
    vmax = np.abs(vals).max()
    for c, (cx, cy) in pos.items():
        i = ch.index(c)
        col = plt.cm.Reds(0.15 + 0.85 * vals[i] / vmax)
        ax.scatter(cx, cy, s=650, color=col, edgecolor='k', lw=0.5, zorder=2)
        ax.text(cx, cy, c, ha='center', va='center', fontsize=6.5, zorder=3)
    sm = plt.cm.ScalarMappable(cmap='Reds',
                               norm=plt.Normalize(0, vmax))
    plt.colorbar(sm, ax=ax, fraction=0.046, label='MSE increase')
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('(b) Topographic importance map', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig7_channels.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 8
def fig8_learned_structure():
    ver_path = os.path.join(EXP, 'mask_verification.json')
    abl_path = os.path.join(EXP, 'results_ablation.json')
    if not (os.path.exists(ver_path) and os.path.exists(abl_path)):
        return
    ver = json.load(open(ver_path))
    abl = json.load(open(abl_path))
    runs = [r for r in abl.get('TempStructNet', {}).get('runs', [])]
    delays = [r.get('delay') for r in runs if r.get('delay') is not None]
    bws = [r.get('bandwidth') for r in runs if r.get('bandwidth') is not None]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    ax = axes[0]
    d = np.array(delays)
    ax.hist(d, bins=8, color=BLUE, alpha=0.8)
    ax.axvline(np.median(d), color=RED, lw=1)
    ax.set_xlabel(r'Learned delay $\delta$ (normalized)')
    ax.set_ylabel('Count (5 folds x seeds)')
    ax.set_title('(a) Learned delay, real data', fontsize=9)

    ax = axes[1]
    syn = ver.get('synthetic', [])
    offs = sorted({s['target_offset'] for s in syn})
    data = [[s['delta_learned'] for s in syn if s['target_offset'] == o] for o in offs]
    ax.bar(range(len(offs)), [np.mean(x) for x in data],
           yerr=[np.std(x) for x in data], color=ORANGE, capsize=2)
    ax.set_xticks(range(len(offs)))
    ax.set_xticklabels([f'{o:.2f}' for o in offs])
    ax.set_xlabel('Generative target offset')
    ax.set_ylabel(r'Learned $\delta$')
    ax.set_title('(b) Synthetic control (no tracking)', fontsize=9)
    ax.axhline(np.mean([np.mean(x) for x in data]), color=GRAY, ls=':')

    ax = axes[2]
    soft_verified_delay = ver['learned_mask']['delay_normalized']
    ax.axvline(soft_verified_delay * 2000, color=GREEN, lw=2)
    ax.text(soft_verified_delay * 2000 + 15, 0.5,
            f"$\\delta$ = {soft_verified_delay:.3f}\n"
            f"({soft_verified_delay*2000:.0f} ms)",
            fontsize=8, color=GREEN)
    ax.set_xlim(0, 600); ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Milliseconds within the 2 s window')
    ax.set_title('(c) Delay conversion (corrected)', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig8_learned_structure.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 9
def fig9_lagged():
    path = os.path.join(EXP, 'lagged_analysis.json')
    if not os.path.exists(path):
        return
    res = json.load(open(path))
    imp1 = np.array([r['eeg_to_gaze_imp'] for r in res])
    p1 = np.array([r['eeg_to_gaze_p'] for r in res])
    imp2 = np.array([r['gaze_to_eeg_imp'] for r in res])
    p2 = np.array([r['gaze_to_eeg_p'] for r in res])

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    ax = axes[0]
    bp = ax.boxplot([imp1, imp2], labels=['EEG$\\rightarrow$gaze', 'Gaze$\\rightarrow$EEG'],
                    patch_artist=True, widths=0.5, showfliers=False)
    for patch, c in zip(bp['boxes'], [BLUE, ORANGE]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_ylabel('Relative MSE improvement (CV)')
    ax.set_title(f"(a) Lagged predictability (L=4, 16 Hz)\n"
                 f"sig={int((p1<0.05).sum())}/{len(p1)} vs "
                 f"{int((p2<0.05).sum())}/{len(p2)}", fontsize=9)

    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    ax.hist(p1, bins=bins, alpha=0.55, color=BLUE, label='EEG$\\rightarrow$gaze')
    ax.hist(p2, bins=bins, alpha=0.55, color=ORANGE, label='Gaze$\\rightarrow$EEG')
    ax.axvline(0.05, color=RED, ls='--', lw=1, label='p=0.05')
    ax.set_xlabel('Permutation p-value')
    ax.set_ylabel('Participants')
    ax.legend(fontsize=6.5)
    ax.set_title('(b) Null distribution check', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig9_lagged.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 11
def fig11_bands():
    path = os.path.join(EXP, 'results_ablation.json')
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    names = [n for n in ['BandTheta', 'BandAlpha', 'BandBeta'] if n in d]
    if not names:
        return
    labels = {'BandTheta': 'Theta (4-8 Hz)', 'BandAlpha': 'Alpha (8-13 Hz)',
              'BandBeta': 'Beta (13-30 Hz)'}
    r = [np.mean([x['r_col'] for x in d[n]['runs']]) for n in names]
    ci = [np.std([x['r_col'] for x in d[n]['runs']], ddof=1) / np.sqrt(len(d[n]['runs'])) * 2
          for n in names]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.arange(len(names))
    ax.bar(x, r, yerr=ci, color=[ORANGE, BLUE, GREEN], capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[n] for n in names], fontsize=7.5)
    ax.set_ylabel(r'$r_{\Delta col}$')
    ax.set_title('Single-band models (5-fold CV)', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig11_bands.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 12
def fig12_permutation():
    s = load_summary()
    names = [n for n in ['TempAlignEEGNet', 'PermutedEEG', 'PermutedTarget']
             if n in s['models']]
    if len(names) < 2:
        return
    labels = {'TempAlignEEGNet': 'Full model', 'PermutedEEG': 'Permuted EEG\n(training)',
              'PermutedTarget': 'Permuted target\n(training)'}
    r = [s['models'][n]['r_col_mean'] for n in names]
    ci = [s['models'][n]['r_col_ci95'] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x, r, yerr=ci, color=[GREEN, ORANGE, RED][:len(names)], capsize=2)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[n] for n in names], fontsize=7)
    ax.set_ylabel(r'$r_{\Delta col}$')
    ax.set_title('(a) Alignment-dependence (CV)', fontsize=9)

    ax = axes[1]
    if len(names) >= 2:
        a = seed_avg_preds('main', 'TempAlignEEGNet')
        b = seed_avg_preds('ablation', 'PermutedEEG')
        if a is not None and b is not None:
            from metrics import per_participant_metrics
            pa = {r_['pid']: r_['r_col'] for r_ in per_participant_metrics(a['targets'], a['preds'], a['pids'])}
            pb = {r_['pid']: r_['r_col'] for r_ in per_participant_metrics(b['targets'], b['preds'], b['pids'])}
            common = sorted(set(pa) & set(pb))
            ax.scatter([pb[p] for p in common], [pa[p] for p in common],
                       s=8, alpha=0.5, color=BLUE)
            lim = [min(min(pa[p], pb[p]) for p in common) - 0.05,
                   max(max(pa[p], pb[p]) for p in common) + 0.05]
            ax.plot(lim, lim, 'k--', lw=0.7)
            ax.set_xlabel('Permuted-EEG model $r$ (per participant)')
            ax.set_ylabel('Full model $r$ (per participant)')
            ax.set_title('(b) Participant-level pairing', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig12_permutation.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 13
def fig13_participants():
    s = load_summary()
    if 'TempAlignEEGNet' not in s['models']:
        return
    pp = s['models']['TempAlignEEGNet'].get('per_participant', [])
    if not pp:
        return
    r = np.array([x['r_col'] for x in pp])

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    ax = axes[0]
    ax.hist(r, bins=20, color=BLUE, alpha=0.85)
    ax.axvline(r.mean(), color=RED, lw=1.2, label=f'mean = {r.mean():.3f}')
    ax.set_xlabel(r'Per-participant $r_{\Delta col}$ (held-out)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=7)
    ax.set_title(f'(a) Interindividual variability (n={len(r)})', fontsize=9)

    ax = axes[1]
    if 'GazeOnly' in s['models']:
        pg = {x['pid']: x['r_col'] for x in s['models']['GazeOnly'].get('per_participant', [])}
        pf = {x['pid']: x['r_col'] for x in pp}
        common = sorted(set(pg) & set(pf))
        ax.scatter([pg[p] for p in common], [pf[p] for p in common],
                   s=8, alpha=0.5, color=GREEN)
        lim = [-0.2, max(max(pg[p], pf[p]) for p in common) + 0.05]
        ax.plot(lim, lim, 'k--', lw=0.7)
        d = np.mean([pf[p] - pg[p] for p in common])
        ax.set_xlabel('Gaze-only $r$ (per participant)')
        ax.set_ylabel('Full model $r$ (per participant)')
        ax.set_title(f'(b) Incremental value of EEG\nmean $\\Delta r$ = {d:+.3f}', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig13_participants.pdf'))
    plt.close(fig)


ALL = {
    'fig1': fig1_demographics, 'fig4': fig4_main, 'fig5': fig5_ablation,
    'fig6': fig6_attention, 'fig7': fig7_channels, 'fig8': fig8_learned_structure,
    'fig9': fig9_lagged, 'fig11': fig11_bands, 'fig12': fig12_permutation,
    'fig13': fig13_participants,
}

if __name__ == '__main__':
    wanted = sys.argv[1:] or list(ALL.keys())
    for k in wanted:
        try:
            ALL[k]()
            print('ok', k)
        except FileNotFoundError as e:
            print('skip', k, e)
