"""Redraw the system architecture figure (Fig. 2) of the revised manuscript:
TempAlign-EEGNet without band decomposition, with the corrected mask rule."""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, GREEN, PURPLE = '#4C72B0', '#DD8452', '#55A868', '#8172B3'


def box(ax, x, y, w, h, text, color, fs=8):
    b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02',
                       fc='white', ec=color, lw=1.6)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs)
    return (x, y, w, h)


def arrow(ax, p1, p2, color='k'):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', color=color,
                                 lw=1.2, mutation_scale=12))


def main():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis('off')

    eeg_in = box(ax, 0.2, 3.9, 2.2, 1.4,
                 'EEG\n14 channels $\\times$ 256 samples\n(2 s @ 128 Hz)', BLUE)
    gaze_in = box(ax, 0.2, 1.4, 2.2, 1.4,
                  'Gaze context\n50 uniformly resampled\npoints (2 s span)', ORANGE)

    eeg_enc = box(ax, 3.2, 3.9, 2.4, 1.4,
                  'Multi-scale CNN encoder\nkernels {3, 7, 15}\n$\\rightarrow$ 256$\\times$48', BLUE)
    gaze_enc = box(ax, 3.2, 1.4, 2.4, 1.4,
                   'Gaze GRU encoder\n(unidirectional)\n$\\rightarrow$ 50$\\times$48', ORANGE)

    attn = box(ax, 6.4, 2.55, 2.6, 1.9,
               'Temporally-masked\ncross-attention ($\\times$2 layers)\n'
               'softmax($QK^{\\top}/\\sqrt{d_k}+M$)V\n'
               '$M_{ij}=0$ if $j/S \\leq i/T$, else $-10^{9}$', PURPLE, fs=7.2)

    head = box(ax, 9.7, 3.3, 2.0, 1.4,
               'Temporal attention\npooling + MLP\n$\\rightarrow (\\Delta$col, $\\Delta$row$)$', GREEN)

    arrow(ax, (2.4, 4.6), (3.2, 4.6), BLUE)
    arrow(ax, (2.4, 2.1), (3.2, 2.1), ORANGE)
    arrow(ax, (5.6, 4.6), (6.4, 3.9), BLUE)
    ax.text(5.9, 4.45, 'Q', fontsize=9, color=BLUE)
    arrow(ax, (5.6, 2.1), (6.4, 3.1), ORANGE)
    ax.text(5.9, 2.35, 'K, V', fontsize=9, color=ORANGE)
    arrow(ax, (9.0, 3.5), (9.7, 3.8), PURPLE)

    ax.text(6.35, 1.35,
            'Temporal constraint: EEG position $i$ attends gaze $j$ only up to\n'
            'its own time point (fixed mask); learned variant: binary Gumbel-\n'
            'Softmax mask with penalty $(1-G)\\cdot(-10^{9})$.',
            fontsize=7, va='top')
    ax.text(0.2, 5.75, 'TempAlign-EEGNet', fontsize=11, weight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig2_architecture.pdf'))
    fig.savefig(os.path.join(FIG, 'fig2_architecture.png'))
    plt.close(fig)
    print('saved fig2_architecture')


if __name__ == '__main__':
    main()
