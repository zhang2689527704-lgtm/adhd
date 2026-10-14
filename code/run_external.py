"""External validation on two additional datasets, under the same protocol as
the main analysis (nested participant-level 5-fold CV, identical windowing,
identical models, predictive R2, paired participant-level tests).

Datasets:
  ds007537      - 23 participants, 64-ch Brain Products EEG @250 Hz (downsampled),
                  Tobii eye tracking, naturalistic smartphone task.
  consumer_muse - 4-ch Muse EEG @256 Hz, eye tracking; nested CV on the
                  official train split only (the authors' test split stays
                  untouched).

Neural models: TempAlignEEGNet (full), GazeOnly, EEGOnly (single seed for the
external sets; documented). Statistical baselines: ridge on gaze history,
zero-delta persistence.
"""

import glob
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_v2 import TempAlignEEGNet, TempAlignGazeOnly, TempAlignEEGOnly, count_params
from metrics import evaluate_predictions
from run_main import train_one, CFG

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, '..', 'experiments')
os.makedirs(OUT, exist_ok=True)

WINDOW_SEC = 2.0
STEP_SEC = 0.5
N_GAZE = 50
PTP_Z_THRESHOLD = 12.0
# Target horizon for the external datasets: gaze change over 0.5 s after the
# window end (these datasets provide continuous, high-rate gaze where the
# next-sample change is noise-dominated; the horizon matches the protocol
# used in the earlier external-validation work).
TARGET_OFFSET_SEC = 0.5


# --------------------------------------------------------------- loaders
def load_ds007537(data_root, pid, fs_target=250.0):
    import mne
    vhdr = os.path.join(data_root, pid, 'eeg', f'{pid}_task-phoneuse_eeg.vhdr')
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
    raw.pick(picks='eeg', verbose=False)
    if fs_target < raw.info['sfreq']:
        raw.resample(fs_target, verbose=False)
    fs = float(raw.info['sfreq'])
    eeg = raw.get_data()  # (64, T)
    for c in range(eeg.shape[0]):
        s = eeg[c].std()
        if s > 0:
            eeg[c] = (eeg[c] - eeg[c].mean()) / s

    ph = os.path.join(data_root, pid, 'beh', f'{pid}_task-phoneuse_physio.tsv.gz')
    df = pd.read_csv(ph, sep='\t')
    gp = df[df['type'] == 'gp']
    gx = pd.Series(gp['x_coordinate'].values.astype(np.float32)).interpolate().ffill().bfill().values
    gy = pd.Series(gp['y_coordinate'].values.astype(np.float32)).interpolate().ffill().bfill().values
    et = gp['et_time_s'].values.astype(np.float64)
    t_eeg = np.arange(eeg.shape[1]) / fs
    gx = np.interp(t_eeg, et, gx)
    gy = np.interp(t_eeg, et, gy)
    gaze = np.stack([gx, gy], axis=1)  # (T, 2)
    return eeg.astype(np.float32), gaze.astype(np.float32), fs


def load_consumer(file_path):
    df = pd.read_csv(file_path)
    df.replace('-', np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    eeg = df[['EEG_TP9', 'EEG_AF7', 'EEG_AF8', 'EEG_TP10']].values.T.astype(np.float32)
    for c in range(4):
        s = eeg[c].std()
        if s > 0:
            eeg[c] = (eeg[c] - eeg[c].mean()) / s
    gaze = df[['Gaze_x', 'Gaze_y']].values.astype(np.float32)
    return eeg, gaze, 256.0


# --------------------------------------------------------------- windows
def make_windows(eeg, gaze, fs):
    """Our windowing: 2 s windows, 0.5 s step, 50 uniformly resampled gaze
    pseudo-samples, target = last-in-window -> first-after-window."""
    win = int(WINDOW_SEC * fs)
    step = int(STEP_SEC * fs)
    off = int(TARGET_OFFSET_SEC * fs)
    T = eeg.shape[1]
    offsets = np.arange(N_GAZE) * (WINDOW_SEC / N_GAZE)
    out = []
    for start in range(0, T - win - off, step):
        end = start + win
        t0 = start / fs
        tk = t0 + offsets
        idx = np.clip((tk * fs).astype(int), 0, T - 1)
        ctx = gaze[idx]  # (50, 2)
        target = gaze[end + off - 1] - gaze[end - 1]
        eeg_win = eeg[:, start:end]
        ptp = float(np.max(eeg_win.max(axis=1) - eeg_win.min(axis=1)))
        out.append({'eeg': eeg_win.astype(np.float32), 'gaze': ctx.astype(np.float32),
                    'target': target.astype(np.float32), 'qc': ptp <= PTP_Z_THRESHOLD,
                    'ptp': ptp})
    return out


def build_dataset(name):
    if name == 'ds007537':
        root = os.path.join(ROOT, 'eeg_eye_net_data', 'ds007537')
        pids = sorted([d for d in os.listdir(root)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(root, d))])
        loader = lambda p: load_ds007537(root, p)
    elif name == 'consumer_muse':
        d = os.path.join(ROOT, 'eeg_eye_consumer_data', 'level-2-saccades', 'train')
        files = sorted(glob.glob(os.path.join(d, 'P*.csv')))
        by_pid = defaultdict(list)
        for f in files:
            pid = os.path.basename(f).split('_')[0]
            by_pid[pid].append(f)
        pids = sorted(by_pid)
        loader = None
    else:
        raise ValueError(name)

    windows, pid_map = [], []
    meta = []
    if name == 'ds007537':
        for i, p in enumerate(pids):
            try:
                eeg, gaze, fs = loader(p)
            except Exception as e:
                print(f'  {p}: load failed: {e}')
                continue
            w = make_windows(eeg, gaze, fs)
            for x in w:
                x['pid'] = i
            windows.extend(w)
            pid_map.append(p)
            print(f'  {p}: {len(w)} windows')
    else:
        for i, p in enumerate(pids):
            ws = []
            for f in by_pid[p]:
                eeg, gaze, fs = load_consumer(f)
                w = make_windows(eeg, gaze, fs)
                for x in w:
                    x['pid'] = i
                ws.extend(w)
            windows.extend(ws)
            pid_map.append(p)
            print(f'  {p}: {len(ws)} windows')
    n_raw = len(windows)
    windows = [w for w in windows if w['qc']]
    print(f'{name}: {len(pid_map)} participants, {n_raw} windows, '
          f'{len(windows)} after QC ({100*len(windows)/max(n_raw,1):.1f}%)')
    return windows, pid_map


# --------------------------------------------------------------- training
class ExtDS(Dataset):
    def __init__(self, windows, stats=None, augment=False, noise=0.05):
        self.w = windows
        self.augment = augment
        self.noise = noise
        eegs = np.stack([x['eeg'] for x in windows])  # (N, C, T)
        if stats is None:
            self.em = eegs.mean(axis=(0, 2)).astype(np.float32)
            self.es = (eegs.std(axis=(0, 2)) + 1e-8).astype(np.float32)
            tg = np.stack([x['target'] for x in windows])
            self.tm = tg.mean(axis=0).astype(np.float32)
            self.ts = (tg.std(axis=0) + 1e-8).astype(np.float32)
        else:
            self.em, self.es, self.tm, self.ts = stats
        self.eegs = eegs

    def __len__(self):
        return len(self.w)

    def __getitem__(self, k):
        x = self.w[k]
        e = (self.eegs[k] - self.em[:, None]) / self.es[:, None]
        if self.augment:
            e = e + np.random.randn(*e.shape).astype(np.float32) * self.noise
        t = (x['target'] - self.tm) / self.ts
        return {'eeg': torch.from_numpy(e.astype(np.float32)),
                'gaze': torch.from_numpy(x['gaze']),
                'mask': torch.ones(N_GAZE, dtype=torch.bool),
                'target': torch.from_numpy(t.astype(np.float32)),
                'pid': x['pid']}

    def stats(self):
        return self.em, self.es, self.tm, self.ts


@torch.no_grad()
def predict(model, ds, device):
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    model.eval()
    P, T_, I = [], [], []
    for b in loader:
        out = model(b['eeg'].to(device), b['gaze'].to(device), b['mask'].to(device))
        P.append(out['pred'].cpu().numpy())
        T_.append(b['target'].numpy())
        I.append(b['pid'].numpy())
    return np.concatenate(T_), np.concatenate(P), np.concatenate(I)


MODELS = {
    'TempAlign': lambda c: TempAlignEEGNet(n_channels=c, hidden_dim=48, dropout=0.2),
    'GazeOnly': lambda c: TempAlignGazeOnly(hidden_dim=48, dropout=0.2),
    'EEGOnly': lambda c: TempAlignEEGOnly(n_channels=c, hidden_dim=48, dropout=0.2),
}


def run(name, seeds=(42,)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    windows, pid_map = build_dataset(name)
    n_ch = windows[0]['eeg'].shape[0]
    pids = np.array([w['pid'] for w in windows])

    results_path = os.path.join(OUT, f'results_external_{name}.json')
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}
    rng = np.random.RandomState(42)
    users = np.unique(pids)
    rng.shuffle(users)
    folds = []
    size = len(users) // 5
    for f in range(5):
        lo = f * size
        hi = lo + size if f < 4 else len(users)
        val = np.isin(pids, users[lo:hi])
        folds.append((~val, val))

    for mname, mfn in MODELS.items():
        results.setdefault(mname, {'runs': [], 'params': None})
        for fold, (tr_m, te_m) in enumerate(folds):
            for seed in seeds:
                if any(r['fold'] == fold and r['seed'] == seed for r in results[mname]['runs']):
                    continue
                t0 = time.time()
                tr_idx = np.where(tr_m)[0]
                te_idx = np.where(te_m)[0]
                inner_rng = np.random.RandomState(1000 + fold)
                tr_users = np.unique(pids[tr_idx])
                inner_users = inner_rng.permutation(tr_users)[:max(1, int(0.2 * len(tr_users)))]
                inner_sel = np.isin(pids[tr_idx], inner_users)
                inner_idx = tr_idx[inner_sel]
                tr_fit = tr_idx[~inner_sel]

                tr_ds = ExtDS([windows[i] for i in tr_fit], augment=True)
                st = tr_ds.stats()
                inner_ds = ExtDS([windows[i] for i in inner_idx], stats=st)
                te_ds = ExtDS([windows[i] for i in te_idx], stats=st)

                model = mfn(n_ch)
                if results[mname]['params'] is None:
                    results[mname]['params'] = count_params(model)
                model, ep, best, _ = train_one(model, tr_ds, inner_ds, CFG, device, seed)
                t, p, i = predict(model, te_ds, device)
                m = evaluate_predictions(t, p)
                m.update({'fold': fold, 'seed': seed, 'epochs': ep,
                          'time': time.time() - t0})
                results[mname]['runs'].append(m)
                print(f'  {name}/{mname} f{fold} s{seed}: r_col={m["r_col"]:.3f} '
                      f'R2={m["R2_col"]:.3f} ep={ep} ({m["time"]:.0f}s)', flush=True)
                np.savez_compressed(
                    os.path.join(OUT, f'preds_external_{name}_{mname}_f{fold}_s{seed}.npz'),
                    targets=t, preds=p, pids=i)
                with open(results_path, 'w') as f2:
                    json.dump(results, f2, indent=1)

    # statistical baselines
    results.setdefault('StatBaselines', {'runs': [], 'params': 0})
    if not results['StatBaselines']['runs']:
        for fold, (tr_m, te_m) in enumerate(folds):
            tr_idx, te_idx = np.where(tr_m)[0], np.where(te_m)[0]
            tg_tr = np.stack([windows[i]['target'] for i in tr_idx])
            mu, sd = tg_tr.mean(axis=0), tg_tr.std(axis=0) + 1e-8
            tg_te = (np.stack([windows[i]['target'] for i in te_idx]) - mu) / sd
            Xtr = np.stack([windows[i]['gaze'] for i in tr_idx]).reshape(len(tr_idx), -1)
            Xte = np.stack([windows[i]['gaze'] for i in te_idx]).reshape(len(te_idx), -1)
            ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]).fit(Xtr, tg_tr)
            pred = (ridge.predict(Xte) - mu) / sd
            m = evaluate_predictions(tg_te, pred)
            m.update({'fold': fold, 'seed': 0, 'name': 'RidgeGaze'})
            results['StatBaselines']['runs'].append(m)
            np.savez_compressed(
                os.path.join(OUT, f'preds_external_{name}_RidgeGaze_f{fold}.npz'),
                targets=tg_te, preds=pred, pids=pids[te_idx])
            m2 = evaluate_predictions(tg_te, np.zeros_like(tg_te))
            m2.update({'fold': fold, 'seed': 0, 'name': 'PersistenceZero'})
            results['StatBaselines']['runs'].append(m2)
            print(f'  {name}/StatBaselines f{fold}: ridge r_col={m["r_col"]:.3f}',
                  flush=True)
        with open(results_path, 'w') as f2:
            json.dump(results, f2, indent=1)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if which in ('ds007537', 'both'):
        run('ds007537')
    if which in ('consumer_muse', 'both'):
        run('consumer_muse')
