#!/usr/bin/env python3
# Combined GA posterior and uncertainty analysis script

import os, re, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
from scipy.stats import gaussian_kde, spearmanr, pearsonr
from scipy.special import logsumexp
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors

warnings.filterwarnings("ignore")

try:
    import corner
except ImportError:
    print("Missing dependency: pip install corner")
    exit(1)

plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 18,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 1.5,
})

# ---------------------- discovery ----------------------
def discover_result_folders(base_dir):
    out = []
    for name in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, name)
        if not os.path.isdir(p) or name.startswith('.'):
            continue
        csvs = sorted(glob.glob(os.path.join(p, "simulation_results*.csv")))
        if csvs:
            out.append((name, p, csvs))
    return out

def _suffix(fname):
    m = re.search(r'_gen_(\d+)\.csv$', fname)
    if m: return int(m.group(1))
    m = re.search(r'_(\d+)\.csv$', fname)
    return int(m.group(1)) if m else None

def choose_primary_csv(csv_list):
    recs = []
    for p in csv_list:
        try:
            size = os.path.getsize(p)
        except OSError:
            size = -1
        recs.append((p, size, _suffix(os.path.basename(p))))
    if any(s is not None for _,_,s in recs):
        return max(recs, key=lambda r: ((r[2] if r[2] is not None else -1), r[1]))[0]
    return max(recs, key=lambda r: r[1])[0]

# ---------------------- pcard axis ranges ----------------------
_PCARD_MAP = {
    'sigma_2_list': 'sigma_2',
    'tmax_1_list': 't_1',
    'tmax_2_list': 't_2',
    'infall_timescale_1_list': 'infall_1',
    'infall_timescale_2_list': 'infall_2',
    'sfe_array': 'sfe',
    'delta_sfe_array': 'delta_sfe',
    'imf_upper_limits': 'imf_upper',
    'mgal_values': 'mgal',
    'nb_array': 'nb',
}

def parse_pcard_ranges(pcard_path):
    if not os.path.isfile(pcard_path):
        return {}
    txt = open(pcard_path, 'r', encoding='utf-8').read()
    ranges = {}
    for key, col in _PCARD_MAP.items():
        m = re.search(rf'^\s*{re.escape(key)}\s*:\s*\[([^\]]+)\]', txt, flags=re.MULTILINE)
        if not m:
            continue
        try:
            nums = [float(x.strip()) for x in m.group(1).split(',')]
            if len(nums) == 2 and np.isfinite(nums[0]) and np.isfinite(nums[1]) and nums[1] > nums[0]:
                ranges[col] = (float(nums[0]), float(nums[1]))
        except Exception:
            pass
    return ranges

# ---------------------- loss → weights + diagnostics ----------------------
def weights_from_loss(loss, mode='exp', temperature=None, power=1.0):
    L = np.asarray(loss, float)
    good = np.isfinite(L)
    if good.sum() < 3:
        return np.zeros_like(L)
    w = np.zeros_like(L, float)
    if mode == 'inv':
        eps = np.nanmin(L[good]) * 1e-3 if np.isfinite(np.nanmin(L[good])) else 1e-12
        w[good] = 1.0 / np.power(L[good] + eps, power)
    else:
        if not (temperature and temperature > 0):
            temperature = 1.0
        L0 = np.nanmin(L[good])
        resid = L[good] - L0
        w[good] = np.exp(-resid / temperature)
    w[~np.isfinite(w)] = 0.0
    s = w.sum()
    if s > 0:
        w /= s
    return w

def effective_sample_size(w):
    w = np.asarray(w, float)
    w = w / (w.sum() + 1e-300)
    return (w.sum()**2) / (np.sum(w**2) + 1e-300)

def tune_temperature(loss, target_frac=0.30):
    L = np.asarray(loss, float)
    msk = np.isfinite(L)
    L = L[msk]
    if L.size < 3:
        return 1.0
    L0 = np.min(L)
    resid = L - L0
    mad = np.median(np.abs(resid - np.median(resid))) or np.std(resid) or 1.0
    N = L.size
    best_T, best_gap = mad, 1e18
    for s in np.logspace(-2, 2, 33):
        T = mad * s
        w = np.exp(-(resid) / T)
        w /= w.sum()
        ess = (w.sum()**2) / np.sum(w**2)
        gap = abs(ess - target_frac * N)
        if gap < best_gap:
            best_T, best_gap = T, gap
    return best_T

# ---------------------- helper functions for plotting ----------------------
def _kde_1d(vals, weights, xs, rng=np.random.default_rng(1337)):
    vals = np.asarray(vals, float)
    weights = np.asarray(weights, float)
    good = np.isfinite(vals) & np.isfinite(weights)
    vals = vals[good]; weights = weights[good]
    if len(vals) == 0:
        return np.zeros_like(xs)
    try:
        return gaussian_kde(vals, weights=weights)(xs)
    except Exception:
        N = min(max(1000, 5*len(vals)), 5000)
        p = weights / np.sum(weights) if np.sum(weights) > 0 else None
        idx = rng.choice(np.arange(len(vals)), size=N, replace=True, p=p)
        return gaussian_kde(vals[idx])(xs)

def _kde_2d(x, y, weights, Xg, Yg, rng=np.random.default_rng(1337)):
    x = np.asarray(x, float); y = np.asarray(y, float)
    weights = np.asarray(weights, float)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights)
    x = x[good]; y = y[good]; weights = weights[good]
    if len(x) == 0:
        return np.zeros_like(Xg)
    xy = np.vstack([x, y])
    try:
        kde = gaussian_kde(xy, weights=weights)
    except Exception:
        N = min(max(1000, 5*len(x)), 5000)
        p = weights / np.sum(weights) if np.sum(weights) > 0 else None
        idx = rng.choice(np.arange(len(x)), size=N, replace=True, p=p)
        kde = gaussian_kde(np.vstack([x[idx], y[idx]]))
    Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
    return Z

def _assoc(x, y, weights, assoc_metric='spearman'):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if assoc_metric == 'pearson':
        r, _ = pearsonr(x, y); return r, r'$\rho$'
    elif assoc_metric == 'axis_ratio':
        mx = np.average(x, weights=weights); sx = np.sqrt(np.average((x-mx)**2, weights=weights))
        my = np.average(y, weights=weights); sy = np.sqrt(np.average((y-my)**2, weights=weights))
        X = np.vstack([(x-mx)/max(sx,1e-12), (y-my)/max(sy,1e-12)])
        W = np.diag(weights/np.sum(weights))
        C = X @ W @ X.T
        evals, _ = np.linalg.eigh(C); evals = np.sort(evals)
        ar = np.sqrt(max(evals[-1],1e-30)/max(evals[0],1e-30))
        return ar, 'AR'
    else:
        r, _ = spearmanr(x, y); return r, r'$\rho_s$'

def _weighted_hpd_1d(x, w, mass=0.68):
    idx = np.argsort(x)
    x = x[idx]; w = w[idx]
    cs = np.cumsum(w)
    cs /= cs[-1]
    lo = np.interp((1-mass)/2, cs, x)
    hi = np.interp((1+mass)/2, cs, x)
    return lo, hi

def _ellipse_from_hpd(Xg, Yg, Z, p=0.68):
    Zn = Z / Z.max()
    thresh = sorted(Zn.ravel(), reverse=True)[int(p * len(Zn.ravel()))]
    contour = plt.contour(Xg, Yg, Zn, levels=[thresh])
    if len(contour.collections) == 0:
        return None
    path = contour.collections[0].get_paths()[0]
    verts = path.vertices
    cx, cy = np.mean(verts, axis=0)
    dx = verts[:,0] - cx; dy = verts[:,1] - cy
    cov = np.cov(dx, dy)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]; evecs = evecs[:, order]
    a = 2 * np.sqrt(evals[0])
    b = 2 * np.sqrt(evals[1])
    theta = np.arctan2(evecs[1,0], evecs[0,0])
    return {"center": (cx, cy), "a": a, "b": b, "theta": theta}

def _draw_ellipse(ax, center, a, b, theta, color='k', lw=1.5):
    ell = mpl.patches.Ellipse(center, a, b, angle=np.degrees(theta), fill=False, color=color, lw=lw)
    ax.add_patch(ell)

def _ellipse_proj_to_axes(a, b, theta):
    cos, sin = np.cos(theta), np.sin(theta)
    px = np.sqrt(a**2 * cos**2 + b**2 * sin**2)
    py = np.sqrt(a**2 * sin**2 + b**2 * cos**2)
    return px, py

# ---------------------- params & plotting ----------------------
DEFAULT_CONT = ['sigma_2','t_1','t_2','infall_1','infall_2','sfe','delta_sfe','imf_upper','mgal','nb']

def pick_params(df, preferred=None, min_unique=20):
    preferred = preferred or DEFAULT_CONT
    cand = []
    for c in df.columns:
        if c == 'fitness': 
            continue
        if np.issubdtype(df[c].dtype, np.number):
            v = df[c].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size and np.unique(v).size >= min_unique:
                cand.append(c)
    ordered = [c for c in preferred if c in cand]
    rest = [c for c in cand if c not in ordered]
    out = ordered + rest
    return out if out else DEFAULT_CONT

def _colors_for(n, ink_colors=None):
    if ink_colors is not None and len(ink_colors) >= n:
        return list(ink_colors)[:n]
    cyc = plt.rcParams.get('axes.prop_cycle').by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6'])
    out = []
    i = 0
    for _ in range(n):
        out.append(cyc[i % len(cyc)])
        i += 1
    return out

def corner_weighted(df, params, weights, out_png, group_by=None, colors=None, bins=40, point_size=4, title_note=None):
    if group_by is None:
        group_by = 'run' if 'run' in df.columns else None
    if group_by is None:
        df = df.copy()
        df['_group'] = 'all'
        group_by = '_group'

    labels = [c.replace('_',' ') for c in params]
    groups = list(df[group_by].astype(str).unique())
    palette = _colors_for(len(groups), ink_colors=colors)

    fig = None
    for gi, g in enumerate(groups):
        sub = df[df[group_by].astype(str) == g]
        X = sub[params].to_numpy(float)
        w = weights.loc[sub.index]
        fig = corner.corner(
            X,
            labels=labels,
            bins=bins,
            weights=w,
            color=palette[gi],
            fig=fig,
            show_titles=True,
            quantiles=[0.16, 0.5, 0.84],
            title_fmt=".3g",
            plot_datapoints=True,
            scatter_kwargs={'s': point_size, 'alpha': 0.5, 'rasterized': True},
            hist_kwargs={'density': True, 'alpha': 0.35, 'linewidth': 1.5}
        )

    if title_note:
        fig.axes[0].text(0.02, 0.98, title_note, transform=fig.axes[0].transAxes,
                         ha='left', va='top', fontsize=9)

    handles = [Line2D([0],[0], color=palette[i], lw=2) for i in range(len(groups))]
    fig.axes[0].legend(handles, groups, frameon=False, fontsize=9, loc='upper right')

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[saved]", out_png)
    return fig

def plot_overlaid_marginals(records, params, bins=60, out_png="analysis/marginals_overlay.png"):
    n = len(params); ncols = 3; nrows = (n + ncols - 1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3*nrows))
    axes = axes.ravel()
    for i, p in enumerate(params):
        ax = axes[i]
        for r in records:
            x = r['df'][p].to_numpy(float); w = r['w']
            hist, edges = np.histogram(x, bins=bins, weights=w, density=True)
            xl = 0.5*(edges[1:]+edges[:-1])
            ax.step(xl, hist, where='mid', alpha=0.9, label=r['name'])
        ax.set_title(p); ax.grid(True, alpha=0.25)
    for j in range(i+1, len(axes)): 
        axes[j].axis('off')
    axes[0].legend(loc='best', fontsize=8)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
    print("[saved]", out_png)

class UncertaintyAnalysis:
    def __init__(self, results_file, output_path='analysis/'):
        self.results_file = results_file
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        
        self.df = pd.read_csv(results_file)
        self.fitness_col = 'fitness' if 'fitness' in self.df.columns else 'wrmse'
        
        self.continuous_params = [
            'sigma_2', 't_1', 't_2', 'infall_1', 'infall_2', 
            'sfe', 'delta_sfe', 'imf_upper', 'mgal', 'nb'
        ]
        self.categorical_params = [
            'comp_idx', 'imf_idx', 'sn1a_idx', 'sy_idx', 'sn1ar_idx'
        ]
        
        self.continuous_params = [p for p in self.continuous_params if p in self.df.columns]
        self.categorical_params = [p for p in self.categorical_params if p in self.df.columns]
        
        self.df_sorted = self.df.sort_values(self.fitness_col, ascending=True)
        
        print(f"Loaded {len(self.df)} models from {results_file}")
        print(f"Best fitness: {self.df_sorted[self.fitness_col].iloc[0]:.6f}")

    def _select_top_and_weights(self, percentile=10, weight_power=1.0):
        df = self.df_sorted.copy()
        n_top = max(1, int(len(df) * percentile / 100))
        top = df.head(n_top)
        fit = np.asarray(top[self.fitness_col].values, dtype=float)
        eps = np.min(fit) * 0.001
        w = 1.0 / np.power(fit + eps, weight_power)
        w = w / np.sum(w)
        return top, w

    def _wmean(self, x, w):
        x = np.asarray(x, dtype=float); w = np.asarray(w, dtype=float)
        return float(np.sum(w*x) / np.sum(w))

    def _wvar(self, x, w):
        m = self._wmean(x, w)
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        return float(np.sum(w*(x-m)**2) / np.sum(w))

    def _wcov(self, x, y, w):
        mx = self._wmean(x, w); my = self._wmean(y, w)
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)
        return float(np.sum(w*(x-mx)*(y-my)) / np.sum(w))

    def plot_corner_with_marginals(self, params=None, percentile=10, weight_power=1.0, bins=40, save_path=None, assoc_metric='spearman', ink_color='#111111', alpha_gamma=0.6):
        top, w = self._select_top_and_weights(percentile=percentile, weight_power=weight_power)
        pcard_path = getattr(self, 'bulge_pcard_path', 'bulge_pcard.txt')
        param_ranges = parse_pcard_ranges(pcard_path)

        if params is None:
            candidates = list(param_ranges.keys())
            candidates = [c for c in candidates if c in top.columns]
            if hasattr(self, 'continuous_params') and isinstance(self.continuous_params, (list, set, tuple)):
                candidates = [c for c in candidates if c in self.continuous_params]
            params = candidates

        if len(params) == 0:
            return None

        k = len(params)
        label_map = {
            'sigma_2': r'$\sigma_2$',
            't_1': r't$_1$ (Gyr)', 't_2': r't$_2$ (Gyr)',
            'infall_1': r'$\tau_1$ (Gyr)', 'infall_2': r'$\tau_2$ (Gyr)',
            'sfe': 'SFE', 'delta_sfe': r'$\Delta$SFE',
            'imf_upper': r'M$_{up}$ (M$_\odot$)',
            'mgal': r'M$_{gal}$ (M$_\odot$)', 'nb': r'N$_{Ia}$ (M$_\odot^{-1}$)'
        }

        fig, axes = plt.subplots(k, k, figsize=(3.0*k, 3.0*k), constrained_layout=False)
        plt.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.06, wspace=0.0, hspace=0.0)
        rgb = np.array(mcolors.to_rgb(ink_color))

        for i, pi in enumerate(params):
            xi = np.asarray(top[pi].values, float)
            xlim_i = param_ranges.get(pi, (np.nanmin(xi), np.nanmax(xi)))

            for j, pj in enumerate(params):
                ax = axes[i, j]
                if i < j:
                    ax.axis('off'); continue

                if i == j:
                    v = xi[np.isfinite(xi)]
                    if v.size == 0:
                        ax.axis('off'); continue

                    lo, hi = xlim_i
                    xs = np.linspace(lo, hi, 512)
                    ax.hist(v, bins=bins, range=(lo, hi), density=True,
                            color='lightgray', edgecolor='black', alpha=0.7, weights=w)
                    dens = _kde_1d(v, w, xs)
                    ax.plot(xs, dens, lw=2, color='C0')
                    ax.set_xlim(lo, hi)
                    ax.set_yticks([])
                    if i < k-1: ax.set_xticklabels([])
                    ax.set_xlabel(label_map.get(pi, pi))

                else:
                    x = np.asarray(top[pj].values, float)
                    y = xi
                    xlim_j = param_ranges.get(pj, (np.nanmin(x), np.nanmax(x)))
                    lo_x, hi_x = xlim_j
                    lo_y, hi_y = xlim_i
                    nx = ny = 220
                    xg = np.linspace(lo_x, hi_x, nx)
                    yg = np.linspace(lo_y, hi_y, ny)
                    Xg, Yg = np.meshgrid(xg, yg)
                    Z = _kde_2d(x, y, w, Xg, Yg)
                    if np.all(~np.isfinite(Z)) or np.nanmax(Z) <= 0:
                        Z = np.zeros_like(Xg)

                    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
                    Zn = Z / (Z.max() + 1e-30)
                    alpha = np.power(Zn, alpha_gamma) * 0.95
                    rgba = np.empty((ny, nx, 4), dtype=float)
                    rgba[..., 0] = rgb[0]
                    rgba[..., 1] = rgb[1]
                    rgba[..., 2] = rgb[2]
                    rgba[..., 3] = alpha

                    ax.imshow(rgba, extent=(lo_x, hi_x, lo_y, hi_y),
                              origin='lower', interpolation='bilinear', aspect='auto')
                    ax.scatter(x, y, s=4, c='k', alpha=0.10, linewidths=0)

                    try:
                        mval, msym = _assoc(x, y, w, assoc_metric)
                        ax.text(0.03, 0.95, f'{msym}={mval:.2f}',
                                transform=ax.transAxes, ha='left', va='top',
                                fontsize=14, bbox=dict(facecolor='white', edgecolor='0.7', boxstyle='round,pad=0.2'))
                    except Exception:
                        pass

                    ax.set_xlim(lo_x, hi_x)
                    ax.set_ylim(lo_y, hi_y)
                    if j == 0:
                        ax.set_ylabel(label_map.get(pi, pi))
                    else:
                        ax.set_yticks([])
                    if i == k-1:
                        ax.set_xlabel(label_map.get(pj, pj))
                    else:
                        ax.set_xticks([])

        if save_path is None:
            save_path = os.path.join(self.output_path, 'posterior_corner_combo.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return save_path

def plot_corner_with_marginals_multi(analyzers, params, percentile=100, weight_power=1.0, bins=40, assoc_metric='spearman', alpha_gamma=0.9, ink_colors=None, legend_labels=None, save_path=None):
    tops = [a._select_top_and_weights(percentile=percentile, weight_power=weight_power)[0] for a in analyzers]
    weights = [a._select_top_and_weights(percentile=percentile, weight_power=weight_power)[1] for a in analyzers]
    
    param_ranges = {}
    for a in analyzers:
        pcard_path = getattr(a, 'bulge_pcard_path', 'bulge_pcard.txt')
        pr = parse_pcard_ranges(pcard_path)
        for k, v in pr.items():
            if k in param_ranges:
                param_ranges[k] = (min(param_ranges[k][0], v[0]), max(param_ranges[k][1], v[1]))
            else:
                param_ranges[k] = v

    k = len(params)
    fig, axes = plt.subplots(k, k, figsize=(3.0*k, 3.0*k), constrained_layout=False)
    plt.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.06, wspace=0.0, hspace=0.0)

    label_map = {
        'sigma_2': r'$\sigma_2$',
        't_1': r't$_1$ (Gyr)', 't_2': r't$_2$ (Gyr)',
        'infall_1': r'$\tau_1$ (Gyr)', 'infall_2': r'$\tau_2$ (Gyr)',
        'sfe': 'SFE', 'delta_sfe': r'$\Delta$SFE',
        'imf_upper': r'M$_{up}$ (M$_\odot$)',
        'mgal': r'M$_{gal}$ (M$_\odot$)', 'nb': r'N$_{Ia}$ (M$_\odot^{-1}$)'
    }

    for i, pi in enumerate(params):
        xlim_i = param_ranges.get(pi, (np.inf, -np.inf))
        for t in tops:
            if pi in t.columns:
                v = t[pi].to_numpy(float)
                xlim_i = (min(xlim_i[0], np.nanmin(v)), max(xlim_i[1], np.nanmax(v)))

        for j, pj in enumerate(params):
            ax = axes[i, j]
            if i < j:
                ax.axis('off'); continue

            if i == j:
                xs = np.linspace(xlim_i[0], xlim_i[1], 512)
                for t, w, col in zip(tops, weights, ink_colors):
                    if pi in t.columns:
                        v = t[pi].to_numpy(float)
                        dens = _kde_1d(v, w, xs)
                        ax.plot(xs, dens, lw=2, color=col, alpha=0.95)
                ax.set_xlim(xlim_i); ax.set_yticks([])
                if i < k-1: ax.set_xticklabels([])
                ax.set_xlabel(label_map.get(pi, pi))

            else:
                xlim_j = param_ranges.get(pj, (np.inf, -np.inf))
                for t in tops:
                    if pj in t.columns:
                        v = t[pj].to_numpy(float)
                        xlim_j = (min(xlim_j[0], np.nanmin(v)), max(xlim_j[1], np.nanmax(v)))
                lo_x, hi_x = xlim_j
                lo_y, hi_y = xlim_i
                nx = ny = 220
                xg = np.linspace(lo_x, hi_x, nx)
                yg = np.linspace(lo_y, hi_y, ny)
                Xg, Yg = np.meshgrid(xg, yg)

                for t, w, col in zip(tops, weights, ink_colors):
                    if pj in t.columns and pi in t.columns:
                        x = t[pj].to_numpy(float)
                        y = t[pi].to_numpy(float)
                        Z = _kde_2d(x, y, w, Xg, Yg)
                        if not np.isfinite(Z).any() or np.nanmax(Z) <= 0: continue
                        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
                        Zn = Z / (Z.max() + 1e-30)
                        alpha = np.power(Zn, alpha_gamma) * 0.90
                        rgb = np.array(mcolors.to_rgb(col))
                        rgba = np.empty((ny, nx, 4), float)
                        rgba[..., 0] = rgb[0]; rgba[..., 1] = rgb[1]; rgba[..., 2] = rgb[2]; rgba[..., 3] = alpha
                        ax.imshow(rgba, extent=(lo_x, hi_x, lo_y, hi_y),
                                  origin='lower', interpolation='bilinear', aspect='auto')

                ax.set_xlim(lo_x, hi_x); ax.set_ylim(lo_y, hi_y)
                if j == 0: ax.set_ylabel(label_map.get(pi, pi))
                else: ax.set_yticks([])
                if i == k-1: ax.set_xlabel(label_map.get(pj, pj))
                else: ax.set_xticks([])

    if legend_labels is not None:
        handles = [plt.Line2D([0],[0], color=c, lw=2) for c in ink_colors]
        fig.legend(handles, legend_labels, ncol=min(len(legend_labels), 2), frameon=False, loc='center right')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[saved]", save_path)

def plot_corner_points_contours(runs, params=None, run_names=None, include_loss_axis=True, alpha_range=(0.05, 0.9), alpha_gamma=0.6, point_size=4, bins=40, levels=(0.68, 0.95), smooth=0.9, colors=None, save_path=None):
    dfs, loss_cols = [], []
    for r in runs:
        df = r.df.copy()
        loss_col = r.fitness_col
        dfs.append(df); loss_cols.append(loss_col)

    if params is None:
        common = None
        for df in dfs:
            cols = set(df.select_dtypes(include=[np.number]).columns)
            common = cols if common is None else (common & cols)
        params = sorted(list(common or []))

    if include_loss_axis:
        params = list(params) + [loss_cols[0]]

    k = len(params)
    ranges = []
    for p in params:
        vmin, vmax = np.inf, -np.inf
        for df in dfs:
            if p in df.columns and df[p].notna().any():
                v = df[p].values
                vmin = min(vmin, np.nanmin(v)); vmax = max(vmax, np.nanmax(v))
        span = (vmax - vmin) or 1.0
        pad = 0.02 * span
        ranges.append((vmin - pad, vmax + pad))

    n = len(dfs)
    if colors is None:
        cyc = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6'])
        colors = [cyc[i % len(cyc)] for i in range(n)]
    run_names = run_names or [f'run {i+1}' for i in range(n)]

    a_min, a_max = alpha_range
    def loss_to_alpha(loss_array):
        L = np.asarray(loss_array, float)
        Lmin, Lmax = np.nanmin(L), np.nanmax(L)
        if not np.isfinite(Lmin) or not np.isfinite(Lmax) or Lmin == Lmax:
            q = np.zeros_like(L)
        else:
            q = (L - Lmin) / (Lmax - Lmin)
        s = (1.0 - q)**float(alpha_gamma)
        return a_min + s * (a_max - a_min)

    fig = None
    for idx, (df, lcol, col) in enumerate(zip(dfs, loss_cols, colors)):
        keep = np.ones(len(df), dtype=bool)
        for p in params:
            keep &= (p in df.columns) & np.isfinite(df[p].values)
        D = df.loc[keep, :]
        if D.empty: continue

        X = np.column_stack([D[p].values for p in params])
        fig = corner.corner(
            X,
            labels=params,
            bins=bins,
            range=ranges,
            color=col,
            smooth=smooth,
            levels=levels,
            plot_datapoints=False,
            plot_density=True,
            plot_contours=True,
            fill_contours=False,
            hist_kwargs=dict(histtype="step", linewidth=1.2),
            contour_kwargs=dict(linewidths=1.5),
            fig=fig if fig else None
        )

        alphas = loss_to_alpha(D[lcol].values)
        base_rgba = np.array(mcolors.to_rgba(col))
        rgba = np.repeat(base_rgba[None, :], len(D), axis=0)
        rgba[:, 3] = np.clip(alphas, 0.0, 1.0)

        axes = np.array(fig.axes).reshape((k, k))
        for i in range(1, k):
            for j in range(i):
                ax = axes[i, j]
                ax.scatter(
                    D[params[j]].values, D[params[i]].values,
                    s=point_size,
                    c=rgba,
                    marker='.',
                    linewidths=0,
                    rasterized=True
                )

    handles = [Line2D([0],[0], color=c, lw=2) for c in colors]
    fig.legend(handles, run_names, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[saved]", save_path)
    return fig

def export_best_per_folder_csv(chosen, out_csv):
    rows = []
    all_cols = set()

    for (name, path, csvs) in chosen:
        primary_csv = choose_primary_csv(csvs)
        df = pd.read_csv(primary_csv)
        fitness_col = 'fitness' if 'fitness' in df.columns else 'wrmse'
        df_sorted = df.sort_values(fitness_col, ascending=True)
        best = df_sorted.iloc[0]

        continuous_params = ['sigma_2', 't_1', 't_2', 'infall_1', 'infall_2', 'sfe', 'delta_sfe', 'imf_upper', 'mgal', 'nb']
        categorical_params = ['comp_idx', 'imf_idx', 'sn1a_idx', 'sy_idx', 'sn1ar_idx']
        keep_cols = list(dict.fromkeys(continuous_params + categorical_params + [fitness_col]))
        row = {"folder": name, "primary_csv": os.path.basename(primary_csv)}
        for c in keep_cols:
            if c in best.index:
                row[c] = best[c]
                all_cols.add(c)
        rows.append(row)

    cols = ["folder", "primary_csv"]
    fit_cols = [c for c in ("fitness", "wrmse") if c in all_cols]
    other = sorted([c for c in all_cols if c not in fit_cols])
    cols.extend(fit_cols + other)

    df = pd.DataFrame(rows, columns=cols)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[best-per-folder] wrote: {out_csv}")

def compute_and_plot_combined_covariant_uncertainties(analyzers, params, percentile=100, weight_power=1.0, p_hpd=0.68, grid_n=240, alpha_gamma=0.9, ink_colors=None, save_dir="analysis"):
    tops = [a._select_top_and_weights(percentile=percentile, weight_power=weight_power)[0] for a in analyzers]
    weights = [a._select_top_and_weights(percentile=percentile, weight_power=weight_power)[1] for a in analyzers]

    df_parts, w_parts = [], []
    for t, w in zip(tops, weights):
        sub = t[params].copy()
        df_parts.append(sub)
        w_parts.append(w)

    T = pd.concat(df_parts, axis=0, ignore_index=True)
    w_comb = np.concatenate(w_parts)
    w_comb /= (w_comb.sum() + 1e-300)

    param_ranges = {}
    for p in params:
        los, his = [], []
        for t in tops:
            if p in t.columns:
                v = t[p].to_numpy(float)
                v = v[np.isfinite(v)]
                if v.size:
                    los.append(float(v.min())); his.append(float(v.max()))
        if los:
            param_ranges[p] = (min(los), max(his))

    k = len(params)
    fig, axes = plt.subplots(k, k, figsize=(3.2*k, 3.2*k), constrained_layout=False)
    plt.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.06, wspace=0.0, hspace=0.0)

    label_map = {
        'sigma_2': r'$\sigma_2$',
        't_1': r't$_1$ (Gyr)', 't_2': r't$_2$ (Gyr)',
        'infall_1': r'$\tau_1$ (Gyr)', 'infall_2': r'$\tau_2$ (Gyr)',
        'sfe': 'SFE', 'delta_sfe': r'$\Delta$SFE',
        'imf_upper': r'M$_{up}$ (M$_\odot$)',
        'mgal': r'M$_{gal}$ (M$_\odot$)', 'nb': r'N$_{Ia}$ (M$_\odot^{-1}$)'
    }

    per_param = []
    for p in params:
        v_all = T[p].to_numpy(dtype=float)
        good = np.isfinite(v_all) & np.isfinite(w_comb)
        v_all = v_all[good]; wv = w_comb[good]
        if v_all.size:
            mean = np.average(v_all, weights=wv)
            median = np.interp(0.5, np.cumsum(np.sort(wv)), np.sort(v_all))
            lo, hi = _weighted_hpd_1d(v_all, wv, mass=p_hpd)
            hpd_width = hi - lo
            gauss_sigma = np.sqrt(np.average((v_all - mean)**2, weights=wv))
            per_param.append(dict(param=p, mean=mean, median=median,
                                  hpd_lo=lo, hpd_hi=hi, hpd_width=hpd_width,
                                  gauss_sigma=gauss_sigma))
    pd.DataFrame(per_param).to_csv(os.path.join(save_dir, "per_param_hpd68.csv"), index=False)

    pair_rows = []
    for i, pi in enumerate(params):
        lo_i, hi_i = param_ranges.get(pi, (np.nan, np.nan))
        for t in tops:
            if pi in t.columns:
                v = t[pi].to_numpy(float)
                lo_i = min(lo_i, np.nanmin(v)); hi_i = max(hi_i, np.nanmax(v))

        for j, pj in enumerate(params):
            ax = axes[i, j]
            if i < j:
                ax.axis('off'); continue

            if i == j:
                xs = np.linspace(lo_i, hi_i, 512)
                for t, w, col in zip(tops, weights, ink_colors):
                    if pi in t.columns:
                        v = t[pi].to_numpy(float)
                        dens = _kde_1d(v, w, xs)
                        ax.plot(xs, dens, lw=2, color=col, alpha=0.95)
                v_all = T[pi].to_numpy(dtype=float)
                good = np.isfinite(v_all) & np.isfinite(w_comb)
                v_all = v_all[good]; wv = w_comb[good]
                if v_all.size:
                    lo_h, hi_h = _weighted_hpd_1d(v_all, wv, mass=p_hpd)
                    ax.axvline(lo_h, color='k', lw=1.4, ls='--', alpha=0.9)
                    ax.axvline(hi_h, color='k', lw=1.4, ls='--', alpha=0.9)
                ax.set_xlim(lo_i, hi_i); ax.set_yticks([])
                if i < k-1: ax.set_xticklabels([])
                ax.set_xlabel(label_map.get(pi, pi))

            else:
                lo_x, hi_x = param_ranges.get(pj, (np.nan, np.nan))
                lo_y, hi_y = param_ranges.get(pi, (np.nan, np.nan))
                for t in tops:
                    if pj in t.columns:
                        v = t[pj].to_numpy(float)
                        lo_x = min(lo_x, np.nanmin(v)); hi_x = max(hi_x, np.nanmax(v))
                nx = ny = grid_n
                xg = np.linspace(lo_x, hi_x, nx)
                yg = np.linspace(lo_y, hi_y, ny)
                Xg, Yg = np.meshgrid(xg, yg)

                for t, w, col in zip(tops, weights, ink_colors):
                    if pj in t.columns and pi in t.columns:
                        x = t[pj].to_numpy(float)
                        y = t[pi].to_numpy(float)
                        Z = _kde_2d(x, y, w, Xg, Yg)
                        if not np.isfinite(Z).any() or np.nanmax(Z) <= 0: continue
                        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
                        Zn = Z / (Z.max() + 1e-30)
                        alpha = np.power(Zn, alpha_gamma) * 0.90
                        rgb = np.array(mcolors.to_rgb(col))
                        rgba = np.empty((ny, nx, 4), float)
                        rgba[..., 0] = rgb[0]; rgba[..., 1] = rgb[1]; rgba[..., 2] = rgb[2]; rgba[..., 3] = alpha
                        ax.imshow(rgba, extent=(lo_x, hi_x, lo_y, hi_y),
                                  origin='lower', interpolation='bilinear', aspect='auto')

                xC = T[pj].to_numpy(float)
                yC = T[pi].to_numpy(float)
                goodC = np.isfinite(xC) & np.isfinite(yC) & np.isfinite(w_comb)
                xC = xC[goodC]; yC = yC[goodC]; wC = w_comb[goodC]
                if xC.size:
                    try:
                        Zc = gaussian_kde(np.vstack([xC, yC]), weights=wC)(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
                    except Exception:
                        rng = np.random.default_rng(1337)
                        N = min(max(1000, 5*len(xC)), 5000)
                        p = wC/np.sum(wC)
                        idx = rng.choice(np.arange(len(xC)), size=N, replace=True, p=p)
                        Zc = gaussian_kde(np.vstack([xC[idx], yC[idx]]))(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
                    Zc = np.nan_to_num(Zc, nan=0.0, posinf=0.0, neginf=0.0)
                    el = _ellipse_from_hpd(Xg, Yg, Zc, p=p_hpd)
                    if el is None:
                        mx = float(np.average(xC, weights=wC))
                        my = float(np.average(yC, weights=wC))
                        C = np.cov(np.vstack([xC, yC]), aweights=wC, bias=True)
                        evals, evecs = np.linalg.eigh(C)
                        order = np.argsort(evals)[::-1]
                        evals = evals[order]; evecs = evecs[:, order]
                        a = 2.0*np.sqrt(max(evals[0],0.0))
                        b = 2.0*np.sqrt(max(evals[1],0.0))
                        theta = float(np.arctan2(evecs[1,0], evecs[0,0]))
                        cx, cy = mx, my
                    else:
                        a = el["a"]; b = el["b"]; theta = el["theta"]
                        cx, cy = el["center"]

                    _draw_ellipse(ax, (cx, cy), a, b, theta, color='k', lw=1.6)
                    px, py = _ellipse_proj_to_axes(a, b, theta)

                    pair_rows.append(dict(
                        p_row=pi, p_col=pj,
                        center_x=cx, center_y=cy,
                        a=a, b=b, theta_rad=theta, theta_deg=np.degrees(theta),
                        proj_on_col=px, proj_on_row=py
                    ))

                ax.set_xlim(lo_x, hi_x); ax.set_ylim(lo_y, hi_y)
                if j == 0: ax.set_ylabel(label_map.get(pi, pi))
                else: ax.set_yticks([])
                if i == k-1: ax.set_xlabel(label_map.get(pj, pj))
                else: ax.set_xticks([])

    corner_png = os.path.join(save_dir, "combined_corner_ellipses.png")
    fig.savefig(corner_png, dpi=300, bbox_inches='tight'); plt.close(fig)

    pd.DataFrame(pair_rows).to_csv(os.path.join(save_dir, "pairwise_hpd68_ellipses.csv"), index=False)

    Td = T[params].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    if Td.shape[0] >= 3:
        ww = w_comb[:len(Td)].copy()
        ww = ww / ww.max()
        mult = np.maximum((ww * 100).astype(int), 1)
        rep = np.repeat(np.arange(Td.shape[0]), mult[:Td.shape[0]])
        C = np.corrcoef(Td.iloc[rep].T)
        pd.DataFrame(C, index=params, columns=params).to_csv(os.path.join(save_dir, "combined_corr.csv"))

    print("[saved]", corner_png)
    print("[saved]", os.path.join(save_dir, "per_param_hpd68.csv"))
    print("[saved]", os.path.join(save_dir, "pairwise_hpd68_ellipses.csv"))
    return dict(
        corner_png=corner_png,
        per_param_csv=os.path.join(save_dir, "per_param_hpd68.csv"),
        pairwise_csv=os.path.join(save_dir, "pairwise_hpd68_ellipses.csv")
    )

if __name__ == "__main__":
    current_dir = os.getcwd()
    found = discover_result_folders(current_dir)

    print(f"\nFound {len(found)} candidate folders:")
    for idx, (name, path, csvs) in enumerate(found, start=1):
        primary = choose_primary_csv(csvs)
        size = os.path.getsize(primary) if os.path.exists(primary) else -1
        suf = _suffix(os.path.basename(primary))
        suf_txt = f"_gen_{suf}" if suf is not None else "(no gen suffix)"
        print(f"  [{idx:>2}] {name:30s}  primary={os.path.basename(primary)} {suf_txt}  size={size}")

    sel = input("\nEnter indices to overlay (e.g., 1 3 5 or 1,3,5). Press Enter for ALL: ").strip()

    if not sel:
        idxs = list(range(1, len(found) + 1))
        print(f"Selected ALL {len(found)} folders.")
    else:
        tokens = re.split(r'[,\s]+', sel)
        idxs = sorted(set(int(t) for t in tokens if t))

    chosen = []
    for i in idxs:
        if 1 <= i <= len(found):
            chosen.append(found[i-1])

    if not chosen:
        print("No valid indices chosen. Exiting.")
        exit(0)

    overlay_dir = os.path.join(current_dir, "analysis")
    os.makedirs(overlay_dir, exist_ok=True)

    export_best_per_folder_csv(chosen, os.path.join(overlay_dir, "best_per_folder.csv"))

    analyzers = []
    labels = []
    for (name, path, csvs) in chosen:
        primary_csv = choose_primary_csv(csvs)
        a = UncertaintyAnalysis(results_file=primary_csv, output_path=overlay_dir)
        pcard_here = os.path.join(path, "bulge_pcard.txt")
        if os.path.isfile(pcard_here):
            a.bulge_pcard_path = pcard_here
        if a.df_sorted[a.fitness_col].iloc[0] < 0.03:
            analyzers.append(a)
            labels.append(name)
        else:
            print(f"Skipping {name}: best fitness {a.df_sorted[a.fitness_col].iloc[0]:.6f} > 0.03")

    inkcolrs = ['#F0B800', '#004C40', '#0099A1', '#C20016', '#E8DCD8', '#97BAAB', '#1E6E6C', '#99724B', '#59454E']

    # Plot 1: corner_points_contours
    plot_corner_points_contours(
        runs=analyzers,
        params=['sigma_2', 't_2', 'infall_2', 't_1', 'infall_1', 'sfe', 'mgal', 'delta_sfe', 'nb', 'imf_upper', 'mae'],
        run_names=labels,
        include_loss_axis=True,
        colors=inkcolrs,
        levels=(0.68,),
        smooth=0.9,
        point_size=4,
        bins=40,
        save_path=os.path.join(overlay_dir, "corner_points_contours.png")
    )

    # Plot 2: bigger_posterior_corner_combo
    plot_corner_with_marginals_multi(
        analyzers,
        params=['sigma_2', 't_2', 'infall_2', 't_1', 'infall_1', 'sfe', 'mgal', 'delta_sfe'],
        percentile=100,
        weight_power=1.0,
        bins=40,
        assoc_metric='spearman',
        alpha_gamma=0.9,
        ink_colors=inkcolrs,
        legend_labels=labels,
        save_path=os.path.join(overlay_dir, "bigger_posterior_corner_combo.png")
    )

    # Plot 3: combined_corner_ellipses
    compute_and_plot_combined_covariant_uncertainties(
        analyzers,
        params=['sigma_2', 't_2', 'infall_2', 't_1', 'infall_1', 'sfe', 'mgal', 'delta_sfe'],
        percentile=100,
        weight_power=1.0,
        p_hpd=0.68,
        grid_n=240,
        alpha_gamma=0.9,
        ink_colors=inkcolrs,
        save_dir=overlay_dir
    )

    # Plot 4: in2_posterior_corner_combo
    plot_corner_with_marginals_multi(
        analyzers,
        params=['sigma_2', 't_2', 'infall_2', 'sfe'],
        percentile=100,
        weight_power=1.0,
        bins=40,
        assoc_metric='spearman',
        alpha_gamma=0.9,
        ink_colors=inkcolrs,
        legend_labels=None,
        save_path=os.path.join(overlay_dir, "in2_posterior_corner_combo.png")
    )

    # Plot 5: in1_posterior_corner_combo
    plot_corner_with_marginals_multi(
        analyzers,
        params=['sigma_2', 't_1', 'infall_1', 'sfe'],
        percentile=100,
        weight_power=1.0,
        bins=40,
        assoc_metric='spearman',
        alpha_gamma=0.9,
        ink_colors=inkcolrs,
        legend_labels=None,
        save_path=os.path.join(overlay_dir, "in1_posterior_corner_combo.png")
    )

    # Plot 6: chem_posterior_corner_combo
    plot_corner_with_marginals_multi(
        analyzers,
        params=['sigma_2', 'mgal', 'delta_sfe', 'sfe', 'nb', 'imf_upper'],
        percentile=100,
        weight_power=1.0,
        bins=40,
        assoc_metric='spearman',
        alpha_gamma=0.9,
        ink_colors=inkcolrs,
        legend_labels=None,
        save_path=os.path.join(overlay_dir, "chem_posterior_corner_combo.png")
    )

    # GA posterior part
    mode = "exp"
    temperature = None
    target_ess_frac = 0.30
    power = 1.0
    bins = 40
    params = ['sigma_2', 't_2', 'infall_2', 't_1', 'infall_1', 'sfe', 'mgal', 'delta_sfe', 'nb', 'imf_upper', 'mae']
    save_json = True

    records = []
    for name, path, csvs in chosen:
        primary = choose_primary_csv(csvs)
        df = pd.read_csv(primary)
        losscol = 'fitness' if 'fitness' in df.columns else 'wrmse'
        df_sorted = df.sort_values(by=losscol, ascending=True)
        L = df_sorted[losscol].to_numpy(float)
        T_used = tune_temperature(L, target_frac=target_ess_frac)
        w = weights_from_loss(L, mode=mode, temperature=T_used, power=power)
        w /= w.sum() + 1e-300
        pranges = parse_pcard_ranges(os.path.join(path, 'bulge_pcard.txt'))
        ess = int(effective_sample_size(w))
        records.append(dict(name=name, path=path, csv=primary, df=df_sorted, w=w, pr=pranges,
                            losscol=losscol, temperature=T_used, ess=ess))

    union_ranges = {}
    for p in params:
        los, his = [], []
        for r in records:
            if p in r['pr']:
                lo, hi = r['pr'][p]; los.append(lo); his.append(hi)
        if not los:
            for r in records:
                if p in r['df'].columns:
                    v = r['df'][p].to_numpy(float)
                    v = v[np.isfinite(v)]
                    if v.size:
                        los.append(float(v.min())); his.append(float(v.max()))
        if los:
            union_ranges[p] = (min(los), max(his))

    df_parts, w_parts = [], []
    for r in records:
        sub = r['df'][params].copy()
        sub['source'] = r['name']
        df_parts.append(sub)
        w_parts.append(r['w'])

    df_comb = pd.concat(df_parts, axis=0, ignore_index=True)
    w_comb = np.concatenate(w_parts)
    w_comb /= (w_comb.sum() + 1e-300)
    ess_comb = int(effective_sample_size(w_comb))

    note = f"weighted by {', '.join(sorted(set(r['losscol'] for r in records)))}; mode={mode}"
    out_comb = os.path.join(overlay_dir, "corner_mcmc_like_weighted.png")
    corner_weighted(
        df=df_comb,
        params=params,
        weights=pd.Series(w_comb, index=df_comb.index),
        out_png=out_comb,
        group_by='source',
        colors=inkcolrs,
        bins=bins,
        point_size=4,
        title_note=note
    )

    print(f"[saved] {out_comb}  | Combined ESS={ess_comb}/{len(w_comb)}")

    out_overlay = os.path.join(overlay_dir, "marginals_overlay.png")
    plot_overlaid_marginals(records, params, bins=min(80, max(30, bins)), out_png=out_overlay)

    if save_json:
        meta = {
            "mode": mode,
            "target_ess_frac": target_ess_frac,
            "temperature_arg": temperature,
            "params": params,
            "runs": [
                dict(name=r['name'], csv=os.path.relpath(r['csv']),
                     N=len(r['w']), ESS=int(r['ess']),
                     losscol=r['losscol'], temperature=r['temperature'])
                for r in records
            ],
            "combined": {"N": int(len(w_comb)), "ESS": int(ess_comb)}
        }
        out_meta = os.path.join(overlay_dir, "posterior_meta.json")
        with open(out_meta, "w") as f:
            json.dump(meta, f, indent=2)
        print("[saved]", out_meta)