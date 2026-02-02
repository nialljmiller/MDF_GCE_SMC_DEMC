# standalone_plots/plot_corner.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
newline = "\\\\" if mpl.rcParams.get("text.usetex", False) else "\n"
from matplotlib.patches import Ellipse
import csv

from scipy.ndimage import gaussian_filter
from .weights import attach_posterior_weights
from .best_selection import *


PARAM_COLUMNS = [
    "sigma_2", "t_1", "t_2", "infall_1", "infall_2",
    "sfe", "delta_sfe", "imf_upper", "mgal", "nb"
]

# Human-readable labels + units, using bulge_pcard definitions
PARAM_LABELS = {
    # sigma_2_list: mass ratio of second to first infall (dimensionless)
    # bulge_pcard: "mass ratio of the second to the first infall"
    "sigma_2": r"Second/first infall mass ratio $\sigma_2$",

    # tmax_1_list / tmax_2_list: times (in Gyr) after the Universe’s birth when each infall occurs
    # bulge_pcard: "Time (in Gyr) after the universe's birth when infall occurs"
    "t_1": r"First infall time $t_1$ (Gyr since Big Bang)",
    "t_2": r"Second infall time $t_2$ (Gyr since Big Bang)",

    # infall_timescale_1_list / 2_list: infall durations in Gyr
    "infall_1": r"First infall timescale $\tau_1$ (Gyr)",
    "infall_2": r"Second infall timescale $\tau_2$ (Gyr)",

    # sfe_array / delta_sfe_array: SFE in Gyr^{-1}
    "sfe":       r"Star formation efficiency SFE (Gyr$^{-1}$)",
    "delta_sfe": r"Change in SFE $\Delta$SFE (Gyr$^{-1}$)",

    # imf_upper_limits: IMF upper mass limit in solar masses
    "imf_upper": r"IMF upper mass $M_{\max}$ ($M_\odot$)",

    # mgal_values: total (bulge) mass in solar masses
    "mgal": r"Initial bulge gas mass $M_{\mathrm{gal}}$ ($M_\odot$)",

    # nb_array: number of SN Ia per solar mass formed
    "nb": r"SNe Ia per formed mass $N_{\rm Ia}/M_\odot$",
}



PARAM_LABELS = {
    "sigma_2":   r"$\sigma_2$",
    "t_1":       r"$t_1$ [Gyr]",
    "t_2":       r"$t_2$ [Gyr]",
    "infall_1":  r"$\tau_1$ [Gyr]",
    "infall_2":  r"$\tau_2$ [Gyr]",
    "sfe":       r"SFE [Gyr$^{-1}$]",
    "delta_sfe": r"$\Delta$SFE [Gyr$^{-1}$]",
    "imf_upper": r"$M_{\max}$ [$M_\odot$]",
    "mgal":      r"$M_{\mathrm{gal}}$ [$M_\odot$]",
    "nb":        r"$N_{\rm Ia}/M_\odot$",
}





# optional log axes (comment out if you want all linear)
LOG_AXES = {"mgal": True, "nb": True, "sfe": False, "t_1": True, "infall_1": False, "infall_2": True}



def param_label(name: str) -> str:
    """
    Map internal parameter name -> human-readable label with units.
    Falls back to a simple underscore→space replacement if unknown.
    """
    return PARAM_LABELS.get(name, name.replace("_", " "))




# ---- minimal helpers (no error handling) ----
def _weighted_mode_and_hdi(x, w, bins=80, mass=0.68):
    x = np.asarray(x, float); w = np.asarray(w, float)
    H, edges = np.histogram(x, bins=bins, weights=w, density=True)
    dx = np.diff(edges)
    p = H * dx
    centers = 0.5*(edges[:-1] + edges[1:])
    k_map = int(np.argmax(H))
    x_map = float(centers[k_map])
    order = np.argsort(H)[::-1]
    acc = 0.0
    kept = np.zeros_like(H, dtype=bool)
    for k in order:
        kept[k] = True
        acc += p[k]
        if acc >= mass: break
    lo = float(edges[np.argmax(kept)])
    hi = float(edges[np.where(kept)[0][-1] + 1])
    return x_map, lo, hi





import numpy as np
from scipy import stats
import csv
import os


def _wmean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    return np.sum(w * x) / np.sum(w)


def _wvar(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = _wmean(x, w)
    return np.sum(w * (x - m)**2) / np.sum(w)


def _wcov(x, y, w):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    mx = _wmean(x, w)
    my = _wmean(y, w)
    return np.sum(w * (x - mx) * (y - my)) / np.sum(w)


def _pairwise_corr_stats(T, w, cols, mi_bins=24):
    """
    T : (N, K) transformed coordinates INSIDE HPD region
    w : (N,) posterior weights
    cols : list of parameter names (len K)

    Returns list of dicts with:
      pi, pj, rho_w, rho_s, MI_w, axis_ratio
    """
    T = np.asarray(T, float)
    w = np.asarray(w, float)
    w = w / np.sum(w)

    N, K = T.shape
    rows = []

    for i in range(K):
        xi = T[:, i]
        vx = _wvar(xi, w)
        sx = np.sqrt(vx) if vx > 0.0 else 0.0
        mx = _wmean(xi, w)

        for j in range(i + 1, K):
            xj = T[:, j]
            vy = _wvar(xj, w)
            sy = np.sqrt(vy) if vy > 0.0 else 0.0
            my = _wmean(xj, w)

            # weighted Pearson
            if vx > 0.0 and vy > 0.0:
                cov_w = _wcov(xi, xj, w)
                rho_w = cov_w / np.sqrt(vx * vy)
            else:
                rho_w = np.nan

            # Spearman (unweighted)
            rho_s, _ = stats.spearmanr(xi, xj)

            # weighted mutual information
            mask = np.isfinite(xi) & np.isfinite(xj) & np.isfinite(w)
            xi_f = xi[mask]
            xj_f = xj[mask]
            w_f = w[mask]

            if xi_f.size < 5 or xj_f.size < 5:
                MI_w = np.nan
            else:
                x_edges = np.linspace(xi_f.min(), xi_f.max(), mi_bins + 1)
                y_edges = np.linspace(xj_f.min(), xj_f.max(), mi_bins + 1)

                Hxy, _, _ = np.histogram2d(xi_f, xj_f,
                                           bins=[x_edges, y_edges],
                                           weights=w_f)
                Hx, _ = np.histogram(xi_f, bins=x_edges, weights=w_f)
                Hy, _ = np.histogram(xj_f, bins=y_edges, weights=w_f)

                Pxy = Hxy / np.sum(Hxy)
                Px = Hx / np.sum(Hx)
                Py = Hy / np.sum(Hy)

                denom = Px[:, None] * Py[None, :]
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where((Pxy > 0) & (denom > 0),
                                     Pxy / denom, 1.0)
                    MI_w = float(np.nansum(Pxy * np.log(ratio)))
                if MI_w < 0.0:
                    MI_w = 0.0

            # ellipse axis ratio in z-scored space
            if sx > 0.0 and sy > 0.0:
                zx = (xi - mx) / sx
                zy = (xj - my) / sy
                C11 = _wcov(zx, zx, w)
                C22 = _wcov(zy, zy, w)
                C12 = _wcov(zx, zy, w)
                C = np.array([[C11, C12],
                              [C12, C22]], float)
                evals = np.linalg.eigvalsh(C)
                evals = np.clip(evals, 1e-12, None)
                axis_ratio = float(np.sqrt(evals.max() / evals.min()))
            else:
                axis_ratio = np.nan

            rows.append(dict(
                pi=cols[i],
                pj=cols[j],
                rho_w=rho_w,
                rho_s=rho_s,
                MI_w=MI_w,
                axis_ratio=axis_ratio,
            ))

    return rows




def _map_hdi_central_interval(x, w, mass=0.68, bins=128):
    x = np.asarray(x, float)
    w = np.asarray(w, float)

    mask = np.isfinite(x) & np.isfinite(w)
    x = x[mask]
    w = w[mask]

    w = w / np.sum(w)

    # MAP from histogram
    H, edges = np.histogram(x, bins=bins, weights=w, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    x_map = float(centers[np.argmax(H)])

    # HDI (shortest interval)
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]
    cumsum = np.cumsum(w_sorted)
    total = cumsum[-1]
    min_width = np.inf
    best_i, best_j = 0, len(x_sorted) - 1
    for i in range(len(x_sorted)):
        lo_cum = cumsum[i-1] if i > 0 else 0.0
        target = lo_cum + mass * total
        j = np.searchsorted(cumsum, target)
        if j >= len(x_sorted): break
        width = x_sorted[j] - x_sorted[i]
        if width < min_width:
            min_width = width
            best_i, best_j = i, j

    hdi_lo = float(x_sorted[best_i])
    hdi_hi = float(x_sorted[best_j])

    # central credible interval (equal tails)
    q_lo = 0.5 - 0.5 * mass
    q_hi = 0.5 + 0.5 * mass
    quantiles = np.interp([q_lo, 0.5, q_hi], cumsum, x_sorted)
    central_lo, median, central_hi = map(float, quantiles)

    return x_map,hdi_lo,hdi_hi,


# weighted HDI per dim in transformed coordinates
def _weighted_hdi_1d(x, w, mass=0.68, bins=200, pad=0.02):
    H, edges = np.histogram(x, bins=bins, weights=w, density=True)
    dx = np.diff(edges); p = H*dx
    order = np.argsort(H)[::-1]; acc = 0.0
    keep = np.zeros_like(H, bool)
    for k in order:
        keep[k] = True; acc += p[k]
        if acc >= mass: break
    lo = edges[np.argmax(keep)]
    hi = edges[np.where(keep)[0][-1] + 1]
    pad_abs = pad*(hi - lo)
    return float(lo - pad_abs), float(hi + pad_abs)




def make_corner_wlit(
    df: pd.DataFrame,
    output_dir,
    *,
    hdi_mass=0.6,
    post_crop=0.6,
    bins1d=30,
    bins2d=30,
    smooth=2.0,
    cmap="Blues",
    point_size=2.0,
    point_alpha=0.5,
    title_fs=16,
    dpi=300,
    excel_path="data/Previous_GCE_results.xlsx",
    loss_metric="loss"
):





    # axis transforms (identity or log10), kept dead simple
    def fwd(i, x):
        name = cols[i]
        if LOG_AXES.get(name, False):
            return np.log10(np.asarray(x, float))
        return np.asarray(x, float)


    def inv(i, x):
        name = cols[i]
        if LOG_AXES.get(name, False):
            return 10.0**np.asarray(x, float)
        return np.asarray(x, float)





    os.makedirs(os.path.join(output_dir, "corner"), exist_ok=True)

    cols = [c for c in PARAM_COLUMNS if c in df.columns]
    S = df[cols].copy()
    w = df["posterior_w"].to_numpy()
    X = S.to_numpy(float)
    K = len(cols)


    # transformed data for range estimation/plotting
    T = np.column_stack([fwd(i, X[:, i]) for i in range(K)])

    best_idx = stable_best_index(df, primary=loss_metric)
    best_vals = df.loc[best_idx, cols].to_numpy(float)
    B = np.array([fwd(i, best_vals[i]) for i in range(K)])  # transformed best


    ranges = [_weighted_hdi_1d(T[:, i], w, bins=max(200, 2*bins1d), mass = 1.0) for i in range(K)]


    #stats = [ _mode_hdi(T[:, i], w, bins=max(128, bins1d), mass=hdi_mass) for i in range(K) ]
    stats = [ _map_hdi_central_interval(T[:, i], w, mass=hdi_mass, bins=max(128, bins1d)) for i in range(K) ]


    fig, axes = plt.subplots(
        K, K, figsize=(2.6*K, 2.6*K),
        gridspec_kw={"wspace": 0.0, "hspace": 0.0}
    )

    # draw
    for i in range(K):  # row (y)
        yi = T[:, i]; ylo, yhi = ranges[i]
        for j in range(K):  # col (x)
            ax = axes[i, j]

            if i == j:
                # diagonal: filled weighted hist, no ticks/labels
                H, edges = np.histogram(yi, bins=bins1d, range=ranges[i], weights=w, density=True)
                centers = 0.5*(edges[:-1] + edges[1:])
                ax.bar(centers, H, width=np.diff(edges), align="center",
                       facecolor="#2b83ba", edgecolor="#1b4d66", linewidth=0.4)
                xmap, lo, hi = stats[i]
                ax.axvline(xmap, color="k", lw=1.4)
                ax.axvline(lo,   color="k", lw=0.9, ls="--")
                ax.axvline(hi,   color="k", lw=0.9, ls="--")
                #label = cols[i].replace("_", " ")
                label = param_label(cols[i])

                # back-transform to data units
                xm, xl, xh = inv(i, xmap), inv(i, lo), inv(i, hi)
                err_plus = xh - xm
                err_minus = xm - xl
                hdi_mass_str = int(100*hdi_mass)


                title = (
                    f"{label}{newline}"
                    f"MAP={xm:.4g}{newline}"
                    f"${{+{err_plus:.2g}}}    {{-{err_minus:.2g}}}$"
                )



                ax.set_title(title, fontsize=title_fs)

                
                ax.set_xlim(edges[0], edges[-1]); ax.set_yticks([]); ax.set_xticks([])
                for side in ("top",):#, "right", "left"):
                    ax.spines[side].set_visible(False)

            elif i > j:
                # lower triangle: rasterized cloud + smoothed cropped density + crosshairs at MAPs
                xi = T[:, j]; xlo, xhi = ranges[j]
                H, xed, yed = np.histogram2d(xi, yi, bins=bins2d,
                                             range=[(xlo, xhi), (ylo, yhi)], weights=w, density=True)
                H /= (np.sum(H) * (xed[1]-xed[0]) * (yed[1]-yed[0]) + 1e-12)
                H = gaussian_filter(H, sigma=smooth)
                peak = np.nanmax(H)
                H = np.where(H >= post_crop*peak, H, np.nan)
                Xg, Yg = np.meshgrid(xed, yed)

                # background cloud first (fast + small PDF)
                ax.scatter(xi, yi, s=point_size, c="k", alpha=point_alpha,
                           linewidths=0, rasterized=True, zorder=1)

                # density on top
                ax.pcolormesh(Xg, Yg, H.T, cmap=cmap, shading="auto", alpha=0.85, zorder=2)
                #ax.axvline(stats[i][0], color="#004C40", lw=1, zorder=3)
                #ax.axhline(stats[j][0], color="#004C40", lw=1, zorder=3)
                ax.scatter(stats[j][0], stats[i][0], s=44, marker='x', c="#C20016", linewidths=1.2, zorder=4)
                ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)


            else:
                ax.axis("off")

            # only bottom row / left col get ticks+labels (in data units)
            if i == K-1 and j <= i and i != j:
                # tick locations already in transformed space — just label with data units
                ticks = ax.get_xticks()
                ax.set_xticklabels([f"{inv(j, t):.3g}" for t in ticks])
                ax.set_xlabel(cols[j], fontsize=22)

                ax.set_xlabel(param_label(cols[j]), fontsize=22)

            else:
                ax.set_xticklabels([])

            if j == 0 and i > j:
                ticks = ax.get_yticks()
                ax.set_yticklabels([f"{inv(i, t):.3g}" for t in ticks])
                ax.set_ylabel(cols[i], fontsize=22)
                ax.set_ylabel(param_label(cols[i]), fontsize=22)
            else:
                ax.set_yticklabels([])

    fig.subplots_adjust(wspace=0.0, hspace=0.0)

    # ---- literature overlay (stars/lines) ----
    df_lit = pd.read_excel(excel_path, sheet_name=0, header=None)
    param_to_rowname = {
        #'sigma_2': 'MassRatio_final',
        't_1': 'Timescale1_final',
        't_2': 'Timescale2_final',
        'infall_1': 'Infall1_time_final',
        'infall_2': 'Infall2_time_final',
        'sfe': 'SFE_final',
        'delta_sfe': 'SFE_infall2_final',
        'imf_upper': 'IMF_mmax_final',
        'mgal': 'BulgeMass_final',
        'nb': 'SNIa_perMsun_final'
    }
    study_cols = [str(s).strip() for s in df_lit.iloc[0, 2:10].tolist() if pd.notna(s)]

    colors = [    '#BE2500',
    '#DCE1C5',
    '#1A006C',
    '#A89F62',
    '#AE0000',
    '#0723BA',
    '#00816E',
    '#ECBB00',
    '#4F4618',
    '#EE0000',
    ]



    row_idx = {p: df_lit[df_lit.iloc[:, 0].astype(str).str.contains(k, na=False)].index[0]
               for p, k in param_to_rowname.items() if any(df_lit.iloc[:, 0].astype(str).str.contains(k, na=False))}

    print(df_lit)
    #df_lit = pd.read_excel(excel_path, sheet_name=0, header=None)
    df_lit = df_lit.set_index(0)   # <--- CRITICAL
    num = pd.to_numeric(df_lit.loc['SFE_infall2_final'], errors='coerce')
    den = pd.to_numeric(df_lit.loc['SFE_final'], errors='coerce')
    df_lit.loc['SFE_infall2_final'] = num / den


    # pulled values
    lit = {study: {} for study in study_cols}
    for ci, study in zip(range(2, 10), study_cols):
        for p, r in row_idx.items():
            v = df_lit.iloc[r, ci]
            if isinstance(v, (int, float, np.floating)): lit[study][p] = float(v)

    # diagonals: stars at mid-height
    for i, p in enumerate(cols):
        if p not in row_idx: continue
        ax = axes[i, i]
        for k, study in enumerate(study_cols):
            if p in lit[study]:
                ymid = (1+k)*0.1*ax.get_ylim()[1]
                xval = fwd(i, lit[study][p])
                lo, hi = ranges[i]
                if lo <= xval <= hi:
                    ax.scatter(xval, ymid, s=600, marker='*', c=[colors[k]],
                               edgecolors='k', linewidths=0.6, zorder=4)

    # lower triangle: stars or guide-lines
    from matplotlib.lines import Line2D
    handles, labels = [], []
    for i in range(K):
        py = cols[i]
        for j in range(i):
            px = cols[j]
            ax = axes[i, j]
            for k, study in enumerate(study_cols):
                vx = lit[study].get(px, None)
                vy = lit[study].get(py, None)
                done = False
                if vx is not None and vy is not None:
                    xv = fwd(j, vx); yv = fwd(i, vy)
                    ax.scatter(xv, yv, s=600, marker='*', c=[colors[k]],
                               edgecolors='k', linewidths=0.8, zorder=4)
                    done = True
                elif vx is not None:
                    xv = fwd(j, vx)
                    ax.axvline(xv, color=colors[k], lw=6.2, alpha=0.8, zorder=3)
                    done = True
                elif vy is not None:
                    yv = fwd(i, vy)
                    ax.axhline(yv, color=colors[k], lw=6.2, alpha=0.8, zorder=3)
                    done = True
                if done and (study not in labels):
                    handles.append(Line2D([], [], color=colors[k], marker='*',
                                          markersize=30, linestyle='None',
                                          markeredgecolor='k', markeredgewidth=0.8))
                    labels.append(study)

    if handles:
        la = fig.add_axes([0.72, 0.72, 0.17, 0.17])
        la.set_facecolor('white'); la.patch.set_alpha(0.95); la.axis('off')
        la.legend(handles=handles, labels=labels, loc='best', bbox_to_anchor=(0.4, 0.4, 0.4, 0.4), fontsize=40, frameon=True, edgecolor='k', markerscale=1)

    out_path = os.path.join(output_dir, "posterior_corner_wlit.pdf")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)









def make_corner(
    df: pd.DataFrame,
    output_dir,
    *,
    hdi_mass=0.68,
    post_crop=0.6,
    bins1d=30,
    bins2d=30,
    smooth=2.0,
    cmap="Blues",
    point_size=1.0,
    point_alpha=0.5,
    title_fs=16,
    dpi=300,
    excel_path="data/Previous_GCE_results.xlsx",
    loss_metric="loss"
):

    # axis transforms (identity or log10), kept dead simple
    def fwd(i, x):
        name = cols[i]
        if LOG_AXES.get(name, False):
            return np.log10(np.asarray(x, float))
        return np.asarray(x, float)


    def inv(i, x):
        name = cols[i]
        if LOG_AXES.get(name, False):
            return 10.0**np.asarray(x, float)
        return np.asarray(x, float)


    os.makedirs(os.path.join(output_dir, "corner"), exist_ok=True)

    cols = [c for c in PARAM_COLUMNS if c in df.columns]
    S = df[cols].copy()
    w = df["posterior_w"].to_numpy()
    X = S.to_numpy(float)
    K = len(cols)


    # transformed data for range estimation/plotting
    T = np.column_stack([fwd(i, X[:, i]) for i in range(K)])

    best_idx = stable_best_index(df, primary=loss_metric)
    best_vals = df.loc[best_idx, cols].to_numpy(float)
    B = np.array([fwd(i, best_vals[i]) for i in range(K)])  # transformed best


    ranges = [_weighted_hdi_1d(T[:, i], w, bins=max(200, 2*bins1d), mass = 1.0) for i in range(K)]


    #stats = [ _mode_hdi(T[:, i], w, bins=max(128, bins1d), mass=hdi_mass) for i in range(K) ]
    stats = [ _map_hdi_central_interval(T[:, i], w, mass=hdi_mass, bins=max(128, bins1d)) for i in range(K) ]


    # --------------------------------------------------
    # write MAP + HDI summary to CSV (in data units)
    # --------------------------------------------------
    out_csv = os.path.join(output_dir, "corner_param_stats.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["param", "MAP", "HDI_lo", "HDI_hi", "err_minus", "err_plus"])
        for i, name in enumerate(cols):
            map_t, lo_t, hi_t = stats[i]
            # back to physical units
            map_val = float(inv(i, map_t))
            hdi_lo = float(inv(i, lo_t))
            hdi_hi = float(inv(i, hi_t))
            err_minus = map_val - hdi_lo
            err_plus  = hdi_hi - map_val
            writer.writerow([name, map_val, hdi_lo, hdi_hi, err_minus, err_plus])
    # --------------------------------------------------





    # Build 68% HPD hyper-rectangle mask from the 1D HDIs
    hpd_mask = np.ones(T.shape[0], dtype=bool)
    for i in range(K):
        map_t, lo_t, hi_t = stats[i]
        hpd_mask &= (T[:, i] >= lo_t) & (T[:, i] <= hi_t)

    T_hpd = T[hpd_mask]
    w_hpd = w[hpd_mask]


    # --------------------------------------------------
    # pairwise degeneracies inside HPD region
    # --------------------------------------------------
    corr_rows = _pairwise_corr_stats(T_hpd, w_hpd, cols, mi_bins=24)

    out_corr = os.path.join(output_dir, "corner_pairwise_degeneracies.csv")
    with open(out_corr, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pi", "pj", "rho_w", "rho_s", "MI_w", "axis_ratio"])
        for r in corr_rows:
            writer.writerow([
                r["pi"],
                r["pj"],
                r["rho_w"],
                r["rho_s"],
                r["MI_w"],
                r["axis_ratio"],
            ])






    fig, axes = plt.subplots(
        K, K, figsize=(2.6*K, 2.6*K),
        gridspec_kw={"wspace": 0.0, "hspace": 0.0}
    )

    # draw
    for i in range(K):  # row (y)
        yi = T[:, i]; ylo, yhi = ranges[i]
        for j in range(K):  # col (x)
            ax = axes[i, j]

            if i == j:
                # diagonal: filled weighted hist, no ticks/labels
                H, edges = np.histogram(yi, bins=bins1d, range=ranges[i], weights=w, density=True)
                centers = 0.5*(edges[:-1] + edges[1:])
                ax.bar(centers, H, width=np.diff(edges), align="center",
                       facecolor="#2b83ba", edgecolor="#1b4d66", linewidth=0.4)
                xmap, lo, hi = stats[i]
                ax.axvline(xmap, color="k", lw=1.4)
                ax.axvline(lo,   color="k", lw=0.9, ls="--")
                ax.axvline(hi,   color="k", lw=0.9, ls="--")
                #label = cols[i].replace("_", " ")
                label = param_label(cols[i])

                # back-transform to data units
                xm, xl, xh = inv(i, xmap), inv(i, lo), inv(i, hi)
                err_plus = xh - xm
                err_minus = xm - xl
                hdi_mass_str = int(100*hdi_mass)

                title = (
                    f"{label}{newline}"
                    f"MAP={xm:.4g}{newline}"
                    f"${{+{err_plus:.2g}}}    {{-{err_minus:.2g}}}$"
                )



                ax.set_title(title, fontsize=title_fs)

                
                ax.set_xlim(edges[0], edges[-1]); ax.set_yticks([]); ax.set_xticks([])
                for side in ("top",):#, "right", "left"):
                    ax.spines[side].set_visible(False)

            elif i > j:
                # lower triangle: rasterized cloud + smoothed cropped density + crosshairs at MAPs
                xi = T[:, j]; xlo, xhi = ranges[j]
                H, xed, yed = np.histogram2d(xi, yi, bins=bins2d,
                                             range=[(xlo, xhi), (ylo, yhi)], weights=w, density=True)
                H /= (np.sum(H) * (xed[1]-xed[0]) * (yed[1]-yed[0]) + 1e-12)
                H = gaussian_filter(H, sigma=smooth)
                peak = np.nanmax(H)
                H = np.where(H >= post_crop*peak, H, np.nan)
                Xg, Yg = np.meshgrid(xed, yed)

                # background cloud first (fast + small PDF)
                ax.scatter(xi, yi, s=point_size, c="k", alpha=point_alpha,
                           linewidths=0, rasterized=True, zorder=1)




                from matplotlib.patches import Rectangle

                # ----- get stats in transformed space -----
                x_map, x_lo, x_hi = stats[j]   # column param
                y_map, y_lo, y_hi = stats[i]   # row param

                # rectangle that runs from lo→hi in each dimension
                width  = x_hi - x_lo
                height = y_hi - y_lo

                rect = Rectangle(
                    (x_lo, y_lo),          # bottom-left corner
                    width,
                    height,
                    edgecolor="#C20013",
                    facecolor="none",
                    lw=2.2,
                    ls="-",
                    zorder=3,
                    alpha=0.4
                )
                ax.add_patch(rect)

                rect = Rectangle(
                    (x_lo, y_lo),          # bottom-left corner
                    width,
                    height,
                    edgecolor="#C20016",
                    facecolor="none",
                    lw=1.2,
                    ls="-",
                    zorder=4,
                    alpha=0.7
                )
                ax.add_patch(rect)





                # density on top
                ax.pcolormesh(Xg, Yg, H.T, cmap=cmap, shading="auto", alpha=0.85, zorder=2)
                #ax.axvline(stats[j][0], color="#004C40", lw=1, zorder=3, alpha = 0.7)
                #ax.axhline(stats[i][0], color="#004C40", lw=1, zorder=3, alpha = 0.7)
                ax.scatter(stats[j][0], stats[i][0], s=44, marker='x', c="k", linewidths=2, zorder=4)
                ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)


            else:
                ax.axis("off")

            # only bottom row / left col get ticks+labels (in data units)
            if i == K-1 and j <= i and i != j:
                # tick locations already in transformed space — just label with data units
                ticks = ax.get_xticks()
                ax.set_xticklabels([f"{inv(j, t):.3g}" for t in ticks])
                ax.set_xlabel(cols[j], fontsize=22)

                ax.set_xlabel(param_label(cols[j]), fontsize=22)

            else:
                ax.set_xticklabels([])

            if j == 0 and i > j:
                ticks = ax.get_yticks()
                ax.set_yticklabels([f"{inv(i, t):.3g}" for t in ticks])
                ax.set_ylabel(cols[i], fontsize=22)
                ax.set_ylabel(param_label(cols[i]), fontsize=22)
            else:
                ax.set_yticklabels([])

    fig.subplots_adjust(wspace=0.0, hspace=0.0)

    out_path = os.path.join(output_dir, "posterior_corner.pdf")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    

def make_corner_mcmc(
    df: pd.DataFrame,
    output_dir,
    *,
    hdi_mass=0.68,
    post_crop=0.6,   # now unused, but kept for API compatibility
    bins1d=30,
    bins2d=30,
    smooth=2.0,
    cmap="Blues",
    point_size=1.0,  # now irrelevant
    point_alpha=0.5, # now irrelevant
    title_fs=16,
    dpi=300,
    excel_path="data/Previous_GCE_results.xlsx",
    loss_metric="loss"
):

    def fwd(i, x):
        name = cols[i]
        if LOG_AXES.get(name, False):
            return np.log10(np.asarray(x, float))
        return np.asarray(x, float)

    def inv(i, x):
        name = cols[i]
        if LOG_AXES.get(name, False):
            return 10.0**np.asarray(x, float)
        return np.asarray(x, float)

    df["t_2"] = df["t_2"] - 2.6

    os.makedirs(os.path.join(output_dir, "corner"), exist_ok=True)

    cols = [c for c in PARAM_COLUMNS if c in df.columns]
    S = df[cols].copy()
    w = df["posterior_w"].to_numpy()
    X = S.to_numpy(float)
    K = len(cols)

    T = np.column_stack([fwd(i, X[:, i]) for i in range(K)])

    best_idx = stable_best_index(df, primary=loss_metric)
    best_vals = df.loc[best_idx, cols].to_numpy(float)
    B = np.array([fwd(i, best_vals[i]) for i in range(K)])

    ranges = [
        _weighted_hdi_1d(T[:, i], w, bins=max(200, 2 * bins1d), mass=1.0)
        for i in range(K)
    ]

    stats = [
        _map_hdi_central_interval(T[:, i], w, mass=hdi_mass, bins=max(128, bins1d))
        for i in range(K)
    ]

    out_csv = os.path.join(output_dir, "corner_param_stats.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["param", "MAP", "HDI_lo", "HDI_hi", "err_minus", "err_plus"])
        for i, name in enumerate(cols):
            map_t, lo_t, hi_t = stats[i]
            map_val = float(inv(i, map_t))
            hdi_lo = float(inv(i, lo_t))
            hdi_hi = float(inv(i, hi_t))
            err_minus = map_val - hdi_lo
            err_plus = hdi_hi - map_val
            writer.writerow([name, map_val, hdi_lo, hdi_hi, err_minus, err_plus])

    # pairwise degeneracies over full posterior
    corr_rows = _pairwise_corr_stats(T, w, cols, mi_bins=24)

    out_corr = os.path.join(output_dir, "corner_pairwise_degeneracies.csv")
    with open(out_corr, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pi", "pj", "rho_w", "rho_s", "MI_w", "axis_ratio"])
        for r in corr_rows:
            writer.writerow(
                [r["pi"], r["pj"], r["rho_w"], r["rho_s"], r["MI_w"], r["axis_ratio"]]
            )

    fig, axes = plt.subplots(
        K, K, figsize=(2.6 * K, 2.6 * K),
        gridspec_kw={"wspace": 0.0, "hspace": 0.0}
    )

    for i in range(K):  # row (y)
        yi = T[:, i]
        ylo, yhi = ranges[i]
        for j in range(K):  # col (x)
            ax = axes[i, j]

            if i == j:
                H, edges = np.histogram(
                    yi, bins=bins1d, range=ranges[i], weights=w, density=True
                )
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.bar(
                    centers,
                    H,
                    width=np.diff(edges),
                    align="center",
                    facecolor="#2b83ba",
                    edgecolor="#1b4d66",
                    linewidth=0.4,
                )
                xmap, lo, hi = stats[i]
                ax.axvline(xmap, color="k", lw=1.4)
                ax.axvline(lo, color="k", lw=0.9, ls="--")
                ax.axvline(hi, color="k", lw=0.9, ls="--")
                label = param_label(cols[i])

                xm, xl, xh = inv(i, xmap), inv(i, lo), inv(i, hi)
                err_plus = xh - xm
                err_minus = xm - xl

                title = (
                    f"{label}{newline}"
                    f"MAP={xm:.4g}{newline}"
                    f"${{+{err_plus:.2g}}}    {{-{err_minus:.2g}}}$"
                )

                ax.set_title(title, fontsize=title_fs)

                ax.set_xlim(edges[0], edges[-1])
                ax.set_yticks([])
                ax.set_xticks([])
                for side in ("top",):
                    ax.spines[side].set_visible(False)

            elif i > j:
                # lower triangle: ONLY smoothed 2D posterior, covering full square
                xi = T[:, j]
                xlo, xhi = ranges[j]
                H, xed, yed = np.histogram2d(
                    xi,
                    yi,
                    bins=bins2d,
                    range=[(xlo, xhi), (ylo, yhi)],
                    weights=w,
                    density=True,
                )
                H /= (np.sum(H) * (xed[1] - xed[0]) * (yed[1] - yed[0]) + 1e-12)
                H = gaussian_filter(H, sigma=smooth)

                Xg, Yg = np.meshgrid(xed, yed)

                from matplotlib.patches import Rectangle

                x_map, x_lo, x_hi = stats[j]
                y_map, y_lo, y_hi = stats[i]

                width = x_hi - x_lo
                height = y_hi - y_lo

                rect = Rectangle(
                    (x_lo, y_lo),
                    width,
                    height,
                    edgecolor="#C20013",
                    facecolor="none",
                    lw=2.2,
                    ls="-",
                    zorder=3,
                    alpha=0.4,
                )
                ax.add_patch(rect)

                rect = Rectangle(
                    (x_lo, y_lo),
                    width,
                    height,
                    edgecolor="#C20016",
                    facecolor="none",
                    lw=1.2,
                    ls="-",
                    zorder=4,
                    alpha=0.7,
                )
                ax.add_patch(rect)

                # full 2D posterior
                ax.pcolormesh(
                    Xg, Yg, H.T, cmap=cmap, shading="auto", alpha=1, zorder=2
                )
                ax.set_xlim(xlo, xhi)
                ax.set_ylim(ylo, yhi)

            else:
                ax.axis("off")

            if i == K - 1 and j <= i and i != j:
                ticks = ax.get_xticks()
                ax.set_xticklabels([f"{inv(j, t):.3g}" for t in ticks])
                ax.set_xlabel(param_label(cols[j]), fontsize=22)
            else:
                ax.set_xticklabels([])

            if j == 0 and i > j:
                ticks = ax.get_yticks()
                ax.set_yticklabels([f"{inv(i, t):.3g}" for t in ticks])
                ax.set_ylabel(param_label(cols[i]), fontsize=22)
            else:
                ax.set_yticklabels([])

    fig.subplots_adjust(wspace=0.0, hspace=0.0)

    out_path = os.path.join(output_dir, "posterior_corner_mcmc.pdf")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)









def summarize_categorical_posterior(
    df: pd.DataFrame,
    output_dir: str,
    cat_cols=None,
    hdi_mass: float = 0.68,
    bins1d: int = 30,
    loss_metric: str = "loss"
):
    """
    Summarise categorical parameters inside the same HPD region
    used by the corner plot.

    - Uses df['posterior_w'] as weights.
    - Defines HPD region as the hyper-rectangle given by the
      1D weighted HDIs in transformed space (same as corner axes).
    - For each categorical column:
        * histogram of raw vs weighted fractions
        * CSV row with raw%, weighted%, mean/median loss
    """

    # -------------------------------
    # 1. Basic inputs and weights
    # -------------------------------
    if "posterior_w" not in df.columns:
        raise ValueError("DataFrame must contain 'posterior_w' column.")

    if loss_metric not in df.columns:
        raise ValueError(f"DataFrame must contain loss metric column '{loss_metric}'.")

    w_all = df["posterior_w"].to_numpy(float)
    L_all = df[loss_metric].to_numpy(float)

    cols = [c for c in PARAM_COLUMNS if c in df.columns]
    if not cols:
        raise ValueError("No continuous PARAM_COLUMNS found in DataFrame.")

    S = df[cols].copy()
    X = S.to_numpy(float)
    K = len(cols)

    # default categorical columns if not given
    if cat_cols is None:
        candidate_cats = ["comp_idx", "imf_idx", "sn1a_idx", "sy_idx", "sn1ar_idx"]
        cat_cols = [c for c in candidate_cats if c in df.columns]

    if not cat_cols:
        raise ValueError("No categorical columns provided or found in DataFrame.")

    # -------------------------------
    # 2. Transform + HDI ranges
    # -------------------------------
    LOG_AXES = {"mgal": True, "nb": True}

    def fwd(i, x):
        name = cols[i]
        if LOG_AXES.get(name, False):
            return np.log10(np.asarray(x, float))
        return np.asarray(x, float)

    # build transformed matrix
    T = np.column_stack([fwd(i, X[:, i]) for i in range(K)])

    # HDI ranges in transformed coordinates
    ranges = [
        _weighted_hdi_1d(T[:, i], w_all, mass=hdi_mass, bins=max(200, 2 * bins1d))
        for i in range(K)
    ]

    # -------------------------------
    # 3. Build HPD mask (same region
    #    that defines corner axes)
    # -------------------------------
    mask_hpd = np.ones(len(df), dtype=bool)
    for i in range(K):
        lo, hi = ranges[i]
        ti = T[:, i]
        mask_hpd &= (ti >= lo) & (ti <= hi)

    if not mask_hpd.any():
        raise ValueError("No samples lie inside the HPD hyper-rectangle.")

    df_hpd = df.loc[mask_hpd].copy()
    w_hpd = w_all[mask_hpd]
    L_hpd = L_all[mask_hpd]

    # normalise weights in HPD for fractional stats
    w_hpd = w_hpd / np.sum(w_hpd)

    # -------------------------------
    # 4. Per–categorical stats table
    # -------------------------------
    rows = []

    for col in cat_cols:
        vals = df_hpd[col].to_numpy()
        if vals.size == 0:
            continue

        unique_vals = np.unique(vals)
        total_count = float(vals.size)

        for u in unique_vals:
            m_cat = (vals == u)
            count = int(m_cat.sum())
            raw_frac = count / total_count

            w_sum = float(w_hpd[m_cat].sum())
            w_frac = w_sum  # already normalised

            L_cat = L_hpd[m_cat]
            loss_mean = float(L_cat.mean())
            loss_median = float(np.median(L_cat))

            rows.append(
                {
                    "param": col,
                    "category": u,
                    "raw_count": count,
                    "raw_frac": raw_frac,
                    "weight_sum": w_sum,
                    "weight_frac": w_frac,
                    "loss_mean": loss_mean,
                    "loss_median": loss_median,
                }
            )

    if not rows:
        raise ValueError("No categorical statistics could be computed (empty rows).")

    stats_df = pd.DataFrame(rows)

    out_dir_corner = os.path.join(output_dir, "corner")
    os.makedirs(out_dir_corner, exist_ok=True)

    stats_path = os.path.join(out_dir_corner, "categorical_posterior_stats.csv")
    stats_df.to_csv(stats_path, index=False)

    # -------------------------------
    # 5. Histograms: raw vs weighted
    # -------------------------------
    for col in cat_cols:
        sub = stats_df[stats_df["param"] == col].copy()
        if sub.empty:
            continue

        # sort by category value for tidier plots
        sub = sub.sort_values(by="category")

        cats = sub["category"].to_list()
        cats_labels = [str(c) for c in cats]
        raw_frac = sub["raw_frac"].to_numpy(float)
        w_frac = sub["weight_frac"].to_numpy(float)

        x = np.arange(len(cats))
        width = 0.4

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - 0.5 * width, raw_frac, width=width, label="raw")
        ax.bar(x + 0.5 * width, w_frac, width=width, label="weighted")

        ax.set_xticks(x)
        ax.set_xticklabels(cats_labels)
        ax.set_xlabel(col)
        ax.set_ylabel("Fraction in posterior HPD")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        out_pdf = os.path.join(out_dir_corner, f"posterior_cat_{col}.pdf")
        fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
        plt.close(fig)
