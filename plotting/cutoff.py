import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys



def choose_cutoff_lognorm_mixture(df_sorted, bins=100, kde_points=1024, em_max_iter=200, tol=1e-6, force_k2=False):
    """
    Simple, reviewer-proof cutoff:
      - Work in y = log(loss).
      - Fit K=1 and K=2 Gaussian mixtures in y by EM; pick K by BIC (unless force_k2=True).
      - If K=2: cutoff = equal-responsibility boundary where pi1*N1(y)=pi2*N2(y).
      - If K=1: no hard cut (use all models).
    Writes two plots and a small audit file; returns cutoff & realized keep fraction.
    """
    import os, numpy as np, matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde, norm
    from scipy.special import logsumexp

    # ---------- data ----------
    L = np.asarray(df_sorted['fitness'].values, float)
    L = L[np.isfinite(L)]
    if L.size == 0:
        raise RuntimeError("No finite losses/fitness values.")
    eps = 1e-12
    y = np.log(L + eps)
    N = y.size

    # ---------- helper: EM for 1D K-component Gaussian mixture ----------
    def em_gmm_1d(y, K, iters=200, tol=1e-6):
        # init by quantiles
        qs = np.linspace(0.2, 0.8, K)
        mu = np.quantile(y, qs) if K > 1 else np.array([float(np.mean(y))])
        s0 = float(np.std(y))
        s0 = s0 if s0 > 1e-6 else 0.1
        sig = np.full(K, s0, float)
        pi = np.full(K, 1.0 / K, float)

        c_norm = -0.5*np.log(2*np.pi)

        def logpdf(y, mu, sig):
            return c_norm - np.log(sig) - 0.5*((y - mu)/sig)**2

        prev_ll = -np.inf
        for _ in range(iters):
            # E-step: responsibilities (log-space)
            log_comp = np.stack([np.log(pi[k]) + logpdf(y, mu[k], sig[k] + 1e-12) for k in range(K)], axis=1)
            log_den = logsumexp(log_comp, axis=1, keepdims=True)
            R = np.exp(log_comp - log_den)  # N x K
            Nk = R.sum(axis=0) + 1e-12

            # M-step
            mu_new = (R * y[:, None]).sum(axis=0) / Nk
            sig_new = np.sqrt((R * (y[:, None] - mu_new[None, :])**2).sum(axis=0) / Nk)
            sig_new = np.maximum(sig_new, 1e-6)
            pi_new = Nk / N

            # log-likelihood
            ll = float(np.sum(log_den))
            if abs(ll - prev_ll) < tol:
                mu, sig, pi = mu_new, sig_new, pi_new
                prev_ll = ll
                break
            mu, sig, pi, prev_ll = mu_new, sig_new, pi_new, ll

        # BIC: p = (K-1) + K (means) + K (stds) = 2K - 1
        bic = -2.0*prev_ll + (2*K - 1)*np.log(N)
        # order by mean
        order = np.argsort(mu)
        return pi[order], mu[order], sig[order], prev_ll, bic

    # ---------- fit K=1 and K=2 ----------
    pi1, mu1, sg1, ll1, bic1 = em_gmm_1d(y, 1, em_max_iter, tol)
    pi2, mu2, sg2, ll2, bic2 = em_gmm_1d(y, 2, em_max_iter, tol)
    choose_K2 = force_k2 or (bic2 < bic1)

    # ---------- cutoff (if K=2), else None ----------
    loss_cutoff = None
    chosen_K = 2 if choose_K2 else 1
    if choose_K2:
        # components already ordered: comp0 is the elite (lower mu)
        pi = pi2; mu = mu2; sig = sg2

        # Solve pi0*N0(y) = pi1*N1(y) analytically
        A = 0.5*(1.0/sig[1]**2 - 1.0/sig[0]**2)
        B = (mu[0]/sig[0]**2 - mu[1]/sig[1]**2)
        D = 0.5*(mu[1]**2/sig[1]**2 - mu[0]**2/sig[0]**2)
        const = np.log((pi[1]/sig[1])/(pi[0]/sig[0]))
        C = D - const

        if abs(A) < 1e-12:
            y_cut = -C / (B + 1e-12)  # equal-variance fallback
        else:
            disc = max(B*B - 4*A*C, 0.0)
            roots = np.sort(( -B + np.array([-1.0, 1.0])*np.sqrt(disc) ) / (2*A))
            # prefer a root between the two means; otherwise, nearest to their midpoint
            mid = 0.5*(mu[0] + mu[1])
            if (mu[0] <= roots[0] <= mu[1]) or (mu[0] <= roots[1] <= mu[1]):
                y_cut = roots[0] if (mu[0] <= roots[0] <= mu[1]) else roots[1]
            else:
                y_cut = roots[np.argmin(np.abs(roots - mid))]

        loss_cutoff = float(np.exp(y_cut))
        frac = float(np.mean(L <= loss_cutoff))
    else:
        # no hard selection
        frac = 1.0

    pct = 100.0 * frac
    
    return pct


