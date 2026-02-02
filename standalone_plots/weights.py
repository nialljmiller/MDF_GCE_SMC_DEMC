# standalone_plots/weights.py
import numpy as np
import pandas as pd

def compute_weights_from_loss(loss_array, temperature=1.0):
    """
    Turn a vector of loss values into normalized posterior weights
    via exp(-loss / T).
    No error handling, no clipping.
    """
    loss_array = np.asarray(loss_array, dtype=float)
    shifted = loss_array - np.nanmin(loss_array)
    logw = -shifted / float(temperature)
    w = np.exp(logw - np.max(logw))
    w_sum = np.sum(w)
    w_norm = w / w_sum
    return w_norm

def attach_posterior_weights(df, loss_col="loss", temperature=1.0):
    w = compute_weights_from_loss(df[loss_col].to_numpy(), temperature=temperature)
    out = df.copy()
    out["posterior_w"] = w
    return out


def temp_calc(df, loss_col="loss", target_ess_frac=0.25, T_lo=1e-3, T_hi=1e3, iters=40):
    import numpy as np
    L = np.asarray(df[loss_col].to_numpy(), float)
    N = len(L)
    for _ in range(iters):
        T = 0.5 * (T_lo + T_hi)
        shifted = L - L.min()
        logw = -shifted / T
        w = np.exp(logw - logw.max())
        w /= w.sum()
        ess = 1.0 / np.sum(w * w)
        if ess < target_ess_frac * N:
            T_hi = T
        else:
            T_lo = T
    T = 0.5 * (T_lo + T_hi)
    return T

