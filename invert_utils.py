# --------------------------------------------------------------
# Created by Kaan Cökerim¹ on 01. December 2025
#
# Utility scripts for the inversion routine
#
# ¹Tectonic Geodesy, Ruhr University Bochum, Germany
# Email: kaan.coekerim@rub.de
# --------------------------------------------------------------

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from typing import Literal


# ------------- DEBUGGING CUBOIDS
def get_debug_cuboids() -> np.ndarray:
    """
    Return set of pre-defined debug cuboids for sanity checking.
    Returns:
        Array of 5 pre-defined cuboids for debugging and sanity checking.
    """
    X_debug = np.array([
        [0, 0, 0.5, 0.3, 0.3, 0.3, 0, 1, 0, 1, 0, 1],  # cube
        [0, 0, 0.5, 0.2, 0.2, 0.5, 0, 1, 0, 1, 0, 1],  # cuboid
        [0, 0, 0.1, 0.2, 0.5, 0.2, 0, 1, 0, 1, 0, 1],  # cuboid sticking out
        [0, 0, 0.5, 0.0005, 0.5, 0.5, 0, 1, 0, 1, 0, 1],  # Plane
        [0, 0, 0.5, 0.0005, 0.5, 0.0005, 0, 1, 0, 1, 0, 1],  # Line
    ])

    return X_debug

# ------------- RANDOM CUBOIDS
def build_random_cuboids(
        n_samples: int,
        value_ranges: Literal['default'] | dict[str, tuple[float, float]] = 'default',
        deterministic: bool = True
                         ) -> np.ndarray:
    """
    Generate N random cuboids.
    Args:
        n_samples: number of samples to generate
        value_ranges: value ranges for the 12 cuboid source parameters
        deterministic: whether to use deterministic generation for reproducibility

    Returns:
        Array of N random generated cuboids
    """

    if isinstance(value_ranges, dict):
        assert len(value_ranges) == 12
    elif value_ranges == 'default':
        value_ranges = {'c1': (-1, 1), 'c2': (-1, 1), 'c3': (1e-4, 1),
                        'L': (1e-4, 1), 'W': (1e-4, 1), 'T': (1e-4, 1),
                        'e11': (-1, 1), 'e12': (-1, 1), 'e13': (-1, 1), 'e22': (-1, 1), 'e23': (-1, 1), 'e33': (-1, 1)}
    else:
        raise ValueError("value_ranges must be 'default' or a dict.")

    if deterministic:
        rng = np.random.default_rng(seed=42)
    else:
        rng = np.random.Generator(np.random.PCG64())

    keys = list(value_ranges.keys())
    n_features = len(keys)
    X_generated = np.empty((n_samples, n_features))

    # --- 1. Sample c3 first (centroid depth) ---
    if 'c3' not in value_ranges or 'W' not in value_ranges:
        raise KeyError("value_ranges must contain 'c3' and 'W' keys.")

    c3_min, c3_max = value_ranges['c3']
    c3 = rng.uniform(c3_min, c3_max, size=n_samples)

    # --- 2. Sample W conditional on c3 so that c3 - W/2 > 0 ---
    W_min, W_max_global = value_ranges['W']
    W = np.empty_like(c3)

    for i in range(n_samples):
        # constraint: W < 2*c3[i]
        W_hi = min(W_max_global, 2 * c3[i])
        W_lo = W_min
        W[i] = rng.uniform(W_lo, W_hi)

    # --- 3. Fill X_generated in the given order ---
    for j, key in enumerate(keys):
        if key == 'c3':
            X_generated[:, j] = c3
        elif key == 'W':
            X_generated[:, j] = W
        else:
            lo, hi = value_ranges[key]
            X_generated[:, j] = rng.uniform(lo, hi, size=n_samples)

    return X_generated


# ------------- INVERSION METRICS

def scale_invariant_mse(eps: float = 1e-12, clamp_nonneg: bool = False):
    """
    Scale-invariant MSE:
      s* = <y, yhat> / (<yhat, yhat> + eps)
      loss = ||y - s* yhat||^2

    y_true, y_pred: (B, H, W, C)
    """
    def loss(y_true, y_pred):
        # per-sample dot products
        num = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        den = tf.reduce_sum(y_pred * y_pred, axis=[1, 2, 3]) + eps
        s = num / den  # (B,)

        if clamp_nonneg:
            # optional: prevents a global sign flip via s
            s = tf.nn.relu(s)

        s = tf.reshape(s, (-1, 1, 1, 1))  # broadcast
        resid = y_true - s * y_pred
        return tf.reduce_mean(tf.square(resid))  # scalar
    return loss

def scale_invariant_rel_mse(eps: float = 1e-12, clamp_nonneg: bool = False, stop_grad_s: bool = True):
    """
    Scale-invariant *relative* MSE:
      s* = <y, yhat> / (<yhat, yhat> + eps)
      loss_i = ||y - s* yhat||^2 / (||y||^2 + eps)
      return mean_i loss_i
    """
    def loss(y_true, y_pred):
        num = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])                 # (B,)
        den = tf.reduce_sum(y_pred * y_pred, axis=[1, 2, 3]) + eps           # (B,)
        s = num / den                                                       # (B,)

        if clamp_nonneg:
            s = tf.nn.relu(s)

        if stop_grad_s:
            s = tf.stop_gradient(s)

        s = tf.reshape(s, (-1, 1, 1, 1))
        resid = y_true - s * y_pred

        resid2 = tf.reduce_sum(tf.square(resid), axis=[1, 2, 3])             # (B,)
        y2     = tf.reduce_sum(tf.square(y_true), axis=[1, 2, 3]) + eps      # (B,)

        return tf.reduce_mean(resid2 / y2)

    return loss


def optimal_scale(y_obs: np.ndarray, y_pred: np.ndarray, clamp_nonneg: bool=True) -> float:
    # y_obs, y_pred: (64,64,3)
    num = np.sum(y_obs * y_pred)
    den = np.sum(y_pred * y_pred) + 1e-12
    s = num / den
    if clamp_nonneg:
        s = max(s, 0.0)
    return float(s)


# ------------- MULTI START SAMPLING
def sample_feasible_cuboids_ms(
    M,
    eps=1e-4,
    max_extent=1.0,
    seed=None,
    plane_frac=0.1,
    thin_max=5e-3,            # max thickness for "plane-like"
    thin_dims=("L", "T", "W")  # which dimension may become thin
):
    rng = np.random.default_rng(seed)
    X = np.empty((M, 12), dtype=np.float32)

    # centroids
    c1 = rng.uniform(-1.0, 1.0, size=(M, 1))
    c2 = rng.uniform(-1.0, 1.0, size=(M, 1))
    c3 = rng.uniform(eps, 1.0 - eps, size=(M, 1))

    # halfspace-feasible W maximum
    Wmax = np.clip(2.0 * np.minimum(c3 - eps, (1.0 - eps) - c3), eps, max_extent)

    # base extents
    L = rng.uniform(eps, max_extent, size=(M, 1))
    T = rng.uniform(eps, max_extent, size=(M, 1))
    W = rng.uniform(eps, 1.0, size=(M, 1)) * Wmax

    # plane-biased subset
    n_plane = int(plane_frac * M)
    if n_plane > 0:
        idx = rng.choice(M, size=n_plane, replace=False)

        # choose which dim is thin for each plane sample
        dim_choices = np.array(thin_dims)
        thin_sel = rng.choice(dim_choices, size=n_plane, replace=True)

        # log-uniform thin thickness in [eps, thin_max]
        log_eps = np.log(eps)
        log_hi = np.log(max(thin_max, eps * 1.01))
        thin = np.exp(rng.uniform(log_eps, log_hi, size=(n_plane, 1))).astype(np.float32)

        # enforce feasibility if thin dim is W (must also be <= Wmax)
        if "W" in thin_dims:
            wcap = Wmax[idx]
            thin_W = np.minimum(thin, wcap)

        # push the other two dims large to help avoid collapse
        large = rng.uniform(0.6 * max_extent, max_extent, size=(n_plane, 1)).astype(np.float32)

        for j, di in enumerate(thin_sel):
            ii = idx[j]
            if di == "L":
                L[ii] = thin[j]
                T[ii] = large[j]
                W[ii] = np.minimum(W[ii], Wmax[ii])
            elif di == "T":
                T[ii] = thin[j]
                L[ii] = large[j]
                W[ii] = np.minimum(W[ii], Wmax[ii])
            elif di == "W":
                W[ii] = thin_W[j]
                L[ii] = large[j]
                T[ii] = large[j]

    # strain components
    strain = rng.uniform(-1.0, 1.0, size=(M, 6))
    strain /= (np.linalg.norm(strain, axis=1, keepdims=True) + 1e-12)

    X[:, 0:1] = c1
    X[:, 1:2] = c2
    X[:, 2:3] = c3
    X[:, 3:4] = L
    X[:, 4:5] = W
    X[:, 5:6] = T
    X[:, 6:] = strain.astype(np.float32)
    return X

def ms_best_seeds(fwd_model, y_obs, M=3000, topk=10, seed=42, clamp_nonneg=True):
    Xcand = sample_feasible_cuboids_ms(M, seed=seed)
    Ycand = fwd_model.predict(Xcand, verbose=0)  # (M,64,64,3)

    y = y_obs[None, ...]  # (1,64,64,3)

    # optimal scale per candidate: s = <y, yhat> / <yhat, yhat>
    num = np.sum(Ycand * y, axis=(1, 2, 3))
    den = np.sum(Ycand * Ycand, axis=(1, 2, 3)) + 1e-12
    s = num / den
    if clamp_nonneg:
        s = np.maximum(s, 0.0)

    resid = y - s[:, None, None, None] * Ycand
    mse = np.mean(resid * resid, axis=(1, 2, 3))

    idx = np.argsort(mse)[:topk]
    return Xcand[idx], mse[idx], s[idx]


# ------------- LOAD EXTERNAL DATA

def _load_faults(fault_path: str, norm_disp: bool = False) -> (np.ndarray, np.ndarray, list, str):
    target_geometry = 'Faults'

    data = loadmat(fault_path)
    y_arr = data['targets']
    # ENU -> NED
    y_arr = y_arr[..., [1, 0, 2]]
    y_arr[..., 2] *= -1

    if norm_disp:
        m = np.max(np.abs(y_arr), axis=(1, 2, 3), keepdims=True) + 1e-12
        y_arr = y_arr / m

    x_arr = np.empty((y_arr.shape[0], 8))
    data['z_cen_inputs'] *= -1

    external_keys = [
        "x_cen", "y_cen", "z_cen",
        "strike", "dip", "rake",
        "fault_length", "fault_width"
    ]
    for sample in range(y_arr.shape[0]):
        x_arr[sample] = np.array([data[k + "_inputs"][sample][0] for k in external_keys])

    return y_arr, x_arr, external_keys, target_geometry


def _load_mogi(mogi_path: str, norm_disp: bool = False) -> (np.ndarray, np.ndarray, list, str):
    target_geometry = 'Mogi'

    data = loadmat(mogi_path)
    y_arr = data["U_NED"]
    y_arr = np.transpose(y_arr, (0, 2, 1, 3))

    if norm_disp:
        m = np.max(np.abs(y_arr), axis=(1, 2, 3), keepdims=True) + 1e-12
        y_arr = y_arr / m

    c1, c2, c3 = data['c1'], data['c2'], data['c3']
    a = data['A']
    x_arr = np.stack([c1[:, 0], c2[:, 0], c3[:, 0], a[:, 0]], axis=1)
    external_keys = ['c1', 'c2', 'c3', 'a']

    return y_arr, x_arr, external_keys, target_geometry



def load_external_ys(external_path: str, norm_disp: bool = False) -> (np.ndarray, np.ndarray, list, str):
    file_ext = os.path.splitext(external_path)[1]

    if file_ext in ['.m', '.mat'] and 'fault' in os.path.split(external_path)[1].lower():
        y_arr, x_arr, external_keys, target_geometry = _load_faults(fault_path=external_path, norm_disp=norm_disp)

    elif file_ext in ['.m', '.mat'] and 'mogi' in os.path.split(external_path)[1].lower():
        y_arr, x_arr, external_keys, target_geometry = _load_mogi(mogi_path=external_path, norm_disp=norm_disp)

    else:
        raise IOError(f'{external_path} is not a supported file type.')

    return y_arr, x_arr, external_keys, target_geometry

