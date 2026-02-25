# --------------------------------------------------------------
# Created by Kaan Cökerim¹ on 29. January 2026
#
# Plotting of MSE relation between cuboid parameters
#
# ¹Tectonic Geodesy, Ruhr University Bochum, Germany
# Email: kaan.coekerim@rub.de
# --------------------------------------------------------------

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from h5_dataloader import H5CuboidDataloader, get_num_samples
from invert_for_cuboids import load_fwd_model
import pandas as pd
from tqdm import trange
import seaborn as sns


@tf.function
def mse_per_sample(model, x, y):
    """
    Returns a scalar MSE per sample: mean over all non-batch axes.
    """
    y_pred = model(x, training=False)
    diff = tf.cast(y_pred, tf.float64) - tf.cast(y, tf.float64)
    se = tf.square(diff)
    return tf.reduce_mean(se, axis=tf.range(1, tf.rank(se)))  # (B,)

def gather_dataframe(h5_path, model_path):
    n_samples = get_num_samples(h5_path)

    full_set = np.arange(n_samples)
    Train_Val_Test = np.array([80, 10, 10])
    pp1 = int((Train_Val_Test[0] / Train_Val_Test.sum()) * full_set.size)
    train_idx = full_set[:pp1]

    BS = 512
    train_seq = H5CuboidDataloader(
        filename=h5_path,
        indices=train_idx,
        batch_size=BS,
        workers=0,
        use_multiprocessing=False,
        max_queue_size=1,
    )

    model_name = os.path.split(model_path)[-1][:-27]
    fwd_model = load_fwd_model(
        model_path=model_path,
        n_var_layers=13,
    )

    # Preallocate
    N = len(train_idx)
    X_out = np.empty((N, 12), dtype=np.float32)
    mse_out = np.empty((N,), dtype=np.float64)

    off = 0
    for i in trange(len(train_seq)):
        X, Y = train_seq[i]  # X: (B,12), Y: (B,64,64,3) scaled by loader

        for samp_id in range(X.shape[0]):
            misfit = fwd_model.evaluate(X[samp_id][None, ...], Y[samp_id][None, ...])
            mse_out[off + samp_id] = misfit[0]


        b = X.shape[0]
        off += b

    # Build DF once
    df = pd.DataFrame(X_out, columns=[f"x{i+1}" for i in range(12)])
    df["mse"] = mse_out

    # close file handle
    if getattr(train_seq, "_file", None) is not None:
        train_seq._file.close()
        train_seq._file = None

    return df


def apply_pairgrid_axis_limits(g, bounds, pad=0.0):
    """
    Set x/y limits for each facet based on variable-specific bounds.

    bounds: dict {var_name: (lo, hi)}
    pad: optional fractional padding
    """
    x_vars = list(getattr(g, "x_vars", []))
    y_vars = list(getattr(g, "y_vars", []))

    for i, yv in enumerate(y_vars):
        for j, xv in enumerate(x_vars):
            ax = g.axes[i, j]
            if ax is None or not ax.get_visible():
                continue

            if xv in bounds:
                lo, hi = bounds[xv]
                if pad:
                    d = (hi - lo) * pad
                    lo, hi = lo - d, hi + d
                ax.set_xlim(lo, hi)
                ax.set_xticks([lo, (lo+hi)/2, hi])

            if yv in bounds:
                lo, hi = bounds[yv]
                if pad:
                    d = (hi - lo) * pad
                    lo, hi = lo - d, hi + d
                ax.set_ylim(lo, hi)
                ax.set_yticks([lo, (lo+hi)/2, hi])


if __name__ == '__main__':
    model_path = "./checkpoints/LARGE_v2i_max_entropy_C2DT_2025_12_19_1547_ckpt.keras"
    h5_path = "./synthetic_data_generation/example_training_cuboids.h5"

    df = gather_dataframe(h5_path, model_path)
    print(df)
    print(df.columns)
    print(df.shape)

    df.to_parquet("inputs_mse.parquet", index=False)

    df = pd.read_parquet('inputs_mse.parquet')
    df.columns = ['c1', 'c2', 'c3', 'L', 'W', 'T',
                  r"$\varepsilon_{11}$", r"$\varepsilon_{12}$", r"$\varepsilon_{13}$",
                  r"$\varepsilon_{22}$", r"$\varepsilon_{23}$", r"$\varepsilon_{33}$",
                  'MSE']
    print(df['MSE'].min(), df['MSE'].max(), df['MSE'].quantile(0.95), df['MSE'].quantile(0.05))

    df = df.sort_values(by=['MSE'], ascending=False)
    print(df)

    param_bounds = {
        "c1": (-1.0, 1.0),
        "c2": (-1.0, 1.0),
        "c3": (0.0, 1.0),
        "L": (0.0, 1.0),
        "W": (0.0, 1.0),
        "T": (0.0, 1.0),
        r"$\varepsilon_{11}$": (-1.0, 1.0),
        r"$\varepsilon_{12}$": (-1.0, 1.0),
        r"$\varepsilon_{13}$": (-1.0, 1.0),
        r"$\varepsilon_{22}$": (-1.0, 1.0),
        r"$\varepsilon_{23}$": (-1.0, 1.0),
        r"$\varepsilon_{33}$": (-1.0, 1.0),
    }

    x_vars = list(df.columns[0:-1])
    y_vars = list(df.columns[0:-1])

    cmap = sns.cm.rocket
    #norm = plt.matplotlib.colors.Normalize(vmin=0.0, vmax=df["MSE"].quantile(0.95))
    norm = plt.matplotlib.colors.LogNorm(vmin=1e-5, vmax=5e-4)

    g = sns.PairGrid(df, hue='MSE', corner=False, palette='rocket', aspect=1, height=2,
                     x_vars=x_vars, y_vars=y_vars
                     )
    g.map(sns.scatterplot, hue_norm=norm, s=3, edgecolors='none', alpha=0.2)
    g.fig.subplots_adjust(right=0.86)

    apply_pairgrid_axis_limits(g, param_bounds, pad=0.0)

    # Add a colorbar for mse
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig = g.figure
    cax = fig.add_axes((0.88, 0.4, 0.01, 0.3))
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(label='\n'+r'Misfit ($m^2)$', fontsize=16)

    plt.savefig('./figs/parameter_misfit.png', format='png', dpi=300, bbox_inches='tight')

    plt.show()