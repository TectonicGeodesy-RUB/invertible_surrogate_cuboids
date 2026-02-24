# --------------------------------------------------------------
# Created by Kaan Cökerim¹ on 13. January 2026
#
# Scripts for multisource inversion
#
# ¹Tectonic Geodesy, Ruhr University Bochum, Germany
# Email: kaan.coekerim@rub.de
# --------------------------------------------------------------

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import cmcrameri.cm as cm
import numpy as np
import tensorflow as tf
import keras
from matplotlib import gridspec, colors
from matplotlib.lines import Line2D
from plot_utils import get_cuboid_corners, get_strain_tensor
from invert_for_cuboids import load_fwd_model
from multisource_utils import BoundedMatMulInputLayer_MultiSource, ScaleLayer_MultiSource
plt.style.use('seaborn-v0_8-colorblind')

color_true = '0.01'
color_inv = "darkorange"
color_init = '0.55'

line_style_true = 'solid'
line_style_inv = 'solid' # (0, (5, 3))  #'dashed'
line_style_init = 'dashed'

def multisource_fwd_model(
        model_path: str | os.PathLike,
        n_sources: int,
        remove_variational: bool = True,
        n_var_layers: int = 13,
        verbose_trainable: bool = True
):
    raw_model = keras.models.load_model(model_path)
    orig_model = keras.models.clone_model(raw_model)
    orig_model.set_weights(raw_model.get_weights())

    inp = keras.Input(shape=(n_sources, 1, 12))  # input layer
    x = inp

    if remove_variational:
        selected_layers = orig_model.layers[n_var_layers:]

        for layer in selected_layers:
            layer.trainable = False
            if isinstance(layer, keras.layers.Reshape):
                x = keras.layers.Reshape((n_sources, 8, 8, 64), trainable=False)(x)
            elif (isinstance(layer, keras.layers.Conv2D)
                  or isinstance(layer, keras.layers.Conv2DTranspose)
                  or isinstance(layer, keras.layers.BatchNormalization)
            ):
                x = keras.layers.TimeDistributed(layer, trainable=False)(x)
            else:
                x = layer(x)

    fwd_model = keras.Model(inputs=inp, outputs=x)
    fwd_model.layers[0].trainable = False  # set Input layer to NOT be trainable

    if verbose_trainable:
        fwd_model.summary()
        for j in range(len(fwd_model.layers)):
            if fwd_model.layers[j].trainable:
                raise ValueError(f'{fwd_model.layers[j].name} should not be trainable!')

    fwd_model.compile(loss='mse', optimizer='adam')

    return fwd_model


def multisource_inversion_model(
        model_path: str | os.PathLike,
        n_sources: int,
        remove_variational: bool = True,
        n_var_layers: int = 13,
        use_constraints: bool = False,
        verbose_trainable: bool = True
):
    raw_model = keras.models.load_model(model_path)
    orig_model = keras.models.clone_model(raw_model)
    orig_model.set_weights(raw_model.get_weights())

    inp = keras.Input(shape=(n_sources, 12, 12))

    matmul_layer = BoundedMatMulInputLayer_MultiSource(n_sources=n_sources, use_constrains=use_constraints)
    x = matmul_layer(inp)

    if remove_variational:
        selected_layers = orig_model.layers[n_var_layers:]
    else:
        selected_layers = orig_model.layers[1:]

    for layer in selected_layers:
        layer.trainable = False
        if isinstance(layer, keras.layers.Reshape):
            x = keras.layers.Reshape((n_sources, 8, 8, 64), trainable=False)(x)
        elif (isinstance(layer, keras.layers.Conv2D)
              or isinstance(layer, keras.layers.Conv2DTranspose)
              or isinstance(layer, keras.layers.BatchNormalization)
        ):
            x = keras.layers.TimeDistributed(layer, trainable=False)(x)
        else:
            x = layer(x)

    x = ScaleLayer_MultiSource(n_sources=n_sources)(x)

    inv_model = keras.Model(inputs=inp, outputs=x)
    inv_model.layers[0].trainable = False  # set Input layer to NOT be trainable
    inv_model.layers[1].trainable = True  # set MatMulInputLayer to be trainable
    inv_model.layers[-1].trainable = True  # set ScaleLayer to be trainable

    if verbose_trainable:
        inv_model.summary()
        for j in range(len(inv_model.layers)):
            print(inv_model.layers[j], 'Trainable? ', inv_model.layers[j].trainable)

    inv_model.compile(loss='mse', optimizer='adam')

    return inv_model


def build_multisources(model_path, n_sources=2, debug=True):

    X_true = np.array([
        [0.5, -0.2, 0.3, 0.2, 0.3, 0.4, 0, 1, 0, 0, 1, 1],  # cube
        [-0.4, 0.3, 0.5, 0.2, 0.4, 0.3, 1, 0, 0, 1, 0, 1],  # cuboid
    ])

    # norm strain tensor
    X_true[:, 6:] = X_true[:, 6:] / np.linalg.norm(X_true[:, 6:], axis=-1, keepdims=True)

    # load model
    trained_model = load_fwd_model(model_path=model_path, n_var_layers=13)

    Y_true_sep = trained_model.predict(X_true)

    Y_true = Y_true_sep.sum(axis=0)

    return X_true, Y_true, Y_true_sep

def plot_3d_cuboids(x_init, x_true, x_inv):
    fig = plt.figure(figsize=(8, 10))

    # Grid: 2 rows heatmaps + 1 big 3D row
    gs = gridspec.GridSpec(
        3, 3,
        height_ratios=[0.18, 0.18, 0.64],
        wspace=0.05, hspace=0.18
    )

    ax_hms_1 = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_hms_2 = [fig.add_subplot(gs[1, i]) for i in range(3)]
    ax3d = fig.add_subplot(gs[2, :], projection="3d")

    # --- strain tensors ---
    e_init_1 = get_strain_tensor(x_init[0][None, ...])
    e_init_2 = get_strain_tensor(x_init[1][None, ...])
    e_inv_1 = get_strain_tensor(x_inv[0][None, ...])
    e_inv_2 = get_strain_tensor(x_inv[1][None, ...])
    e_true_1 = get_strain_tensor(x_true[0][None, ...])
    e_true_2 = get_strain_tensor(x_true[1][None, ...])

    tensors = [
        [e_init_1, e_inv_1, e_true_1],
        [e_init_2, e_inv_2, e_true_2],
    ]

    vmin, vmax = -1, 1
    cmap = cm.vik

    # Draw all heatmaps
    for j, ax in enumerate(ax_hms_1):
        sns.heatmap(
            tensors[0][j], ax=ax,
            cmap=cmap, vmin=vmin, vmax=vmax,
            cbar=False, square=True,
            xticklabels=False, yticklabels=False,
            linewidths=.8,
            annot=True, fmt=".2f",
            annot_kws={"fontsize": 9}
        )
        ax.set_title(["\n\nInitial", "a) Eigenstrain Tensors\n\nInverted", "\n\nTrue"][j], fontsize=12, pad=6)

    for j, ax in enumerate(ax_hms_2):
        sns.heatmap(
            tensors[1][j], ax=ax,
            cmap=cmap, vmin=vmin, vmax=vmax,
            cbar=False, square=True,
            xticklabels=False, yticklabels=False,
            linewidths=.8,
            annot=True, fmt=".2f",
            annot_kws={"fontsize": 9}
        )

    # --- shared colorbar for all strain heatmaps ---
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Compute bounding box around the 2x3 heatmap block, then place cbar to the right
    bboxes = [ax.get_position() for ax in (ax_hms_1 + ax_hms_2)]
    left = min(bb.x0 for bb in bboxes)
    right = max(bb.x1 for bb in bboxes)
    bottom = min(bb.y0 for bb in bboxes)
    top = max(bb.y1 for bb in bboxes)

    pad = 0.012
    cbar_w = 0.010
    cax = fig.add_axes([right + pad, bottom, cbar_w, top - bottom])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_label(r"Strain Components $\epsilon_{ij}$ (-)", rotation=90, labelpad=10)
    cbar.outline.set_edgecolor('#f9f2d7')

    # Row labels
    row1_mid = 0.5 * (ax_hms_1[0].get_position().y0 + ax_hms_1[0].get_position().y1)
    row2_mid = 0.5 * (ax_hms_2[0].get_position().y0 + ax_hms_2[0].get_position().y1)
    fig.text(left - 0.02, row1_mid, r"Source 1: $\epsilon_{ij}^{(1)}$", rotation=90,
             ha="center", va="center", fontsize=11)
    fig.text(left - 0.02, row2_mid, r"Source 2: $\epsilon_{ij}^{(2)}$", rotation=90,
             ha="center", va="center", fontsize=11)

    # --- 3D cuboids ---
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]

    lw_init, lw_inv, lw_true = 1.2, 1.8, 1.8
    ls_init, ls_inv, ls_true = (0, (3, 2)), "-", "-"

    ax3d.set_proj_type("ortho")
    ax3d.view_init(elev=22, azim=-55)
    ax3d.grid(True)

    # transparent panes + lighter axis lines
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((0.85, 0.85, 0.85, 1))
    ax3d.xaxis.line.set_color("0.6")
    ax3d.yaxis.line.set_color("0.6")
    ax3d.zaxis.line.set_color("0.6")
    ax3d.tick_params(colors="0.35", labelsize=9)

    for idx_source in range(x_init.shape[0]):
        corners_init, c_init, _ = get_cuboid_corners(x_init[idx_source][None, ...])
        corners_inv, c_inv, _ = get_cuboid_corners(x_inv[idx_source][None, ...])
        corners_true, c_true, _ = get_cuboid_corners(x_true[idx_source][None, ...])

        # Initial
        ax3d.plot(*c_init, marker="o", color=color_init, markersize=4)
        for e in edges:
            pts = corners_init[e]
            ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color_init, lw=lw_init, ls=ls_init)

        # Inverted
        ax3d.plot(*c_inv, marker="*", color=color_inv, markersize=7)
        for e in edges:
            pts = corners_inv[e]
            ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color_inv, lw=lw_inv, ls=ls_inv, zorder=100)

        # True
        ax3d.plot(*c_true, marker="^", color=color_true, markersize=5)
        for e in edges:
            pts = corners_true[e]
            ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color_true, lw=lw_true, ls=ls_true)

        # Optimization path
        opt_path = np.array([c_init, c_inv, c_true])
        ax3d.plot(opt_path[:, 0], opt_path[:, 1], opt_path[:, 2], color="0.25", lw=1.0, ls="--", alpha=0.8)

        # Source id annotation near the centroid
        ax3d.text(c_true[0]+1e-4, c_true[1]+1e-4, c_true[2]+1e-4, f"{idx_source + 1}", color=color_true,
                  fontsize=10, zorder=200, fontweight="bold")

    # Legend
    handles = [
        Line2D([0], [0], color=color_init, lw=lw_init, ls=ls_init, label="Initial"),
        Line2D([0], [0], color=color_inv, lw=lw_inv, ls=ls_inv, label="Inverted"),
        Line2D([0], [0], color=color_true, lw=lw_true, ls=ls_true, label="True"),
    ]
    ax3d.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.9)

    ax3d.set_xlabel(r"$x_1$ (North)")
    ax3d.set_ylabel(r"$x_2$ (East)")
    ax3d.set_zlabel(r"$x_3$ (Down)")

    ax3d.set_xlim([-1, 1])
    ax3d.set_ylim([-1, 1])
    ax3d.set_zlim([1, 0])
    ax3d.set_aspect('equal')

    # Panel label for 3D axis
    ax3d.set_title('b) Cuboid Geometry', pad=0.0)

    return fig


def plot_3d_cuboids_OLD(x_init, x_true, x_inv):

    fig = plt.figure(figsize=(8, 12))
    gs = gridspec.GridSpec(3, 3, height_ratios=[0.18, 0.18, 0.64])

    ax_hms_1 = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_hms_2 = [fig.add_subplot(gs[1, i]) for i in range(3)]
    ax = fig.add_subplot(gs[2, :], projection="3d")

    # strain tensors
    e_init_1 = get_strain_tensor(x_init[0][None, ...])
    e_init_2 = get_strain_tensor(x_init[1][None, ...])

    e_inv_1 = get_strain_tensor(x_inv[0][None, ...])
    e_inv_2 = get_strain_tensor(x_inv[1][None, ...])

    e_true_1 = get_strain_tensor(x_true[0][None, ...])
    e_true_2 = get_strain_tensor(x_true[1][None, ...])

    sns.heatmap(e_init_1, ax=ax_hms_1[0], cmap=cm.vik, vmin=-1, vmax=1, cbar=True, square=True,
                xticklabels=False, yticklabels=False, linewidths=.8, annot=True, fmt=".2f",
                cbar_kws={"shrink": 0.8, "label": r"$\epsilon^{(1)}_{ij}$ Initial (-)"})
    sns.heatmap(e_inv_1, ax=ax_hms_1[1], cmap=cm.vik, vmin=-1, vmax=1, cbar=True, square=True,
                xticklabels=False, yticklabels=False, linewidths=.8, annot=True, fmt=".2f",
                cbar_kws={"shrink": 0.8, "label": r"$\epsilon^{(1)}_{ij}$ Inverted (-)"})
    sns.heatmap(e_true_1, ax=ax_hms_1[2], cmap=cm.vik, vmin=-1, vmax=1, cbar=True, square=True,
                xticklabels=False, yticklabels=False, linewidths=.8, annot=True, fmt=".2f",
                cbar_kws={"shrink": 0.8, "label": r"$\epsilon^{(1)}_{ij}$ True (-)"})

    sns.heatmap(e_init_2, ax=ax_hms_2[0], cmap=cm.vik, vmin=-1, vmax=1, cbar=True, square=True,
                xticklabels=False, yticklabels=False, linewidths=.8, annot=True, fmt=".2f",
                cbar_kws={"shrink": 0.8, "label": r"$\epsilon^{(2)}_{ij}$ Initial (-)"})
    sns.heatmap(e_inv_2, ax=ax_hms_2[1], cmap=cm.vik, vmin=-1, vmax=1, cbar=True, square=True,
                xticklabels=False, yticklabels=False, linewidths=.8, annot=True, fmt=".2f",
                cbar_kws={"shrink": 0.8, "label": r"$\epsilon^{(2)}_{ij}$ Inverted (-)"})
    sns.heatmap(e_true_2, ax=ax_hms_2[2], cmap=cm.vik, vmin=-1, vmax=1, cbar=True, square=True,
                xticklabels=False, yticklabels=False, linewidths=.8, annot=True, fmt=".2f",
                cbar_kws={"shrink": 0.8, "label": r"$\epsilon^{(2)}_{ij}$ True (-)"})

    ax_hms_1[0].set_title('Initial')
    ax_hms_1[1].set_title('Inverted')
    ax_hms_1[2].set_title('True')

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom square
        [4, 5], [5, 6], [6, 7], [7, 4],  # top square
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
    ]

    for idx_source in range(x_init.shape[0]):
        corners_init, c_init, dims_init = get_cuboid_corners(x_init[idx_source][None, ...])
        corners_inv, c_inv, dims_inv = get_cuboid_corners(x_inv[idx_source][None, ...])
        corners_true, c_true, dims_true = get_cuboid_corners(x_true[idx_source][None, ...])

        ax.plot(*c_init, 'bo')
        for e in edges:
            ax.plot(corners_init[e][:, 0], corners_init[e][:, 1], corners_init[e][:, 2], color='blue')

        ax.plot(*c_inv, 'r*')
        for e in edges:
            ax.plot(corners_inv[e][:, 0], corners_inv[e][:, 1], corners_inv[e][:, 2], color='red')

        ax.plot(*c_true, 'g^')
        for e in edges:
            ax.plot(corners_true[e][:, 0], corners_true[e][:, 1], corners_true[e][:, 2], color='green')

        opt_path = np.array([c_init, c_inv, c_true])
        ax.plot(opt_path[:, 0], opt_path[:, 1], opt_path[:, 2], 'k--')

    custom_lines = [Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], color='green', lw=2) if x_true is not None else None]

    ax.legend(custom_lines, ['Initial', 'Inverted', 'True' if x_true is not None else None])

    ax.set_xlabel(r'$x_1$ (North)')
    ax.set_ylabel(r'$x_2$ (East)')
    ax.set_zlabel(r'$x_3$ (Down)')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([1, 0])
    ax.xaxis.set_ticks(np.linspace(-1, 1, 11, endpoint=True))
    ax.yaxis.set_ticks(np.linspace(-1, 1, 11, endpoint=True))
    ax.zaxis.set_ticks(np.linspace(0, 1, 6, endpoint=True))
    ax.set_aspect('equal')

    return fig


def plot_surfaces(x_init, x_true, x_inv, Y_true, Y_inv):

    x = np.linspace(-1, 1, Y_true.shape[1], endpoint=True)
    y = np.linspace(-1, 1, Y_true.shape[0], endpoint=True)
    x1, x2 = np.meshgrid(x, y)

    # reorder ys  from NED -> ENU
    Y_true = Y_true[..., [1, 0, 2]]
    Y_inv = Y_inv[..., [1, 0, 2]]

    Y_true[..., 2] *= -1
    Y_inv[..., 2] *= -1

    norm = colors.CenteredNorm(vcenter=0, halfrange=np.abs(Y_true).max() / 2)
    cmap = cm.vik
    fig, (ax1, ax2) = plt.subplots(figsize=(12, 9), nrows=2, ncols=3, constrained_layout=False)

    fig.subplots_adjust(
        left=0.06, right=0.98,
        bottom=0.17, top=0.9,
        wspace=0.04,
        hspace=0.4
    )

    corners_true1, centroids_true1, dims_true1 = get_cuboid_corners(x_true[0][None, ...])
    centroids_true1 = centroids_true1[[1, 0, 2]]
    corners_true1 = corners_true1[..., [1, 0, 2]]

    corners_true2, centroids_true2, dims_true2 = get_cuboid_corners(x_true[1][None, ...])
    centroids_true2 = centroids_true2[[1, 0, 2]]
    corners_true2 = corners_true2[..., [1, 0, 2]]

    corners_init, centroids_init, dims_init = get_cuboid_corners(x_init[0][None, ...])
    centroids_init = centroids_init[[1, 0, 2]]
    corners_init = corners_init[..., [1, 0, 2]]

    corners_inv1, centroids_inv1, dims_inv1 = get_cuboid_corners(x_inv[0][None, ...])
    centroids_inv1 = centroids_inv1[[1, 0, 2]]
    corners_inv1 = corners_inv1[..., [1, 0, 2]]

    corners_inv2, centroids_inv2, dims_inv2 = get_cuboid_corners(x_inv[1][None, ...])
    centroids_inv2 = centroids_inv2[[1, 0, 2]]
    corners_inv2 = corners_inv2[..., [1, 0, 2]]

    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]

    for idx_comp in range(3):
        pc1 = ax1[idx_comp].pcolormesh(x1, x2, Y_true[:, :, idx_comp].T, norm=norm, cmap=cmap)
        pc2 = ax2[idx_comp].pcolormesh(x1, x2, Y_inv[:, :, idx_comp].T, norm=norm, cmap=cmap)

        ax1[idx_comp].plot(*centroids_true1[:2], '^', mec='w', color=color_true, ms=13, mew=1, zorder=150)
        ax1[idx_comp].plot(*centroids_true2[:2],'^', mec='w', color=color_true, ms=13, mew=1, zorder=150)
        ax1[idx_comp].plot(*centroids_inv1[:2], '*', mec='w', color=color_inv, ms=12, mew=0.5, zorder=200)
        ax1[idx_comp].plot(*centroids_inv2[:2], '*', mec='w', color=color_inv, ms=12, mew=0.5, zorder=200)

        ax2[idx_comp].plot(*centroids_true1[:2],'^', mec='w', color=color_true, ms=13, mew=1, zorder=150)
        ax2[idx_comp].plot(*centroids_true2[:2],'^', mec='w', color=color_true, ms=13, mew=1, zorder=150)
        ax2[idx_comp].plot(*centroids_inv1[:2], '*', mec='w', color=color_inv, ms=12, mew=0.5, zorder=200)
        ax2[idx_comp].plot(*centroids_inv2[:2], '*', mec='w', color=color_inv, ms=12, mew=0.5, zorder=200)

        for e in edges:
            ax1[idx_comp].plot(corners_true1[e][:, 0], corners_true1[e][:, 1],
                               color=color_true, lw=2.5, linestyle=line_style_true)
            ax1[idx_comp].plot(corners_inv1[e][:, 0], corners_inv1[e][:, 1],
                               color=color_inv, lw=2, linestyle=line_style_inv, zorder=100)

            ax2[idx_comp].plot(corners_true1[e][:, 0], corners_true1[e][:, 1],
                               color=color_true, lw=2.5, linestyle=line_style_true)
            ax2[idx_comp].plot(corners_inv1[e][:, 0], corners_inv1[e][:, 1],
                               color=color_inv, lw=2, linestyle=line_style_inv, zorder=100)

            ax1[idx_comp].plot(corners_true2[e][:, 0], corners_true2[e][:, 1],
                               color=color_true, lw=2.5, linestyle=line_style_true)
            ax1[idx_comp].plot(corners_inv2[e][:, 0], corners_inv2[e][:, 1],
                               color=color_inv, lw=2, linestyle=line_style_inv, zorder=100)

            ax2[idx_comp].plot(corners_true2[e][:, 0], corners_true2[e][:, 1],
                               color=color_true, lw=2.5, linestyle=line_style_true)
            ax2[idx_comp].plot(corners_inv2[e][:, 0], corners_inv2[e][:, 1],
                               color=color_inv, lw=2, linestyle=line_style_inv, zorder=100)


        ax1[idx_comp].set_xlabel(r'$x_2$ (East)')
        ax1[idx_comp].set_ylabel(r'$x_1$ (North)' if idx_comp == 0 else None)
        ax2[idx_comp].set_xlabel(r'$x_2$ (East)')
        ax2[idx_comp].set_ylabel(r'$x_1$ (North)' if idx_comp == 0 else None)

        ax1[idx_comp].set_aspect("equal", adjustable="box")
        ax2[idx_comp].set_aspect("equal", adjustable="box")

        ax1[idx_comp].set_yticks([-1, -0.5, 0, 0.5, 1])
        ax2[idx_comp].set_yticks([-1, -0.5, 0, 0.5, 1])
        ax1[idx_comp].set_xticks([-1, -0.5, 0, 0.5, 1])
        ax2[idx_comp].set_xticks([-1, -0.5, 0, 0.5, 1])

        ax1[idx_comp].set_xlim([-1, 1])
        ax2[idx_comp].set_xlim([-1, 1])
        ax1[idx_comp].set_ylim([-1, 1])
        ax2[idx_comp].set_ylim([-1, 1])

    custom_lines = [Line2D([0], [0], marker='*', markeredgecolor='w', markerfacecolor=color_inv,
                           markersize=14, color=color_inv, linestyle=line_style_inv),
                    Line2D([0], [0], marker='^', markeredgecolor='w', markerfacecolor=color_true,
                           markersize=8, color=color_true, linestyle=line_style_true)]
    ax1[1].legend(custom_lines, ['Inverted', 'True'])
    ax2[1].legend(custom_lines, ['Inverted', 'True'])

    cax = fig.add_axes([0.22, 0.07, 0.6, 0.02])
    cbar = fig.colorbar(pc2, cax=cax, orientation='horizontal')
    cbar.set_label(label='Displacement (-)', fontsize=12)
    cbar.outline.set_edgecolor('#f9f2d7')

    ax1[0].set_title("\nEast")
    ax1[1].set_title("a) Observed\nNorth")
    ax1[2].set_title("\nUp")
    ax2[0].set_title("\nEast")
    ax2[1].set_title("b) Inverted\nNorth")
    ax2[2].set_title("\nUp")

    return fig



def invert_multisource(
        model_path: str | os.PathLike,
                       ):


    X_true, Y_true, Y_true_sep = build_multisources(model_path=model_path, n_sources=2, debug=True)
    fwd_model = multisource_fwd_model(model_path=model_path, n_sources=X_true.shape[0], verbose_trainable=False)

    I = tf.eye(12, dtype=tf.float32)[None, :, :]
    I = tf.tile(I, [X_true.shape[0], 1, 1])[None, ...]
    Y0 = tf.convert_to_tensor(Y_true[None, ...], tf.float32)

    n_steps_1, n_steps_2 = 500, 1500
    ds = tf.data.Dataset.from_tensors((I, Y0)).repeat(n_steps_1 + n_steps_2)

    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[n_steps_1],  # after this many updates, switch LR
        values=[1e-2, 1e-3],
    )

    inv_model = multisource_inversion_model(model_path=model_path, n_sources=X_true.shape[0], verbose_trainable=False)

    # for reproducibility
    init_weights = np.array([
        [-0.13074541,  0.73863935,  0.25406748, 0.04967972, 0.09128058, 0.09574779,
         -0.41960865, -0.13699341, -0.20226575, 0.5167792, 0.61557114, 0.34389165],
     [-0.7992966, 0.11263609, 0.48579448, 0.06207821, 0.1, 0.07772668,
      -0.23981695, 0.26809952, -0.56762755, 0.28384823, -0.32789043, 0.60027283]
    ])[:, None, :]

    inv_model.layers[1].set_weights([init_weights])

    X_init = np.squeeze(inv_model.layers[1].get_weights()[0].copy())  # (2, 12)
    # one compiled model
    inv_model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    )
    inv_model.fit(ds, epochs=1, steps_per_epoch=n_steps_1 + n_steps_2, verbose=0)

    # read out inverted parameters
    X_inv = np.squeeze(inv_model.layers[1].get_weights()[0]) # (2, 12)
    print(X_init)

    # compute forward prediction
    Y_pred_inverted = inv_model.predict(I, verbose=0)[0]

    # ---------------------------------------------------------------
    # PLOTTING
    # ---------------------------------------------------------------
    fig3d = plot_3d_cuboids(x_init=X_init, x_true=X_true, x_inv=X_inv)
    savepath = f'figs/multi_source/multisource_figure_00_3DPlot.png'
    fig3d.savefig(savepath, format='png', dpi=300, bbox_inches='tight')

    fig_surf = plot_surfaces(x_init=X_init, x_true=X_true, x_inv=X_inv, Y_true=Y_true, Y_inv=Y_pred_inverted)
    savepath = f'figs/multi_source/multisource_figure_00_surf.png'
    fig_surf.savefig(savepath, format='png', dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    m2 = "./checkpoints/LARGE_v2i_max_entropy_C2DT_2025_12_19_1547_ckpt.keras"

    invert_multisource(model_path=m2)