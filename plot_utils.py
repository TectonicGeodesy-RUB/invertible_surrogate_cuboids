# --------------------------------------------------------------
# Created by Kaan Cökerim¹ on 03. December 2025
#
# Scripts used for plotting of the training model
#
# ¹Tectonic Geodesy, Ruhr University Bochum, Germany
# Email: kaan.coekerim@rub.de
# --------------------------------------------------------------
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors, gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
from cmcrameri import cm


# ------- EVALUATE TRAINING
def plot_history(history_path: str, show: bool = False) -> plt.Figure:
    df = pd.read_csv(history_path)

    fig, ax = plt.subplots()
    df[['loss', 'val_loss']].plot(ax=ax, logy=True, legend=True)
    if 'learning_rate' in df.columns:
        df['learning_rate'].plot(ax=ax, logy=True, secondary_y=True, legend=True)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.right_ax.set_ylabel('Learning Rate')
    if show:
        plt.show()

    return fig


def logvar_plot(logvars: np.ndarray, show: bool = False) -> plt.Figure:
    fig, ax = plt.subplots()
    input_vars = ['c₁', 'c₂', 'c₃',
                  'L', 'W', 'T',
                  'ε₁₁', 'ε₁₂', 'ε₁₃', 'ε₂₂', 'ε₂₃', 'ε₃₃']
    for j in range(logvars.shape[1])[0:]:
        ax.plot(np.sort(logvars[:, j]), label=input_vars[j])
    ax.legend()
    if show:
        plt.show()
    return fig


def surf_plots(x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, show: bool = False) -> plt.Figure:

    c1, c2, c3 = x[:3]
    L, W, T = x[3:6]
    q1 = c1 - (L / 2)
    q2 = c2

    y_resd = y_true - y_pred

    x1v = np.linspace(-1, 1, 64, endpoint=True)
    x2v = np.linspace(-1, 1, 64, endpoint=True)
    x1, x2 = np.meshgrid(x1v, x2v)

    cmap = cm.vik
    norm = colors.CenteredNorm(vcenter=0, halfrange=np.abs(y_true).max())

    comps = ['North', 'East', 'Down']
    subs = ['True', 'Pred', 'Resd']

    fig, ax = plt.subplots(
        figsize=(10, 10), nrows=4, ncols=3, subplot_kw={'aspect': 'equal'},
    )

    for i_comp in range(3):
        pc = ax[0, i_comp].pcolormesh(x1, x2, y_true[:, :, i_comp], cmap=cmap, norm=norm)
        ax[1, i_comp].pcolormesh(x1, x2, y_pred[:, :, i_comp], cmap=cmap, norm=norm)
        ax[2, i_comp].pcolormesh(x1, x2, y_resd[:, :, i_comp], cmap=cmap, norm=norm)

        mae = np.abs(y_resd[:, :, i_comp].ravel()).mean()
        r2 = 1 - (np.sum(y_resd[:, :, i_comp].ravel() ** 2) /
                  np.sum((y_true[:, :, i_comp].ravel() - y_true[:, :, i_comp].ravel().mean()) ** 2))
        metrics = f'{mae=:.3f}\n{r2=:.3f}'
        ax[3, i_comp].plot(y_true[:, :, i_comp].ravel(), y_pred[:, :, i_comp].ravel(), 'bx', label=metrics)
        ax[3, i_comp].axline((0, 0), slope=1, color='k', linestyle='--')
        ax[3, i_comp].grid()
        ax[3, i_comp].legend()
        ax[3, i_comp].set_xlabel('Norm. True Displ.')

        for i_subplot in range(3):
            ax[i_comp, i_subplot].plot(c1, c2, '^', color='cyan', mec='k', zorder=100)
            ax[i_comp, i_subplot].plot([q1, q1 + L], [q2 + T / 2, q2 + T / 2], ':', color='gray')
            ax[i_comp, i_subplot].plot([q1, q1 + L], [q2 - T / 2, q2 - T / 2], ':', color='gray')
            ax[i_comp, i_subplot].plot([q1, q1], [q2 - T / 2, q2 + T / 2], '-', color='gray')
            ax[i_comp, i_subplot].plot([q1 + L, q1 + L], [q2 - T / 2, q2 + T / 2], '-', color='gray')
            ax[i_comp, i_subplot].set_title(f'{comps[i_subplot]} {subs[i_comp]}')

        if i_comp == 0:
            ax[0, i_comp].xaxis.set_ticks([])
            ax[1, i_comp].xaxis.set_ticks([])
            ax[3, i_comp].set_ylabel('Norm. Pred Displ.')
        else:
            ax[0, i_comp].xaxis.set_ticks([])
            ax[1, i_comp].xaxis.set_ticks([])
            ax[0, i_comp].yaxis.set_ticks([])
            ax[1, i_comp].yaxis.set_ticks([])
            ax[2, i_comp].yaxis.set_ticks([])

    cbar = fig.colorbar(pc, ax=ax[:3, 2], aspect=40, label='Norm. Displ.')

    fig.suptitle(f'c₁={c1:.3f} c₂={c2:.3f} c₃={c3:.3f}, L={L:.3f}, T={T:.3f}, W={W:.3f}\n'
                 f'ε₁₁={x[8]:.2e}, ε₁₂={x[9]:.2e}, ε₁₃={x[10]:.2e},'
                 f'ε₂₂={x[11]:.2e}, ε₂₃={x[12]:.2e}, ε₃₃={x[13]:.2e}')

    if show:
        plt.show()

    return fig

# ------- EVALUATE INVERSION

def get_strain_tensor(x:np.ndarray):
    e_tensor = np.array([
        [x[0, 6], x[0, 7], x[0, 8]],
        [np.nan, x[0, 9], x[0, 10]],
        [np.nan, np.nan, x[0, 11]]
    ])
    return e_tensor


def get_cuboid_corners(x: np.ndarray):
    L, W, T = x[0, 3:6]
    c1, c2, c3 = x[0, :3]
    q1 = c1 - (L / 2)
    q2 = c2
    q3 = c3 - (W / 2)

    corners = np.array([
        [q1, q2 + T / 2, q3],
        [q1, q2 - T / 2, q3],
        [q1 + L, q2 - T / 2, q3],
        [q1 + L, q2 + T / 2, q3],

        [q1, q2 + T / 2, q3 + W],
        [q1, q2 - T / 2, q3 + W],
        [q1 + L, q2 - T / 2, q3 + W],
        [q1 + L, q2 + T / 2, q3 + W],
    ])

    c_arr = np.array([c1, c2, c3])
    dims_arr = np.array([L, W, T])

    return corners, c_arr, dims_arr


def get_cuboid_param_txt(x: np.ndarray):
    return (f"  x_cen: {x[0, 0]:7.4f}\n"
            f"  y_cen: {x[0, 1]:7.4f}\n"
            f"  z_cen: {x[0, 2]:7.4f}\n"
            f"Length: {x[0, 3]:7.4f}\n"
            f"  Width: {x[0, 4]:7.4f}\n"
            f"  Thick: {x[0, 5]:7.4f}")


def get_fault_geom(fault_parameters: dict):
    centroid = [fault_parameters['y_cen'], fault_parameters['x_cen'], fault_parameters['z_cen']]
    strike_deg = fault_parameters['strike']
    dip_deg = fault_parameters['dip']
    length = fault_parameters['fault_length']
    width = fault_parameters['fault_width']

    centroid = np.asarray(centroid, dtype=float)
    strike = np.deg2rad(strike_deg)
    dip = np.deg2rad(dip_deg)

    # Unit vector along strike (horizontal)
    # strike 0° -> pointing north (1,0,0)
    strike_vec = np.array([np.cos(strike), np.sin(strike), 0.0])

    # Dip direction azimuth = strike + 90° (right-hand rule)
    dip_dir = strike + np.pi / 2.0

    # Unit vector down-dip: horizontal projection * cos(dip), vertical component = sin(dip)
    dip_vec = np.array([
        np.sin(dip_dir) * np.cos(dip),
        np.cos(dip_dir) * np.cos(dip),
        np.sin(dip),  # positive down
    ])

    # Half-length and half-width vectors
    half_len = 0.5 * length * strike_vec
    half_wid = 0.5 * width * dip_vec

    # Four corners around centroid (order them so polygon is not twisted)
    c1 = centroid - half_len - half_wid
    c2 = centroid + half_len - half_wid
    c3 = centroid + half_len + half_wid
    c4 = centroid - half_len + half_wid

    corners = np.vstack([c1, c2, c3, c4])
    if (corners[:, 2] < 0).any():
        print('FAULT CORNERS STICKING OUT')

    return corners, centroid


def plot_surface_displacement(y_true: np.ndarray, y_inv: np.ndarray, x_true, x_inverted, external_keys=None):

    x = np.linspace(-1, 1, y_true.shape[1], endpoint=True)
    y = np.linspace(-1, 1, y_true.shape[0], endpoint=True)
    x1, x2 = np.meshgrid(x, y)

    norm = colors.CenteredNorm(vcenter=0)
    cmap = cm.vik

    fig, (ax1, ax2) = plt.subplots(figsize=(10, 8), nrows=2, ncols=3, subplot_kw={'aspect': 'equal'})
    for idx_comp in range(3):
        pc = ax1[idx_comp].pcolormesh(x1, x2, y_true[:, :, idx_comp], norm=norm, cmap=cmap)
        ax1[idx_comp].set_xlabel('x1 (North)')
        ax1[idx_comp].set_ylabel('x2 (East)')
    fig.colorbar(pc, ax=ax1[:3], orientation='horizontal', shrink=0.8, aspect=50, label='Displacement (-)')

    for idx_comp in range(3):
        pc = ax2[idx_comp].pcolormesh(x1, x2, y_inv[:, :, idx_comp], norm=norm, cmap=cmap)
        ax2[idx_comp].set_xlabel('x1 (North)')
        ax2[idx_comp].set_ylabel('x2 (East)')
    fig.colorbar(pc, ax=ax2[:3], orientation='horizontal', shrink=0.8, aspect=50, label='Displacement (-)')

    corners_inverted, c_inverted, dims_inverted = get_cuboid_corners(x_inverted)
    if external_keys is not None and len(external_keys) == 8:
        fault_params = {external_keys[i]: x_true[0, i] for i in range(8)}
        fault_corners, fault_centroid = get_fault_geom(fault_params)
        for ax in [ax1, ax2]:
            for idx_comp in range(3):
                ax[idx_comp].plot(*fault_centroid[:2], 'g-^', mec='w', ms=10, label='True')
                ax[idx_comp].plot(*c_inverted[:2], 'r-*', mec='w', ms=10, label='Inverted')

                # plot fault
                xs = fault_corners[:, 0]
                ys = fault_corners[:, 1]
                xs_closed = np.r_[xs, xs[0]]
                ys_closed = np.r_[ys, ys[0]]
                ax[idx_comp].plot(xs_closed, ys_closed, 'g-')

                # plot the cuboid
                edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
                for e in edges:
                    ax[idx_comp].plot(corners_inverted[e][:, 0], corners_inverted[e][:, 1], color='red')

    elif external_keys is not None and len(external_keys) == 4:
        mogi_centroids = x_true[0, :3]
        for ax in [ax1, ax2]:
            for idx_comp in range(3):
                ax[idx_comp].plot(*mogi_centroids[:2], 'g-^', mec='w', ms=10, label='True')
                ax[idx_comp].plot(*c_inverted[:2], 'r-*', mec='w', ms=10, label='Inverted')

                edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
                for e in edges:
                    ax[idx_comp].plot(corners_inverted[e][:, 0], corners_inverted[e][:, 1], color='red')

    else:
        corners_true, c_ture, dims_inverted = get_cuboid_corners(x_true)
        for ax in [ax1, ax2]:
            for idx_comp in range(3):
                ax[idx_comp].plot(*c_ture[:2], 'g-^', mec='w', ms=10, label='True')
                ax[idx_comp].plot(*c_inverted[:2], 'r-*', mec='w', ms=10, label='Inverted')
                # plot the cuboid
                edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
                for e in edges:
                    ax[idx_comp].plot(corners_inverted[e][:, 0], corners_inverted[e][:, 1], color='red')
                    ax[idx_comp].plot(corners_true[e][:, 0], corners_true[e][:, 1], color='green')


    ax1[0].set_title("North")
    ax1[1].set_title("a) True\nEast")
    ax1[1].legend()
    ax1[2].set_title("Down")
    ax2[1].set_title("b) Inverted")

    return fig

def plot_cuboids(
        x_init: np.ndarray, x_inverted: np.ndarray, x_true: np.ndarray | None,
        external_keys: list[str], fig_title: str | None = None,
                 ):
    """

    Args:
        x_init:
        x_inverted:
        x_true:
        external_keys:
        fig_title:

    Returns:

    """

    corners_init, c_init, dims_init = get_cuboid_corners(x_init)
    corners_inverted, c_inverted, dims_inverted = get_cuboid_corners(x_inverted)
    if (corners_init[:, 2] < 0).any():
        warnings.warn('INIT CUBOID CORNERS STICKING OUT', UserWarning)
    if (corners_inverted[:, 2] < 0).any():
        warnings.warn('INVERTED CUBOID CORNERS STICKING OUT', UserWarning)

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom square
        [4, 5], [5, 6], [6, 7], [7, 4],  # top square
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
    ]

    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(3, 3, height_ratios=[0.2, 0.7, 0.1])  # 3 rows, 3 columns
    ax_hms = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_params = [fig.add_subplot(gs[2, i]) for i in range(3)]

    e_inverted = get_strain_tensor(x_inverted)
    e_init = get_strain_tensor(x_init)

    sns.heatmap(e_init, ax=ax_hms[0], cmap=cm.vik, vmin=-1, vmax=1, cbar=True, square=True,
                xticklabels=False, yticklabels=False, linewidths=.8, annot=True, fmt=".2f",
                cbar_kws={"shrink": 0.8, "label": r"$\epsilon_{ij}$ Initial (-)"})
    sns.heatmap(e_inverted, ax=ax_hms[1], cmap=cm.vik, vmin=-1, vmax=1, cbar=True, square=True,
                xticklabels=False, yticklabels=False, linewidths=.8, annot=True, fmt=".2f",
                cbar_kws={"shrink": 0.8, "label": r"$\epsilon_{ij}$ Inverted (-)"})
    ax_hms[0].set_title('Initial\nStrain Tensor ' +
                        r"$||\epsilon||$:" + f"{np.linalg.norm(e_init[~np.isnan(e_init)]):.2f}")
    ax_hms[1].set_title('Inverted\nStrain Tensor '
                        + r"$||\epsilon||$:" + f"{np.linalg.norm(e_inverted[~np.isnan(e_inverted)]):.2f}")

    disp_txt_init = get_cuboid_param_txt(x_init)
    ax_params[0].text(0.2, -0.2, disp_txt_init)
    ax_params[0].set_title('Initial\n Cuboid Parameters')
    ax_params[0].axis('off')

    disp_txt_inverted = get_cuboid_param_txt(x_inverted)
    ax_params[1].text(0.2, -0.2, disp_txt_inverted)
    ax_params[1].set_title('Inverted\n Cuboid Parameters')
    ax_params[1].axis('off')

    if x_true is not None:
        if x_true.size == 12:
            e_true = get_strain_tensor(x_true)
            sns.heatmap(e_true, ax=ax_hms[2], cmap=cm.vik, vmin=-1, vmax=1, cbar=True, square=True,
                        xticklabels=False, yticklabels=False, linewidths=.8, annot=True, fmt=".2f",
                        cbar_kws={"shrink": 0.8, "label": r"$\epsilon_{ij}$ True (-)"})
            ax_hms[2].set_title('True\nStrain Tensor '
                                + r"$||\epsilon||$:" + f"{np.linalg.norm(e_true[~np.isnan(e_true)]):.2f}")

            disp_txt_true = get_cuboid_param_txt(x_true)
            ax_params[2].text(0.2, -0.2, disp_txt_true)
            ax_params[2].set_title('True\n Cuboid Parameters')
            ax_params[2].axis('off')

        elif x_true.size == 8:
            fault_params = {external_keys[i]: x_true[0, i] for i in range(8)}
            ax_hms[2].axis('off')

            disp_txt = (f"  x_cen: {fault_params['x_cen']:7.4f}\n"
                        f"  y_cen: {fault_params['y_cen']:7.4f}\n"
                        f"  z_cen: {fault_params['z_cen']:7.4f}\n"
                        f"Length: {fault_params['fault_length']:7.4f}\n"
                        f"  Width: {fault_params['fault_width']:7.4f}\n"
                        f"      Dip: {fault_params['dip']:6.3f}°\n"
                        f" Strike: {fault_params['strike']:7.3f}°")

            ax_params[2].text(0.2, -0.3, disp_txt)
            ax_params[2].set_title('True\n Fault Parameters')
            ax_params[2].axis('off')

        elif x_true.size == 4:
            fault_params = {external_keys[i]: x_true[0, i] for i in range(3)}
            ax_hms[2].axis('off')
            disp_txt = (f"c1: {fault_params['c1']:7.4f}\n"
                        f"c2: {fault_params['c2']:7.4f}\n"
                        f"c2: {fault_params['c3']:7.4f}")
            ax_params[2].text(0.2, -0.3, disp_txt)
            ax_params[2].set_title('True\n Mogi Parameters')
            ax_params[2].axis('off')

    ax = fig.add_subplot(gs[1, :], projection="3d")

    ax.plot(*c_init, 'bo')
    for e in edges:
        ax.plot(corners_init[e][:, 0], corners_init[e][:, 1], corners_init[e][:, 2], color='blue')

    ax.plot(*c_inverted, 'r*')
    for e in edges:
        ax.plot(corners_inverted[e][:, 0], corners_inverted[e][:, 1], corners_inverted[e][:, 2], color='red')

    if x_true is not None:
        if x_true.size == 12:
            corners_true, c_true, dims_true = get_cuboid_corners(x_true)
            if (corners_true[:, 2] < 0).any():
                warnings.warn('TRUE CUBOID CORNERS STICKING OUT', UserWarning)
            ax.plot(*c_true, 'g^')
            for e in edges:
                ax.plot(corners_true[e][:, 0], corners_true[e][:, 1], corners_true[e][:, 2], color='green')
            ax.set_title(f'Distance Centroids\n(true-inverted): '
                         f'{np.linalg.norm(c_true - c_inverted):.4e}')
            ax.plot([c_true[0], c_inverted[0]], [c_true[1], c_inverted[1]], [c_true[2], c_inverted[2]], 'k:')

        elif x_true.size == 8:
            fault_corners, fault_centroid = get_fault_geom(fault_params)
            ax.plot(*fault_centroid, 'g^')
            poly = Poly3DCollection([fault_corners], edgecolor='green', alpha=0., linewidth=2)
            ax.add_collection3d(poly)
            ax.set_title(f'Distance Centroids\n(true-inverted): '
                         f'{np.linalg.norm(fault_centroid - c_inverted):.4e}')
            ax.plot([fault_centroid[0], c_inverted[0]],
                    [fault_centroid[1], c_inverted[1]],
                    [fault_centroid[2], c_inverted[2]], 'k--')

        elif x_true.size == 4:
            mogi_centroid = x_true[0, :3]
            mogi_radius = x_true[0, 3]
            ax.plot(*mogi_centroid, 'g^')
            ax.plot([mogi_centroid[0], mogi_centroid[0] + mogi_radius],
                    [mogi_centroid[1], mogi_centroid[1] + mogi_radius],
                    [mogi_centroid[2], mogi_centroid[2] + mogi_radius], 'g-')
            ax.set_title(f'Distance Centroids\n(true-inverted): '
                         f'{np.linalg.norm(mogi_centroid - c_inverted):.4e}')

    custom_lines = [Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], color='green', lw=2) if x_true is not None else None]

    ax.legend(custom_lines, ['Initial', 'Inverted', 'True' if x_true is not None else None])

    ax.set_xlabel('x1 (North)')
    ax.set_ylabel('x2 (East)')
    ax.set_zlabel('x3 (Down)')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([1, 0])
    ax.set_aspect('equal')

    if fig_title is not None and isinstance(fig_title, str):
        fig.suptitle(fig_title)

    return fig