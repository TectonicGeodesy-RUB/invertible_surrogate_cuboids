# --------------------------------------------------------------
# Created by Kaan Cökerim¹ on 26. January 2026
#
# Scripts for recreating the inversion plots from the paper
#
# ¹Tectonic Geodesy, Ruhr University Bochum, Germany
# Email: kaan.coekerim@rub.de
# --------------------------------------------------------------
import os
import warnings
import numpy as np
import matplotlib as mpl
from matplotlib import gridspec, colors
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import seaborn as sns
import cmcrameri.cm as cm

color_true = '0.01'
color_inv = "darkorange"
color_init = '0.55'

line_style_true = 'solid'
line_style_inv = 'solid'
line_style_init = 'dashed'


def stylize_line(line, halo="black", halo_lw=3.5):
    line.set_path_effects([pe.Stroke(linewidth=halo_lw, foreground=halo), pe.Normal()])


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


def get_strain_tensor(x:np.ndarray):
    e_tensor = np.array([
        [x[0, 6], x[0, 7], x[0, 8]],
        [np.nan, x[0, 9], x[0, 10]],
        [np.nan, np.nan, x[0, 11]]
    ])

    return e_tensor

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
        np.sin(dip),
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


def get_cuboid_param_txt(x: np.ndarray):
    return (f"  x_cen: {x[0, 0]:7.4f}\n"
            f"  y_cen: {x[0, 1]:7.4f}\n"
            f"  z_cen: {x[0, 2]:7.4f}\n"
            f"Length: {x[0, 3]:7.4f}\n"
            f"  Width: {x[0, 4]:7.4f}\n"
            f"  Thick: {x[0, 5]:7.4f}")



def plot_surface_displacement(y_true: np.ndarray, y_inv: np.ndarray, x_true, x_inverted, external_keys=None):
    x = np.linspace(-1, 1, y_true.shape[1], endpoint=True)
    y = np.linspace(-1, 1, y_true.shape[1], endpoint=True)
    x1, x2 = np.meshgrid(x, y)

    # reorder ys  from NED -> ENU
    y_true = y_true[..., [1, 0, 2]]
    y_inv = y_inv[..., [1, 0, 2]]

    if len(external_keys) != 4:
        y_true[..., 2] *= -1
        y_inv[..., 2] *= -1

    norm = colors.CenteredNorm(vcenter=0, halfrange=np.abs(y_true).max())
    cmap = cm.vik

    fig, (ax1, ax2) = plt.subplots(figsize=(12, 9), nrows=2, ncols=3, constrained_layout=False)

    fig.subplots_adjust(
        left=0.06, right=0.98,
        bottom=0.17, top=0.9,
        wspace=0.04,
        hspace=0.4
    )

    for idx_comp in range(3):
        pc1 = ax1[idx_comp].pcolormesh(x1, x2, y_true[:, :, idx_comp].T, norm=norm, cmap=cmap)
        pc2 = ax2[idx_comp].pcolormesh(x1, x2, y_inv[:, :, idx_comp].T, norm=norm, cmap=cmap)
        ax2[idx_comp].set_xlabel(r'$x_1$ (North)')

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

    corners_inverted, c_inverted, dims_inverted = get_cuboid_corners(x_inverted)
    c_inverted = c_inverted[[1, 0, 2]]
    corners_inverted = corners_inverted[..., [1, 0, 2]]

    if external_keys is not None:
        if len(external_keys) == 8:
            fault_params = {external_keys[i]: x_true[0, i] for i in range(8)}
            fault_corners, fault_centroid = get_fault_geom(fault_params)
            fault_centroid = fault_centroid[[1, 0, 2]]
            fault_corners = fault_corners[..., [1, 0, 2]]

            for ax in [ax1, ax2]:
                for idx_comp in range(3):
                    ax[idx_comp].plot(*fault_centroid[:2], '^', mec='w', color=color_true, ms=13, mew=1,
                                      label='True', zorder=100)
                    ax[idx_comp].plot(*c_inverted[:2], '*', mec='w', color=color_inv, ms=12, mew=0.5,
                                      label='Inverted', zorder=150)

                    # plot fault
                    xs = fault_corners[:, 0]
                    ys = fault_corners[:, 1]
                    xs_closed = np.r_[xs, xs[0]]
                    ys_closed = np.r_[ys, ys[0]]
                    ax[idx_comp].plot(xs_closed, ys_closed, ls=line_style_true, color=color_true, lw=2.5)

                    # plot the cuboid
                    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
                    for e in edges:
                        ax[idx_comp].plot(corners_inverted[e][:, 0], corners_inverted[e][:, 1],
                                          color=color_inv, lw=2, linestyle=line_style_inv, zorder=50)

        elif len(external_keys) == 4:
            mogi_centroids = x_true[0, :3][[1, 0, 2]]
            for ax in [ax1, ax2]:
                for idx_comp in range(3):
                    ax[idx_comp].plot(*mogi_centroids[:2], '^', mec='w', color=color_true, ms=13, mew=1,
                                      label='True', zorder=100)
                    ax[idx_comp].plot(*c_inverted[:2], '*', mec='w', color=color_inv, ms=12, mew=0.5,
                                      label='Inverted', zorder=150)

                    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
                    for e in edges:
                        ax[idx_comp].plot(corners_inverted[e][:, 0], corners_inverted[e][:, 1],
                                          color=color_inv, lw=2, linestyle=line_style_inv, zorder=50)


    else:
        corners_true, c_true, dims_inverted = get_cuboid_corners(x_true)
        c_true = c_true[[1, 0, 2]]
        corners_true = corners_true[..., [1, 0, 2]]

        for ax in [ax1, ax2]:
            for idx_comp in range(3):
                ax[idx_comp].plot(*c_true[:2], '^', mec='w', color=color_true, ms=13, mew=1,
                                  label='True', zorder=100)
                ax[idx_comp].plot(*c_inverted[:2], '*', mec='w', color=color_inv, ms=12, mew=0.5,
                                  label='Inverted', zorder=150)
                # plot the cuboid
                edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
                for e in edges:
                    ax[idx_comp].plot(corners_inverted[e][:, 0], corners_inverted[e][:, 1],
                                      color=color_inv, lw=2, linestyle=line_style_inv, zorder=51)

                    ax[idx_comp].plot(corners_true[e][:, 0], corners_true[e][:, 1],
                                      color=color_true, lw=2.5, linestyle=line_style_true, zorder=50)

    custom_lines = [Line2D([0], [0], marker='*', markeredgecolor='w', markerfacecolor=color_inv,
                           markersize=14, color=color_inv, linestyle=line_style_inv),
                    Line2D([0], [0], marker='^', markeredgecolor='w', markerfacecolor=color_true,
                           markersize=8, color=color_true, linestyle=line_style_true)]
    ax1[1].legend(custom_lines, ['Inverted', 'True'])
    ax2[1].legend(custom_lines, ['Inverted', 'True'])

    cax = fig.add_axes([0.22, 0.07, 0.6, 0.02])
    cbar = fig.colorbar(pc2, cax=cax, orientation='horizontal')
    cbar_ticks = cbar.ax.get_xticks()
    if cbar_ticks[-1] <= 1e-3:
        cbar.ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3), useMathText=True)

    cbar.set_label(label='Displacement (-)', fontsize=12)
    cbar.outline.set_edgecolor('#f9f2d7')

    ax1[0].set_title("\nEast")
    ax1[1].set_title("a) Observed\nNorth")
    ax1[2].set_title("\nUp")
    ax2[0].set_title("\nEast")
    ax2[1].set_title("b) Inverted\nNorth")
    ax2[2].set_title("\nUp")

    return fig


def plot_cuboids(
        x_init: np.ndarray, x_inverted: np.ndarray, x_true: np.ndarray| None,
        external_keys: list[str], fig_title: str | None = None,
                 ):
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

    fig = plt.figure(figsize=(8, 10))

    gs = gridspec.GridSpec(3, 3,
                           height_ratios=[0.2, 0.7, 0.1],
                           wspace=0.05, hspace=0.18)
    ax_hms = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_params = [fig.add_subplot(gs[2, i]) for i in range(3)]

    e_inverted = get_strain_tensor(x_inverted)
    e_init = get_strain_tensor(x_init)

    sns.heatmap(e_init, ax=ax_hms[0], cmap=cm.vik, vmin=-1, vmax=1, cbar=False, square=True,
                xticklabels=False, yticklabels=False, linewidths=.8, annot=True, fmt=".2f",
                cbar_kws={"shrink": 0.8, "label": r"$\epsilon_{ij}$ Initial (-)"})
    sns.heatmap(e_inverted, ax=ax_hms[1], cmap=cm.vik, vmin=-1, vmax=1, cbar=False, square=True,
                xticklabels=False, yticklabels=False, linewidths=.8, annot=True, fmt=".2f",
                cbar_kws={"shrink": 0.8, "label": r"$\epsilon_{ij}$ Inverted (-)"})
    ax_hms[0].set_title('Initial')
    ax_hms[1].set_title('a) Eigenstrain Tensor\n\n Inverted')

    disp_txt_init = get_cuboid_param_txt(x_init)
    ax_params[0].text(0.2, -0.2, disp_txt_init)
    ax_params[0].set_title('Initial\n Cuboid Parameters')
    ax_params[0].axis('off')

    disp_txt_inverted = get_cuboid_param_txt(x_inverted)
    ax_params[1].text(0.2, -0.2, disp_txt_inverted)
    ax_params[1].set_title('Inverted\n Cuboid Parameters')
    ax_params[1].axis('off')

    # --- shared colorbar for all strain heatmaps ---
    vmin, vmax = -1, 1
    cmap = cm.vik

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    if x_true is not None:
        if x_true.size == 12:
            e_true = get_strain_tensor(x_true)
            sns.heatmap(e_true, ax=ax_hms[2], cmap=cm.vik, vmin=-1, vmax=1, cbar=False, square=True,
                        xticklabels=False, yticklabels=False, linewidths=.8, annot=True, fmt=".2f",
                        cbar_kws={"shrink": 0.8, "label": r"$\epsilon_{ij}$ True (-)"})
            ax_hms[2].set_title('True')

            disp_txt_true = get_cuboid_param_txt(x_true)
            ax_params[2].text(0.2, -0.2, disp_txt_true)
            ax_params[2].set_title('True\n Cuboid Parameters')
            ax_params[2].axis('off')

            bboxes = [ax.get_position() for ax in ax_hms]
            left = min(bb.x0 for bb in bboxes)
            right = max(bb.x1 for bb in bboxes)
            bottom = min(bb.y0 for bb in bboxes)
            top = max(bb.y1 for bb in bboxes)

            pad = 0.012
            cbar_w = 0.010
            cax = fig.add_axes([right + pad, bottom, cbar_w, top - bottom])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            cbar.set_label(r"$\epsilon_{ij}$ (-)", rotation=90, labelpad=10)
            cbar.outline.set_edgecolor('#f9f2d7')

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

            bboxes = [ax.get_position() for ax in ax_hms[:2]]
            left = min(bb.x0 for bb in bboxes)
            right = max(bb.x1 for bb in bboxes)
            bottom = min(bb.y0 for bb in bboxes)
            top = max(bb.y1 for bb in bboxes)

            pad = 0.012
            cbar_w = 0.010
            cax = fig.add_axes([right + pad, bottom, cbar_w, top - bottom])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            cbar.set_label(r"$\epsilon_{ij}$ (-)", rotation=90, labelpad=10)
            cbar.outline.set_edgecolor('#f9f2d7')

        elif x_true.size == 4:
            fault_params = {external_keys[i]: x_true[0, i] for i in range(3)}
            ax_hms[2].axis('off')

            disp_txt = (f"c1: {fault_params['c1']:7.4f}\n"
                        f"c2: {fault_params['c2']:7.4f}\n"
                        f"c2: {fault_params['c3']:7.4f}")
            ax_params[2].text(0.2, -0.3, disp_txt)
            ax_params[2].set_title('True\n Mogi Parameters')
            ax_params[2].axis('off')

            bboxes = [ax.get_position() for ax in ax_hms[:2]]
            left = min(bb.x0 for bb in bboxes)
            right = max(bb.x1 for bb in bboxes)
            bottom = min(bb.y0 for bb in bboxes)
            top = max(bb.y1 for bb in bboxes)

            pad = 0.012
            cbar_w = 0.010
            cax = fig.add_axes([right + pad, bottom, cbar_w, top - bottom])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            cbar.set_label(r"$\epsilon_{ij}$ (-)", rotation=90, labelpad=10)
            cbar.outline.set_edgecolor('#f9f2d7')

    ax = fig.add_subplot(gs[1, :], projection="3d")

    #----------------------------
    # prettier 3D axes
    ax.view_init(elev=22, azim=-55)
    ax.grid(True)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((0.85, 0.85, 0.85, 1))
    ax.xaxis.line.set_color("0.6")
    ax.yaxis.line.set_color("0.6")
    ax.zaxis.line.set_color("0.6")
    ax.tick_params(colors="0.35", labelsize=9)
    #----------------------------

    ax.plot(*c_init, 'o', color=color_init)
    for e in edges:
        ax.plot(corners_init[e][:, 0], corners_init[e][:, 1], corners_init[e][:, 2],
                color=color_init, ls=line_style_init)

    ax.plot(*c_inverted, '*', color=color_inv)
    for e in edges:
        ax.plot(corners_inverted[e][:, 0], corners_inverted[e][:, 1], corners_inverted[e][:, 2],
                color=color_inv, zorder=100, ls=line_style_inv)

    if x_true is not None:
        if x_true.size == 12:
            corners_true, c_true, dims_true = get_cuboid_corners(x_true)
            if (corners_true[:, 2] < 0).any():
                warnings.warn('TRUE CUBOID CORNERS STICKING OUT', UserWarning)
            ax.plot(*c_true, '^', color=color_true)
            for e in edges:
                ax.plot(corners_true[e][:, 0], corners_true[e][:, 1], corners_true[e][:, 2],
                        color=color_true, linestyle=line_style_true, lw=2)

            ax.plot([c_true[0], c_inverted[0]], [c_true[1], c_inverted[1]], [c_true[2], c_inverted[2]], 'k:')

        elif x_true.size == 8:
            fault_corners, fault_centroid = get_fault_geom(fault_params)
            ax.plot(*fault_centroid, '^', color=color_true)
            poly = Poly3DCollection([fault_corners], edgecolor=color_true, alpha=0., linewidth=2)
            ax.add_collection3d(poly)
            ax.plot([fault_centroid[0], c_inverted[0]],
                    [fault_centroid[1], c_inverted[1]],
                    [fault_centroid[2], c_inverted[2]], 'k--')

        elif x_true.size == 4:
            mogi_centroid = x_true[0, :3]
            ax.plot(*mogi_centroid, '^', color=color_true)

    custom_lines = [Line2D([0], [0], color=color_init, lw=2, linestyle=line_style_init),
                    Line2D([0], [0], color=color_inv, lw=2, linestyle=line_style_inv),
                    Line2D([0], [0], color=color_true, lw=2, linestyle=line_style_true)
                    if x_true is not None else None]

    ax.legend(custom_lines, ['Initial', 'Inverted', 'True' if x_true is not None else None], loc='center left')

    ax.set_title('b) Cuboid Geometry', pad=0.0)
    ax.set_xlabel('x1 (North)')
    ax.set_ylabel('x2 (East)')
    ax.set_zlabel('x3 (Depth)')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([1, 0])
    ax.set_aspect('equal')

    if fig_title is not None and isinstance(fig_title, str):
        fig.suptitle(fig_title)

    return fig

def plot_main(res_path: str):
    with np.load(res_path) as d:
        y_true = d["y_true"]  # (N,64,64,3)
        y_inv = d["y_inv"]  # (N,64,64,3)
        x_true = d["x_true"]  # (N,1,D)  (D=8 for faults, 12 for cuboids)
        x_inv = d["x_inverted"]  # (N,1,12)
        x_init = d["x_init"]
        external_keys = d["external_keys"].tolist() if d["external_keys"].size else None

    source_type = os.path.split(res_path)[1].split('_')[0].lower()

    if source_type == 'debug_cuboids':
        source_type = 'cuboids_debug'
    elif source_type == 'random_cuboids':
        source_type = 'cuboids'

    n_samples = y_true.shape[0]
    print(external_keys)
    for sample in range(3, 5):
        fig_surf = plot_surface_displacement(y_true=y_true[sample], y_inv=y_inv[sample],
                                             x_true=x_true[sample], x_inverted=x_inv[sample],
                                             external_keys=external_keys if external_keys is not None else None)
        savepath = f'figs/{source_type}/{source_type}_figure_{sample:02d}_surf.png'
        os.makedirs(os.path.split(savepath)[0], exist_ok=True)
        fig_surf.savefig(savepath, format='png', dpi=300, bbox_inches='tight')

        fig_3d = plot_cuboids(x_init=x_init[sample], x_inverted=x_inv[sample], x_true=x_true[sample],
                              external_keys=external_keys)
        savepath = f'figs/{source_type}/{source_type}_figure_{sample:02d}_3DPlot.png'
        os.makedirs(os.path.split(savepath)[0], exist_ok=True)
        fig_3d.savefig(savepath, format='png', dpi=300, bbox_inches='tight')

        plt.show()


if __name__ == '__main__':

    res_path = "./inv_outs/Debug_Cuboids_inv_outputs.npz"
    plot_main(res_path=res_path)

    res_path = "./inv_outs/Random_Cuboids_inv_outputs.npz"
    plot_main(res_path=res_path)

    res_path = "./inv_outs/Faults_inv_outputs.npz"
    plot_main(res_path=res_path)

    res_path = "./inv_outs/Mogi_inv_outputs.npz"
    plot_main(res_path=res_path)