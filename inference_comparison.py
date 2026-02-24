# --------------------------------------------------------------
# Created by Kaan Cökerim¹ on 26. January 2026
#
# Scripts for recreating the forward inference plots from the paper
#
# ¹Tectonic Geodesy, Ruhr University Bochum, Germany
# Email: kaan.coekerim@rub.de
# --------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from invert_for_cuboids import load_fwd_model
from plot_utils import get_cuboid_corners
from cmcrameri import cm
from synthetic_data_generation.computeDisplacementVerticalShearZone import computeDisplacementVerticalShearZone
from h5_dataloader import scale_Y_maxnorm_per_sample, unscale_Y_maxnorm_per_sample


def plot_comparison(model_path: str):

    model_name = os.path.split(model_path)[-1][:-27]
    remove_variational = False if model_name[-19:] == 'variational_removed' else True
    # Forward/inference model used for MS screening and scale estimation
    fwd_model = load_fwd_model(
        model_path=model_path,
        n_var_layers=13,
    )

    c1, c2, c3 = (0.13, -0.22, 0.57)
    L, W, T = (0.41, 0.37, 0.56)
    espv_ij = np.array([0.0, -0.8, 1, 0.0, 0.4, 0.0])
    espv_ij = espv_ij / np.linalg.norm(espv_ij)
    X = np.array([c1, c2, c3, L, W, T, *espv_ij])

    x = np.linspace(-1, 1, 64, endpoint=True)
    y = np.linspace(-1, 1, 64, endpoint=True)
    x1, x2 = np.meshgrid(x, y)
    x3 = np.zeros(x2.shape)

    Y_surrogate = fwd_model.predict(X[None, :])[0]

    q1 = c1 - (L / 2)
    q2 = c2
    q3 = c3 - (W / 2)
    u1, u2, u3 = computeDisplacementVerticalShearZone(
        x1=x1, x2=x2, x3=x3,
        q1=q1, q2=q2, q3=q3,
        L=L, W=W, T=T,
        epsv11p=espv_ij[0], epsv12p=espv_ij[1], epsv13p=espv_ij[2],
        epsv22p=espv_ij[3], epsv23p=espv_ij[4],
        epsv33p=espv_ij[5],
        G=1, nu=0.25, theta=0,
    )
    Y_analytic = np.stack([u1, u2, u3], axis=-1)

    # reorder ys  from NED -> ENU
    Y_analytic = Y_analytic[..., [1, 0, 2]]
    Y_surrogate = Y_surrogate[..., [1, 0, 2]]

    Y_analytic[..., 2] *= -1
    Y_surrogate[..., 2] *= -1

    # get scaled and unscaled versions
    Y_analytic_scaled, max_disp = scale_Y_maxnorm_per_sample(Y_analytic[None, :])
    Y_analytic_scaled = Y_analytic_scaled[0, :]

    # Y_surrogate_unscaled = unscale_Y_maxnorm_per_sample(Y_surrogate[None, :], max_disps=max_disp)
    # Y_surrogate_unscaled = Y_surrogate_unscaled[0, :]

    norm = colors.CenteredNorm(vcenter=0, halfrange=np.quantile(np.abs(Y_analytic_scaled), 0.9))
    cmap = cm.vik

    line_style_true = 'solid'
    color_true = '0.01'

    fig, (ax1, ax2) = plt.subplots(figsize=(12, 9), nrows=2, ncols=3, constrained_layout=False)

    fig.subplots_adjust(
        left=0.06, right=0.98,
        bottom=0.17, top=0.9,
        wspace=0.04,
        hspace=0.4
    )

    for idx_comp in range(3):
        pc1 = ax1[idx_comp].pcolormesh(x1, x2, Y_analytic_scaled[:, :, idx_comp].T,
                                       norm=norm, cmap=cmap)
        pc2 = ax2[idx_comp].pcolormesh(x1, x2, Y_surrogate[:, :, idx_comp].T,
                                       norm=norm, cmap=cmap)
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

    corners, centroids, dims = get_cuboid_corners(X[None, :])
    centroids = centroids[[1, 0, 2]]
    corners = corners[..., [1, 0, 2]]

    for ax in [ax1, ax2]:
        for idx_comp in range(3):
            ax[idx_comp].plot(*centroids[:2], '^', mec='w', color=color_true, ms=13, mew=1,
                              label='True', zorder=100)

            # plot the cuboid
            edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
            for e in edges:
                ax[idx_comp].plot(corners[e][:, 0], corners[e][:, 1],
                                  color=color_true, lw=2.5, linestyle=line_style_true, zorder=50)

    custom_lines = [Line2D([0], [0], marker='^', markeredgecolor='w', markerfacecolor=color_true,
                           markersize=8, color=color_true, linestyle=line_style_true)]
    ax1[1].legend(custom_lines, ['Cuboid'])
    ax2[1].legend(custom_lines, ['Cuboid'])

    cax = fig.add_axes([0.22, 0.07, 0.6, 0.02])
    cbar = fig.colorbar(pc2, cax=cax, orientation='horizontal')
    cbar.set_label(label='Displacement (-)', fontsize=12)
    cbar.outline.set_edgecolor('#f9f2d7')

    ax1[0].set_title("\nEast")
    ax1[1].set_title("a) Analytical Solution\nNorth")
    ax1[2].set_title("\nUp")
    ax2[0].set_title("\nEast")
    ax2[1].set_title("b) Surrogate Model\nNorth")
    ax2[2].set_title("\nUp")

    savepath = f'figs/surrogate_forward/surrogate_forward.png'
    fig.savefig(savepath, format='png', dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':

    model_path = "./checkpoints/LARGE_v2i_max_entropy_C2DT_2025_12_19_1547_ckpt.keras"
    plot_comparison(model_path=model_path)
