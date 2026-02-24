# --------------------------------------------------------------
# Created by Kaan Cökerim¹ on 27. January 2026
#
# Scripts for recreating the animation sweep of cuboid parameters
# from the paper across the forward inference model
#
# ¹Tectonic Geodesy, Ruhr University Bochum, Germany
# Email: kaan.coekerim@rub.de
# --------------------------------------------------------------
import os
from tqdm import trange
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns
import cmcrameri.cm as cm
from invert_for_cuboids import load_fwd_model
from synthetic_data_generation.computeDisplacementVerticalShearZone import computeDisplacementVerticalShearZone
from h5_dataloader import scale_Y_maxnorm_per_sample
import matplotlib.animation as animation

color_true = '0.01'
color_inv = "darkorange"
color_init = '0.55'

line_style_true = 'solid'
line_style_inv = 'solid'
line_style_init = 'dashed'

def ease(t, kind="cosine"):
    if kind == "linear":
        return t
    elif kind == "cubic":
        return 3*t**2 - 2*t**3
    elif kind == "cosine":
        return 0.5 - 0.5*np.cos(np.pi*t)


def build_parameter_path(base, vary_idx, ranges, n, easing_kind="cosine"):
    X = np.repeat(base[None, :], n, axis=0).astype(np.float32)
    s = ease(np.linspace(0.0, 1.0, n), kind=easing_kind)
    for k, idx in enumerate(vary_idx):
        a, b = ranges[k]
        X[:, idx] = a + s * (b - a)
    return X  # (n, 12)

def plot_displ(
        x1: np.ndarray, x2: np.ndarray, y: np.ndarray,
        cs, qs,
        L, T,
        ax: plt.axis,
        norm=colors.CenteredNorm(vcenter=0),
        cmap=cm.vik
):
    c1, c2 = cs
    q1, q2 = qs

    pc = ax.pcolormesh(x1, x2, y, norm=norm, cmap=cmap)
    c = ax.plot(c1, c2, '^', color='cyan', mec='k', zorder=100)
    l1 = ax.plot([q1, q1 + L], [q2 + T / 2, q2 + T / 2], ':', color='gray')
    l2 = ax.plot([q1, q1 + L], [q2 - T / 2, q2 - T / 2], ':', color='gray')
    l3 = ax.plot([q1, q1], [q2 - T / 2, q2 + T / 2], '-', color='gray')
    l4 = ax.plot([q1 + L, q1 + L], [q2 - T / 2, q2 + T / 2], '-', color='gray')

    return pc, c, l1, l2, l3, l4


def make_animations(
        x_params,
        y1: np.ndarray[float],
        y2: np.ndarray[float] | None,
        norm, cmap,
        out_name: str,
                     ):

    x = np.linspace(-1, 1, y1.shape[1], endpoint=True)
    y = np.linspace(-1, 1, y1.shape[1], endpoint=True)
    x1, x2 = np.meshgrid(x, y)

    pcs = []
    cs = []
    ls = []

    c1 = x_params[0, 0]
    c2 = x_params[0, 1]
    c3 = x_params[0, 2]
    L, W, T = x_params[0, 3:6]
    q1 = c1 - (L / 2)
    q2 = c2

    n_frames = y1.shape[0]

    if y2 is None:
        fig, ax1 = plt.subplots(
            figsize=(11, 8), nrows=1, ncols=4, subplot_kw={'aspect': 'equal'},
        )
    else:
        fig, (ax1, ax2) = plt.subplots(
            figsize=(11, 8), nrows=2, ncols=4, subplot_kw={'aspect': 'equal'}, gridspec_kw={'hspace': 0.05},
        )
        pcs2 = []
        cs2 = []
        ls2 = []

    txt = ax1[1].text(-0.7, 1.6,
                      f"Surrogate Inference Parameter Sweep\n"
                      f"Sample {0:0{len(str(n_frames))}d}\n"
                      f"c₁={c1:.3f} c₂={c2:.3f} c₃={c3:.3f}\n"
                      f"L: {L:.2f}, T: {T:.2f}, W: {W:.2f}",
                      multialignment='center')

    for i_comp in range(3):
        pc, c, l1, l2, l3, l4 = plot_displ(x1=x1, x2=x2, y=y1[0, :, :, i_comp].T,
                                           cs=[c2, c1], qs=[q2, q1],
                                           L=T, T=L,
                                           ax=ax1[i_comp], norm=norm, cmap=cmap)
        if i_comp > 0:
            ax1[i_comp].set_yticks([])
        pcs.append(pc)
        cs.append(c[0])
        ls.append([l1[0], l2[0], l3[0], l4[0]])
        ax1[i_comp].set(xlim=[-1, 1], ylim=[-1, 1])
        ax1[i_comp].set_xlabel('x2 (East)')

    if y2 is not None:
        for i_comp in range(3):
            pc, c, l1, l2, l3, l4 = plot_displ(x1=x1, x2=x2, y=y2[0, :, :, i_comp].T,
                                               cs=[c2, c1], qs=[q2, q1],
                                               L=T, T=L,
                                               ax=ax2[i_comp], norm=norm, cmap=cmap)
            if i_comp > 0:
                ax2[i_comp].set_yticks([])
            pcs2.append(pc)
            cs2.append(c[0])
            ls2.append([l1[0], l2[0], l3[0], l4[0]])
            ax2[i_comp].set(xlim=[-1, 1], ylim=[-1, 1])
            ax2[i_comp].set_xlabel('x2 (East)')
        ax2[3].set_axis_off()

    cax = fig.add_axes([0.15, 0.07, 0.5, 0.02])
    cbar = fig.colorbar(pcs[0], cax=cax, orientation='horizontal')
    cbar_ticks = cbar.ax.get_xticks()
    if cbar_ticks[-1] <= 1e-3:
        cbar.ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3), useMathText=True)
    cbar.set_label(label='Displacement (-)', fontsize=12)
    cbar.outline.set_edgecolor('#f9f2d7')

    ax1[0].set_ylabel('x1 (North)')
    ax2[0].set_ylabel('x1 (North)')

    e_tensor = np.array([
        [X_params[0, 6], X_params[0, 7], X_params[0, 8]],
        [np.nan, X_params[0, 9], X_params[0, 10]],
        [np.nan, np.nan, X_params[0, 11]]
    ])

    sns.heatmap(e_tensor, ax=ax1[3], cmap=cmap, vmin=-1, vmax=1, cbar=True, square=True,
                xticklabels=False, yticklabels=False, linewidths=.8, annot=True, fmt=".2f",
                cbar_kws={"shrink": 0.35, "label": r"$\epsilon_{ij}$ (-)"})
    quadmesh = ax1[3].collections[0]
    annots = [t for t in ax1[3].texts]
    ax1[3].set_title("Eigenstrain Tensor\n")
    ax1[0].set_title("East")
    ax1[1].set_title("a) Analytic Solutions\nNorth")
    ax1[2].set_title("Up")

    if y2 is not None:
        ax2[0].set_title("East")
        ax2[1].set_title('b) Surrogate Inference Predictions\nNorth')
        ax2[2].set_title("Up")

    def animate(frame):
        c1 = X_params[frame, 0]
        c2 = X_params[frame, 1]
        c3 = X_params[frame, 2]
        L, W, T = X_params[frame, 3:6]
        q1 = c1 - (L / 2)
        q2 = c2

        txt.set_text(f"Surrogate Inference Parameter Sweep\n"
                     f"Sample {frame:0{len(str(n_frames))}d}\n"
                     f"c₁={c1:.3f} c₂={c2:.3f} c₃={c3:.3f}\n"
                     f"L: {L:.2f}, T: {T:.2f}, W: {W:.2f}")

        e_tensor = np.array([
            [X_params[frame, 6], X_params[frame, 7], X_params[frame, 8]],
            [np.nan, X_params[frame, 9], X_params[frame, 10]],
            [np.nan, np.nan, X_params[frame, 11]]
        ])

        for i_comp in range(3):
            pcs[i_comp].set_array(y1[frame, :, :, i_comp].T.ravel())
            cs[i_comp].set_ydata([c1])
            cs[i_comp].set_xdata([c2])

            ls[i_comp][0].set_ydata([q1, q1 + L])
            ls[i_comp][0].set_xdata([q2 + T / 2, q2 + T / 2])
            ls[i_comp][1].set_ydata([q1, q1 + L])
            ls[i_comp][1].set_xdata([q2 - T / 2, q2 - T / 2])
            ls[i_comp][2].set_ydata([q1, q1])
            ls[i_comp][2].set_xdata([q2 - T / 2, q2 + T / 2])
            ls[i_comp][3].set_ydata([q1 + L, q1 + L])
            ls[i_comp][3].set_xdata([q2 - T / 2, q2 + T / 2])

        if y2 is not None:
            for i_comp in range(3):
                pcs2[i_comp].set_array(y2[frame, :, :, i_comp].T.ravel())
                cs2[i_comp].set_ydata([c1])
                cs2[i_comp].set_xdata([c2])

                ls2[i_comp][0].set_ydata([q1, q1 + L])
                ls2[i_comp][0].set_xdata([q2 + T / 2, q2 + T / 2])
                ls2[i_comp][1].set_ydata([q1, q1 + L])
                ls2[i_comp][1].set_xdata([q2 - T / 2, q2 - T / 2])
                ls2[i_comp][2].set_ydata([q1, q1])
                ls2[i_comp][2].set_xdata([q2 - T / 2, q2 + T / 2])
                ls2[i_comp][3].set_ydata([q1 + L, q1 + L])
                ls2[i_comp][3].set_xdata([q2 - T / 2, q2 + T / 2])

        quadmesh.set_array(e_tensor)
        quadmesh.set_clim(-1, 1)
        # Update annotation text values
        for a, val in zip(annots, e_tensor.flatten()[~np.isnan(e_tensor.flatten())]):
            a.set_text(f"{val:.2f}")

        if y2 is not None:
            return pcs, pcs2, quadmesh, txt, annots
        else:
            return pcs, quadmesh, txt, annots

    ani = animation.FuncAnimation(fig=fig, func=animate, frames=n_frames, interval=100)
    ani.save(out_name, writer="ffmpeg", fps=fps, dpi=300)
    plt.show()



if __name__ == '__main__':
    deterministic = True

    # ------------------------------------------------------------
    # BUILD SMOOTHLY VARYING PARAMETERS
    # init inputs as zeros
    base_params = np.zeros(12, dtype=np.float32)

    # init random generation of strain components
    if deterministic:
        rng = np.random.default_rng(seed=42)
    else:
        rng = np.random.Generator(np.random.PCG64())
    base_params[6:] = rng.uniform(low=-1e-6, high=1e-6, size=6)

    # Indices of the parameters to vary
    vary_indices = (0, 1, 2, 3, 4, 5, 11)

    # Ranges (start, end) for each varying parameter, aligned with vary_indices
    vary_ranges = ((-1, 1), (1, -1), (0.45, 0.85), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (-1e-6, 1e-6))
    # ------------------------------------------------------------
    # Animation settings
    n_frames = 360
    fps = 30
    batch_size = 64
    easing = "cubic"  # "linear", "cosine", "cubic"
    cmap = cm.vik
    # ------------------------------------------------------------

    X_params = build_parameter_path(base_params, vary_indices, vary_ranges, n_frames, easing)  # shape: (n_frames, 12)
    X_params[:, 6:] = X_params[:, 6:] / np.linalg.norm(X_params[:, 6:], axis=1, keepdims=True)  # norm strain tensor

    model_path = "./checkpoints/LARGE_v2i_max_entropy_C2DT_2025_12_19_1547_ckpt.keras"
    model_name = os.path.split(model_path)[-1][:-27]
    remove_variational = False if model_name[-19:] == 'variational_removed' else True
    # Forward/inference model
    fwd_model = load_fwd_model(
        model_path=model_path,
        n_var_layers=13,
    )

    Y_surrogate = fwd_model.predict(X_params)
    Y_analytic = np.empty_like(Y_surrogate)

    x = np.linspace(-1, 1, 64, endpoint=True)
    y = np.linspace(-1, 1, 64, endpoint=True)
    x1, x2 = np.meshgrid(x, y)
    x3 = np.zeros(x2.shape)

    for idx_sample in trange(X_params.shape[0]):
        c1, c2, c3 = X_params[idx_sample, :3]
        L, W, T = X_params[idx_sample, 3:6]
        espv_ij = X_params[idx_sample, 6:]

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
        Y_analytic[idx_sample] = np.stack([u1, u2, u3], axis=-1)

    # reorder ys  from NED -> ENU
    Y_analytic = Y_analytic[..., [1, 0, 2]]
    Y_surrogate = Y_surrogate[..., [1, 0, 2]]

    Y_analytic[..., 2] *= -1
    Y_surrogate[..., 2] *= -1

    # get scaled and unscaled versions
    Y_analytic_scaled, max_disp = scale_Y_maxnorm_per_sample(Y_analytic)

    norm = colors.CenteredNorm(vcenter=0, halfrange=np.quantile(Y_surrogate, 0.995))

    make_animations(x_params=X_params,
                    y1=Y_analytic_scaled,
                    y2=Y_surrogate,
                    norm=norm,
                    cmap=cmap,
                    out_name='surrogate_inference_parameter_sweep.mp4')

