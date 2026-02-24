# --------------------------------------------------------------
# Created by Kaan Cökerim¹ on 01. December 2025
#
# Script to run the inversion
#
# ¹Tectonic Geodesy, Ruhr University Bochum, Germany
# Email: kaan.coekerim@rub.de
# --------------------------------------------------------------

import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from tqdm import trange, tqdm
from surrogate_utils import BoundedMatMulInputLayer
from plot_utils import plot_cuboids, plot_surface_displacement
from invert_utils import (get_debug_cuboids, build_random_cuboids, ms_best_seeds, load_external_ys,
                          scale_invariant_rel_mse, scale_invariant_mse, optimal_scale)


def load_fwd_model(
        model_path: str,
        n_var_layers: int = 13,
        verbose_trainable: bool = False
                   ) -> keras.Model:
    """
    Build frozen forward model by removing the stochastic sampling block
    Args:
        model_path: path to the trained surrogate model
        n_var_layers: number of layers at the beginning of the model to remove (incl. `keras.layers.Input()`)
        verbose_trainable: print whether each layer is trainable or frozen

    Returns:
        Frozen forward model
    """
    orig_trained_model = keras.models.load_model(model_path)
    trained_model = keras.models.clone_model(orig_trained_model)
    trained_model.set_weights(orig_trained_model.get_weights())

    selected_layers = trained_model.layers[n_var_layers:]
    inp = keras.Input(shape=(12,))
    x = inp
    for layer in selected_layers:
        layer.trainable = False
        x = layer(x)
    trained_model = keras.Model(inputs=inp, outputs=x)
    trained_model.layers[0].trainable = False

    if verbose_trainable:
        for j in range(len(trained_model.layers)):
            print(trained_model.layers[j], 'Trainable? ', trained_model.layers[j].trainable)

    trained_model.compile(loss="mse", optimizer="adam")
    return trained_model


def load_invert_model(
        model_path: str,
        n_var_layers: int = 14,
        bounded_init: bool = False,
        use_constraints: bool = False,
        verbose_trainable: bool = False
                      ):
    """

    Args:
        model_path: path to the trained surrogate model
        n_var_layers: number of layers at the beginning of the model to remove (incl. `keras.layers.Input()`)
        bounded_init: whether to constrain the initialization of cuboid parameters
        use_constraints: whether to enforce constraints on cuboid parameters during inversions
        verbose_trainable: print whether each layer is trainable or frozen

    Returns:

    """

    raw_model = keras.models.load_model(model_path)
    orig_model = keras.models.clone_model(raw_model)
    orig_model.set_weights(raw_model.get_weights())

    inp = keras.Input(shape=(12, 12))

    matmul_layer = BoundedMatMulInputLayer(use_constrains=use_constraints, bounded_init=bounded_init)

    x = matmul_layer(inp)
    selected_layers = orig_model.layers[n_var_layers:]
    for layer in selected_layers:
        layer.trainable = False
        x = layer(x)

    inv_model = keras.Model(inputs=inp, outputs=x)
    inv_model.layers[0].trainable = False  # set Input layer to NOT be trainable
    inv_model.layers[1].trainable = True  # set MatMulInputLayer to be trainable

    if verbose_trainable:
        for j in range(len(inv_model.layers)):
            print(inv_model.layers[j], 'Trainable? ', inv_model.layers[j].trainable)

    inv_model.compile(loss='mse', optimizer='adam')
    return inv_model


def invert(
        model_path: str | os.PathLike,
        n_samples: int = 1,
        use_debug_params: bool = True,
        y_external_path: str | os.PathLike | None = None,
        bounded_init: bool = True,
        use_constraints: bool = True,
        multi_start: bool = True,
        multi_start_inv_metric: Literal['mse', 'scale_invariant', 'scale_invariant_rel'] = 'scale_invariant_rel',
        use_optimal_scale: bool = True,
        save_results: bool = True,
        save_figs: bool = True
           ):
    """
    Routine to run the inversion of cuboid sources parameters from surface displacement fields
    Args:
        model_path: path to trained surrogate model
        n_samples: number of samples to invert
        use_debug_params: whether to use pre-defined cuboids to sanity check inversion
        y_external_path: path to external displacement fields from other methods
        bounded_init: whether to constrain the initialization of cuboid parameters
        use_constraints: whether to enforce constraints on cuboid parameters during inversions
        multi_start: whether to use multi-start initialization
        multi_start_inv_metric: the metric to use during optimization.
                                Either 'mse', 'scale_invariant' or 'scale_invariant_rel'
        use_optimal_scale: whether to use optimal scale factor scaling losses for multi-start initialization
        save_results: if `True`, save results as a `.npz`-file in `./inv_outs/`
        save_figs: if `True`, save result figures as a `.png`-file in `./figs/`
    Returns:
        None
    """

    # ------------- LOAD FORWARD INFERENCE MODEL
    model_name = os.path.split(model_path)[-1][:-27]
    fwd_model = load_fwd_model(model_path=model_path, n_var_layers=13)


    # ------------- GENERATE SAMPLES FOR INVERSION OR LOAD EXTERNAL DATA
    if y_external_path is None:  # build X and Y from trained model

        # build X input params
        if use_debug_params:  # build basic debugging X data
            X_true = get_debug_cuboids()
            n_samples = X_true.shape[0]
            target_geometry = 'Debug_Cuboid'

        else:  # build random X samples
            X_true = build_random_cuboids(n_samples=n_samples, value_ranges='default', deterministic=True)
            target_geometry = 'Random_Cuboids'

        # norm strain tensor
        X_true[:, 6:] = X_true[:, 6:] / np.linalg.norm(X_true[:, 6:], axis=-1, keepdims=True)

        # build Y ground truth from trained model
        Y_true = fwd_model.predict(X_true)
        external_keys = None

    else:  # use external data set to invert cuboids from
        Y_true, X_true, external_keys, target_geometry = load_external_ys(external_path=y_external_path)

        if n_samples > Y_true.shape[0]:
            n_samples = Y_true.shape[0]
        else:
            Y_true = Y_true[:n_samples]
            X_true = X_true[:n_samples]


    fig_path = f'./figs/figs_{target_geometry}/{model_name}'
    os.makedirs(fig_path, exist_ok=True)

    results_fname = f"{model_name.replace(' ', '_')}_{target_geometry}_inv_outputs_{n_samples}_samples.npz"
    results_path = f"inv_outs/{results_fname}"

    save_buf = {
        "sample_id": [], "y_true": [], "y_inv": [],
        "x_init": [], "x_true": [], "x_inverted": [],
    }

    print(target_geometry)
    print(f'Will save to inversion outputs -> {results_path}')

    for idx_sample in trange(n_samples, position=0, leave=True, desc='Inverting '):
        if multi_start:
            # --- Monte Carlo screening: get good starting points ---
            seeds, seed_mse, seed_s = ms_best_seeds(fwd_model, Y_true[idx_sample], M=4000, topk=5,
                                                    seed=idx_sample + 123, clamp_nonneg=False)
            I = tf.eye(12, dtype=tf.float32)[None, :, :]
            Y0 = tf.convert_to_tensor(Y_true[idx_sample][None, ...], tf.float32)

            n_steps_1, n_steps_2 = 500, 1500

            ds = tf.data.Dataset.from_tensors((I, Y0)).repeat(n_steps_1 + n_steps_2)

            lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=[n_steps_1],  # after this many updates, switch LR
                values=[1e-2, 1e-3],
            )

            best = {"mse_scale_invar": np.inf, "x": None, "s": 1.0, "x_init": None}

            for iseed, seed in enumerate(
                    tqdm(seeds, position=1, leave=False, desc=f'MC Sampling sample {idx_sample:02d}')
            ):
                inv_model = load_invert_model(
                    model_path=model_path,
                    n_var_layers=13,
                    bounded_init=bounded_init,
                    use_constraints=use_constraints,
                    verbose_trainable=False
                )

                # set MC-seed as initial parameters
                inv_model.layers[1].set_weights([seed.reshape(1, 12).astype(np.float32)])
                X_init = inv_model.layers[1].get_weights()[0].copy()

                if multi_start_inv_metric == 'mse':
                    loss_metric = 'mse'
                elif multi_start_inv_metric == 'scale_invariant':
                    loss_metric = scale_invariant_mse(eps=1e-12, clamp_nonneg=False)
                elif multi_start_inv_metric == 'scale_invariant_rel':
                    loss_metric = scale_invariant_rel_mse(eps=1e-12, clamp_nonneg=False)

                # compile model
                inv_model.compile(
                    loss=loss_metric,
                    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
                )
                inv_model.fit(ds, epochs=1, steps_per_epoch=n_steps_1+n_steps_2, verbose=0)

                # read out inverted parameters
                x_inv = inv_model.layers[1].get_weights()[0]
                # compute forward prediction (deterministic) and optimal scale
                y_pred = fwd_model.predict(x_inv, verbose=0)[0]  # (64,64,3)

                if use_optimal_scale:
                    s = optimal_scale(Y_true[idx_sample], y_pred, clamp_nonneg=False)
                else:
                    s = 1
                y_hat = s * y_pred
                mse_scale_invar = np.mean((Y_true[idx_sample] - y_hat) ** 2)

                if mse_scale_invar < best["mse_scale_invar"]:
                    best.update({"mse_scale_invar": mse_scale_invar, "x": x_inv, "s": s, "x_init": X_init})

            # after the multi-start loop, use the best solution
            X_init = best["x_init"]
            X_inverted = best["x"]
            Y_pred_inverted = (best["s"] * fwd_model.predict(X_inverted, verbose=0)[0])[None, ...]
            # -------------------------------------------------------------------------

        else:
            inv_model = load_invert_model(model_path=model_path,
                                          n_var_layers=13,
                                          bounded_init=bounded_init,
                                          use_constraints=use_constraints,
                                          verbose_trainable=False)

            # set epochs and inversion X and Y
            epoch_size = 100
            Y_inv = np.array([Y_true[idx_sample] for _ in range(epoch_size)])
            X_identities = np.array([np.eye(12) for _ in range(epoch_size)])

            X_init = inv_model.layers[1].get_weights()[0].copy()

            inv_model.fit(x=X_identities, y=Y_inv, epochs=epoch_size, batch_size=1, verbose=0)
            X_inverted = inv_model.layers[1].get_weights()[0]

            Y_pred_inverted = inv_model.predict(np.eye(12)[None, :], verbose=0)

        # ----------------------------------------------
        # SAVE RESULTS
        save_buf["sample_id"].append(idx_sample)
        save_buf["y_true"].append(Y_true[idx_sample].astype(np.float32))  # (64,64,3)
        save_buf["y_inv"].append(Y_pred_inverted[0].astype(np.float32))  # (64,64,3)
        save_buf["x_init"].append(X_init.astype(np.float32))  # (1,12)
        save_buf["x_true"].append(X_true[idx_sample][None, :].astype(np.float32))  # (1,12)
        save_buf["x_inverted"].append(X_inverted.astype(np.float32))  # (1,12)

        # ----------------------------------------------
        # PLOTTING
        fig_3d = plot_cuboids(x_init=X_init, x_inverted=X_inverted, x_true=X_true[idx_sample][None, :],
                              external_keys=external_keys)
        fig_3d.suptitle(f'{model_name}\nSamp ID: {idx_sample:04d} -- {target_geometry}')


        fig_surf = plot_surface_displacement(y_true=Y_true[idx_sample], y_inv=Y_pred_inverted[0],
                                             x_true=X_true[idx_sample][None, :], x_inverted=X_inverted,
                                             external_keys=external_keys)
        fig_surf.suptitle(f'{model_name}\nSamp ID: {idx_sample:04d} -- {target_geometry}')

        if save_figs:
            fig_3d.savefig(os.path.join(fig_path, f'{model_name}_{target_geometry}_{idx_sample:04d}_3d_plots.png'),
                        dpi=300, format='png', bbox_inches='tight')
            fig_surf.savefig(os.path.join(fig_path, f'{model_name}_{target_geometry}_{idx_sample:04d}_surface_disp.png'),
                        dpi=300, format='png', bbox_inches='tight')

        plt.show()

    # --------------------------------------
    # SAVE RESULTS
    if save_results:
        np.savez_compressed(
            results_path,
            sample_id=np.asarray(save_buf["sample_id"], dtype=np.int32),
            y_true=np.stack(save_buf["y_true"], axis=0),
            y_inv=np.stack(save_buf["y_inv"], axis=0),
            x_init=np.stack(save_buf["x_init"], axis=0),
            x_true=np.stack(save_buf["x_true"], axis=0),
            x_inverted=np.stack(save_buf["x_inverted"], axis=0),
            external_keys=(np.asarray(external_keys, dtype="U") if external_keys is not None
                           else np.asarray([], dtype="U")),
            model_path=str(model_path),
            target_geometry=target_geometry,
            model_name=model_name,
        )
        print(f"[Saved] inversion outputs -> {results_path}")


if __name__ == '__main__':
    n_samples = 100

    model_path = "./checkpoints/LARGE_v2i_max_entropy_C2DT_2025_12_19_1547_ckpt.keras"

    faults_external_path = "./synthetic_training_data_generation/random_faults_34186.mat"
    mogi_external_path = "./synthetic_training_data_generation/mogi_random_100.mat"

    # INVERT DEBUG CUBOIDS
    invert(model_path=model_path, n_samples=n_samples, use_debug_params=True,
           y_external_path=None, multi_start=True)
    # INVERT RANDOM CUBOIDS
    invert(model_path=model_path, n_samples=n_samples, use_debug_params=False,
           y_external_path=None, multi_start=True)

    # INVERT FAULTS
    invert(model_path=model_path, n_samples=n_samples, use_debug_params=False,
           y_external_path=faults_external_path, multi_start=True)

    # INVERT MOGI SOURCES
    invert(model_path=model_path, n_samples=n_samples, use_debug_params=False,
           y_external_path=mogi_external_path, multi_start=True)
