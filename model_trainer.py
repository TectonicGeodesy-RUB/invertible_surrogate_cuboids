# --------------------------------------------------------------
# Created by Kaan Cökerim¹ on 03. December 2025
#
# Script to train the surrogate model
#
# ¹Tectonic Geodesy, Ruhr University Bochum, Germany
# Email: kaan.coekerim@rub.de
# --------------------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import time
import datetime
import tensorflow as tf
import keras
import numpy as np
import h5py
import json
from matplotlib import pyplot as plt
from surrogate_utils import surrogate_model, VerboseCallback, LearningRateLogger
from h5_dataloader import H5CuboidDataloader, get_num_samples, build_X_from_params, scale_Y_maxnorm_per_sample
from plot_utils import logvar_plot, surf_plots, plot_history


def train_main(
        h5_path: str | None = None,
        run_training: bool = True,
        continue_path: str | None = None,
        plot_figures: bool = False,
               ) -> None:
    """
    Routine to run the training of the surrogate model
    Args:
        h5_path: path of the HDF5-file containing the cuboid samples
        run_training: whether to run training or not. If `False`: runs only the evaluation stage
        continue_path: path to previously saved model to use for continuing training
        plot_figures: whether to plot the training and evaluation plots

    Returns:
        None
    """

    # ------- Reading in data
    # HDF5 data source
    if h5_path is None:
        h5_path = './synthetic_data_generation/random_example_training_cuboids.h5'
    n_samp_str = h5_path.split('_')[-1].split('.')[0]
    # Number of samples in HDF
    n_samples = get_num_samples(h5_path)

    # ------- Train/val/test split
    full_set = np.arange(n_samples)
    Train_Val_Test = np.array([80, 10, 10])
    pp1 = int((Train_Val_Test[0] / Train_Val_Test.sum()) * full_set.size)
    pp2 = int((Train_Val_Test[0:2].sum() / Train_Val_Test.sum()) * full_set.size)

    train_idx = full_set[:pp1]
    val_idx = full_set[pp1:pp2]
    test_idx = full_set[pp2:]

    # ------- Continuing the training of a previously trained model or starting from scratch
    if continue_path:
        model = keras.saving.load_model(continue_path)
        if run_training:
            print(f'Continuing with training of: {continue_path}')
        else:
            print(f'Running evaluation with: {continue_path}')
    else:
        model = surrogate_model()

    # -----------------------------------
    # -----------------------------------

    # ------- Running the training
    BS = 512
    LR = 1e-4

    steps_per_epoch = int(np.ceil(len(train_idx) / BS))

    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=LR,
        first_decay_steps=30 * steps_per_epoch,  # measured in batches
        t_mul=1.0,
        m_mul=1.0,
        alpha=1e-1,
        name="CosineDecayRestarts",
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss=['mse', None, None], )

    ts = datetime.datetime.now().strftime("_%Y_%m_%d_%H%M")

    model_type = model.name

    save_path = f'./trained_models/{model_type}_{ts}_{n_samp_str}.keras'
    if continue_path:
        history_path = os.path.splitext(continue_path)[0] + '_history_log.csv'
    else:
        history_path = os.path.splitext(save_path)[0] + "_history_log.csv"
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)

    # br_cb = keras.callbacks.BackupAndRestore()
    # hist_cb = keras.callbacks.History()
    cp_cb = keras.callbacks.ModelCheckpoint(filepath=f'./checkpoints/{model_type}_{ts}_ckpt.keras',
                                            monitor='val_loss', verbose=0, save_best_only=True)
    es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, restore_best_weights=True, patience=100)
    lr_logger = LearningRateLogger()
    csv_logger = keras.callbacks.CSVLogger(history_path, append=True if (continue_path and run_training) else False)

    # Keras Sequences
    train_seq = H5CuboidDataloader(
        filename=h5_path,
        indices=train_idx,
        batch_size=BS,
        workers=6,
        use_multiprocessing=True,
        max_queue_size=60,
    )

    val_seq = H5CuboidDataloader(
        filename=h5_path,
        indices=val_idx,
        batch_size=BS,
        workers=6,
        use_multiprocessing=True,
        max_queue_size=60,
    )

    if run_training:
        t0 = time.time()
        try:
            model.fit(x=train_seq,
                      validation_data=val_seq,
                      epochs=10_000,
                      batch_size=BS,
                      callbacks=[es_cb, cp_cb, VerboseCallback(), lr_logger, csv_logger],
                      verbose=0)
        except KeyboardInterrupt:
            print(f'\n\n++++{datetime.datetime.now().strftime("%d-%b %H:%M:%S")}] - '
                  f'\033[1;91m Manually Killed Training... Triggering model saving. \033[00m')
        finally:
            # ---------- Saving the model
            print(f'+\tElapsed training time: {datetime.timedelta(seconds=time.time() - t0)}')
            model.save(save_path)

            print(f'++++[{datetime.datetime.now().strftime("%d-%b %H:%M:%S")}] - '
                  f'Saved trained model to \033[1m {save_path} \033[00m')

    # --------------------------------------
    # --------- Inspecting Results ---------
    # --------------------------------------
    if plot_figures:
        # plot loss curves
        fig_loss = plot_history(history_path=history_path)

        # Grab up to 1000 test samples for inspection
        n_plot = min(1000, len(test_idx))
        plot_indices = test_idx[:n_plot]

        # Load test X/Y from HDF once
        with h5py.File(h5_path, "r") as f:
            params_test = f["params"][plot_indices, :]
            u_test = f["u"][plot_indices, ...]

        X_test = build_X_from_params(params_test)  # (n_plot, 12)
        Y_test_true, _ = scale_Y_maxnorm_per_sample(u_test)

        out = model.predict(X_test)
        Y_test_pred, logvars_test, x_test_sampled = out

        fig_logvar = logvar_plot(logvars=logvars_test, show=False)

        for sample_id in np.random.permutation(Y_test_pred.shape[0])[0:10]:
            surf_plots(x=X_test[sample_id], y_true=Y_test_true[sample_id], y_pred=Y_test_pred[sample_id], show=False)

        plt.show()



if __name__ == '__main__':
    cl = argparse.ArgumentParser()
    cl.add_argument('--h5_path', default=None, type=str)
    cl.add_argument('--run_training', default=False, action='store_true')
    cl.add_argument('--continue_path', default=None, type=str)
    cl.add_argument('--plot_figures', default=False, action='store_true')
    args = cl.parse_args()

    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(tf.config.list_physical_devices('GPU'))
    print(f'+\tH5 Path: {args.h5_path}')
    print(f'+\tRun Training: {args.run_training}')
    print(f'+\tContinuing Training: {args.continue_path}')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    train_main(
        h5_path=args.h5_path,
        run_training=args.run_training,
        continue_path=args.continue_path,
        plot_figures=args.plot_figures
               )
