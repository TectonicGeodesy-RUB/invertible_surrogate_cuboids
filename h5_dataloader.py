# --------------------------------------------------------------
# Created by Kaan Cökerim¹ on 03. December 2025
#
# Script containing the dataloader routine passed to the training model
#
# ¹Tectonic Geodesy, Ruhr University Bochum, Germany
# Email: kaan.coekerim@rub.de
# --------------------------------------------------------------

import keras
import numpy as np
import h5py


def build_X_from_params(params_batch: np.ndarray) -> np.ndarray:
    """
    Convert 'params' rows from the HDF5 file into the 12-D input X expected by the model.

    Assumes each row looks like:
      [q1,q2,q3, c1,c2,c3, L, W, T, eps11,eps12,eps13,eps22,eps23,eps33]

    and return
      X = [c1, c2, c3, L, W, T, eps11, eps12, eps13, eps22, eps23, eps33]
    """

    B = params_batch.shape[0]
    X = np.empty((B, 12), dtype=np.float32)
    # c1, c2, c3
    X[:, 0:3] = params_batch[:, 3:6]

    # L, T, W
    X[:, 3] = params_batch[:, 6]  # L
    X[:, 4] = params_batch[:, 7]  # W
    X[:, 5] = params_batch[:, 8]  # T

    # strain normalize to unit norm
    eps = params_batch[:, 9:]  # (B, 6)
    eps_norm = np.linalg.norm(eps, axis=1, keepdims=True)  # (B, 1)
    X[:, 6:] = eps / eps_norm

    return X


def scale_Y_maxnorm_per_sample(Y_unscaled: np.ndarray):
    """
    Scale displacements per sample by the maximum vector norm:

        max_disps[i] = max_{x,y} ||Y[i,x,y,:]||
        Y_scaled[i]  = Y[i] / max_disps[i]

    Y: (B, H, W, 3)
    Returns:
      Y_scaled: (B, H, W, 3)
      max_disps: (B,)
    """

    max_disps = np.linalg.norm(Y_unscaled, axis=3).max(axis=1).max(axis=1)
    Y = np.einsum('ijkl,i->ijkl', Y_unscaled, 1 / max_disps)

    return Y.astype(np.float32), max_disps.astype(np.float32)


def unscale_Y_maxnorm_per_sample(Y_scaled: np.ndarray, max_disps: np.ndarray):
    """
    Undo per-sample max-norm scaling:
        Y_unscaled[i] = Y_scaled[i] * max_disps[i]

    Y_scaled: (B, H, W, 3)
    max_disps: (B,)
    Returns:
      Y_unscaled: (B, H, W, 3)
    """
    Y_unscaled = np.einsum('ijkl,i->ijkl', Y_scaled, max_disps)
    return Y_unscaled.astype(np.float32)


class H5CuboidDataloader(keras.utils.PyDataset):
    """
    HDF5 data loader to pass to the training model
    """
    def __init__(self, filename: str, indices: np.ndarray, batch_size: int, **kwargs) -> None:
        super().__init__(**kwargs)

        self.filename = filename
        self.indices = indices
        self.batch_size = batch_size

        self._file = None
        self._u = None
        self._params = None

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.filename, 'r')
            self._u = self._file['u']
            self._params = self._file['params']

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx: int):
        self._ensure_open()

        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.indices))
        batch_ids = self.indices[start:end]

        params_batch = self._params[batch_ids, :]  # (B, 15)
        u_batch = self._u[batch_ids, :]  # (B, H, W, 3)

        X = build_X_from_params(params_batch)  # (B, 12)

        Y, _ = scale_Y_maxnorm_per_sample(u_batch)  # (B, 64, 64, 3)

        return X, Y

def get_num_samples(filename: str) -> int:
    """Return N for /u dataset in the HDF5 file."""
    with h5py.File(filename, "r") as f:
        return f["u"].shape[0]

