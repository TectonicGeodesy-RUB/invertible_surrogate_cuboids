# -----------------------------------------------------
# Created by Kaan Cökerim¹ on 01. December 2025
# Functions to create large number of samples
# fast in parallel and save them in a HDF5-file
#
# ¹Tectonic Geodesy, Ruhr University Bochum, Germany
# Email: kaan.coekerim@rub.de
# -----------------------------------------------------

import argparse
import numpy as np
import h5py
from joblib import Parallel, delayed
from tqdm import trange
from computeDisplacementVerticalShearZone import computeDisplacementVerticalShearZone


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    """
    Logarithmic uniform distribution sampling function
    Args:
        rng: NumPy random generator
        low: lower bound of the uniform distribution. Value has to be > 0
        high: upper bound of the uniform distribution

    Returns:
        random sample from a logarithmic uniform distribution

    """
    return 10 ** rng.uniform(np.log10(low), np.log10(high))


def sample_parameters(rng: np.random.Generator) -> dict[str, float | np.ndarray]:
    """
    Function to randomly generate one cuboid sample
    Args:
        rng: NumPy random generator

    Returns:
        A single cuboid sample as a dictionary as {c1, c2, c3, L, W, T, espv}
    """

    # sample centroid location
    c1 = rng.uniform(-1, 1)
    c2 = rng.uniform(-1, 1)
    c3 = rng.uniform(0.0001, 1)

    # sample cuboid extend
    T  = rng.uniform(0.0001, 1)
    W  = rng.uniform(0.0001, min(2*c3, 1))
    L  = rng.uniform(0.0001, 1)

    # sample eigenstrain components
    epsv = rng.uniform(-1e-6, 1e-6, size=6)

    return dict(c1=c1, c2=c2, c3=c3, L=L, W=W, T=T, epsv=epsv)


def sample_parameters_oversample_planes(
        rng: np.random.Generator,
        p_plane: float = 0.35,
        thin_low: float = 1e-4,
        thin_high: float = 3e-3,
        big_low: float = 0.2,
) -> dict[str, float | np.ndarray]:
    """
    Function to randomly generate one cuboid sample and oversample plane-like cuboids
    Args:
        rng: NumPy random generator
        p_plane: fraction of plane-like samples (suggested between 0.2–0.6)
        thin_low: minimal horizontal extend of plane-like cuboids
        thin_high: maximum horizontal extend of plane-like cuboids
        big_low: minimum extend of non-thin extend of plane-like cuboids

    Returns:

    """

    # --- horizontal location ---
    c1 = rng.uniform(-1, 1)
    c2 = rng.uniform(-1, 1)

    # --- depth ---
    c3 = rng.uniform(1e-4, 1.0)

    # --- sample horizontal cuboid extend ---
    L = rng.uniform(1e-4, 1.0)
    T = rng.uniform(1e-4, 1.0)

    # --- sample W constrained by depth  ---
    Wmax = min(2 * c3, 1.0)
    W = rng.uniform(1e-4, Wmax)

    # --- randomly trigger plane-like oversampling: thin T OR thin L (50/50) ---
    if rng.random() < p_plane:
        if rng.random() < 0.5:
            # thin T
            T = log_uniform(rng, thin_low, thin_high)
            # make in-plane sizes reasonably large
            L = log_uniform(rng, big_low, 1.0)
        else:
            # thin L
            L = log_uniform(rng, thin_low, thin_high)
            T = log_uniform(rng, big_low, 1.0)

        # encourage W large as allowed by depth
        if Wmax > big_low:
            W = log_uniform(rng, big_low, Wmax)
        else:
            W = rng.uniform(1e-4, Wmax)

    # --- strain ---
    epsv = rng.uniform(-1e-6, 1e-6, size=6)

    return dict(c1=c1, c2=c2, c3=c3, L=L, W=W, T=T, epsv=epsv)


def compute_displacement_single(seed: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Get a random cuboid sample and compute the associated displacement field.
    Returns the three-component displacement field and the 12 cuboid parameters.
    Args:
        seed: random seed for generating cuboid samples

    Returns:
        Tuple containing the three-component displacement field and the 12 cuboid parameters
    """

    rng = np.random.default_rng(seed)

    p = sample_parameters_oversample_planes(rng=rng)

    c1, c2, c3 = p['c1'], p['c2'], p['c3']
    L, W, T = p['L'], p['W'], p['T']
    epsv = p['epsv']

    # Grid
    x = np.linspace(-1, 1, 64, endpoint=True)
    y = np.linspace(-1, 1, 64, endpoint=True)
    x1, x2 = np.meshgrid(x, y)
    x3 = np.zeros_like(x2)

    # anchor
    q1 = c1 - (L / 2.0)
    q2 = c2
    q3 = c3 - (W / 2.0)

    u1, u2, u3 = computeDisplacementVerticalShearZone(
        x1=x1, x2=x2, x3=x3,
        q1=q1, q2=q2, q3=q3,
        L=L, W=W, T=T,
        epsv11p=epsv[0], epsv12p=epsv[1], epsv13p=epsv[2],
        epsv22p=epsv[3], epsv23p=epsv[4], epsv33p=epsv[5],
        G=1.0, nu=0.25, theta=0.0,
    )

    params_vec = np.asarray(
        [q1, q2, q3,
         c1, c2, c3,
         L, W, T,
         *epsv],
        dtype=np.float32,
    )

    return (
        u1.astype(np.float32),
        u2.astype(np.float32),
        u3.astype(np.float32),
        params_vec,
    )


def generate_hdf5_parallel(
    filename: str,
    n_samples: int = 1_000_000,
    batch_size: int = 512,
    n_jobs: int = 8,
    compression: str | None = "gzip",
    compression_level: int = 4,
) -> None:
    """
    Parallel sample generation and saving as HDF5.
    Args:
        filename: output filename
        n_samples: number of samples to generate
        batch_size: Number of samples per batch to randomly generate samples and compute displacement fields
        n_jobs: number of parallel jobs
        compression: compression level of the hdf5 file. Select `None` to speed up.
        compression_level: compression level of the hdf5 file.

    Returns:

    """

    n_grid = 64
    n_components = 3
    n_params = 15  # q1,q2,q3,c1,c2,c3,L,W,T,epsv*6

    with h5py.File(filename, "w") as f:
        u_ds = f.create_dataset(
            "u",
            shape=(0, n_grid, n_grid, n_components),
            maxshape=(None, n_grid, n_grid, n_components),
            dtype="float32",
            chunks=(batch_size, n_grid, n_grid, n_components),
            compression=compression,
            compression_opts=compression_level if compression else None,
        )

        params_ds = f.create_dataset(
            "params",
            shape=(0, n_params),
            maxshape=(None, n_params),
            dtype="float32",
            chunks=(batch_size, n_params),
            compression=compression,
            compression_opts=compression_level if compression else None,
        )

        n_written = 0

        # each sample gets its own seed
        global_rng = np.random.default_rng(12345)

        for start in trange(0, n_samples, batch_size, desc="Generating HDF5"):
            end = min(start + batch_size, n_samples)
            current_batch = end - start

            # draw seeds for each sample in this batch
            seeds = global_rng.integers(0, 2**31 - 1, size=current_batch)

            # parallel compute
            results = Parallel(
                n_jobs=n_jobs,
                backend="loky",
            )(delayed(compute_displacement_single)(int(s)) for s in seeds)

            # pack into batch arrays
            u_batch = np.empty(
                (current_batch, n_grid, n_grid, n_components),
                dtype=np.float32,
            )
            params_batch = np.empty(
                (current_batch, n_params),
                dtype=np.float32,
            )

            for i, (u1, u2, u3, p_vec) in enumerate(results):
                u_batch[i, :, :, 0] = u1
                u_batch[i, :, :, 1] = u2
                u_batch[i, :, :, 2] = u3
                params_batch[i] = p_vec

            # append to datasets
            new_size = n_written + current_batch
            u_ds.resize((new_size, n_grid, n_grid, n_components))
            params_ds.resize((new_size, n_params))

            u_ds[n_written:new_size] = u_batch
            params_ds[n_written:new_size] = params_batch

            n_written = new_size

    print(f"Finished writing {n_written} samples to {filename}")


if __name__ == "__main__":
    cl = argparse.ArgumentParser()
    cl.add_argument('--out_path', type=str, required=True)
    args = cl.parse_args()

    generate_hdf5_parallel(
        filename=args.out_path,
        n_samples=1_000_000,
        batch_size=1024,
        n_jobs=10,
        compression="gzip",
        compression_level=4,
    )