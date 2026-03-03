[![DOI](https://zenodo.org/badge/1165868423.svg)](https://doi.org/10.5281/zenodo.18768998)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-darkred.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

# Invertible Surrogate of Anelastic Eigenstrain Cuboids Deformation Sources

The paper associated with this repository is currently submitted and under review at __Geophysical Journal International__:

> Kaan Çökerim and Jonathan Bedford (2026). One Source to Fit Them All: 
> Versatile Surrogate Inversion of Diverse Surface Deformation Sources,
>
> A preprint version is openly available on _ESS Open Archive_ : [https://doi.org/10.22541/essoar.177249026.64692223/v1](https://doi.org/10.22541/essoar.177249026.64692223/v1)\
> If you use this codebase in your own work **please cite the preprint**.

## What this repository contains:
This repository contains a minimal example to reproduce the results presented in the paper with the codes listed below:

### 1. Synthetic Data Generation
`./synthetic_data_generation/` contains scripts for sample generation:
  - `computeDisplacementVerticalShearZone.py`: modified version of
  the anelastic cuboid eigenstrain solutions originally by 
  [Barbot et al. (2017)](https://pubs.geoscienceworld.org/ssa/bssa/article/107/2/821/354173/Displacement-and-Stress-Associated-with).
  Specifically, we vectorized the original version by changing from `sympy` to `numpy` for numerical computations and 
  providing a vectorized `acoth()` routine.
  The original version is provided [here](https://bitbucket.org/sbarbot/bssa-2016237/src/master/python/).
  - `generate_samples_hdf.py`: parallelized python routine to generate a large number of random cuboid 
  samples for surrogate training. The samples are saved in a HDF5 file. 
  **Please ensure that you have enough disk space before executing (~45 GB for 1 Mio. samples)**
  - `random_example_training_cuboids.h5` contains an example dataset of random 1024 cuboids.
  - `random_faults_34186.mat` contains an example dataset of 100 deformation fields from 
  random Okada-type dislocations generated using the numerical implementation of [Nikkhoo & Walter (2015)](https://doi.org/10.1093/gji/ggv035)
  which is available [here](https://www.volcanodeformation.com/software).
  - `mogi_random_100.mat` contains an example dataset of 100 deformation fields from random Mogi sources
  generated using the numerical implementation of Institut de Physique du Globe de Paris
  which is available [here](https://github.com/IPGP/deformation-lib/tree/master/mogi).

> The original `h5`-file with the 1 Million samples from the paper has a file size of **45 GB**. 
> Hence, we can't provide here in this repository but encourage users to generate a training set on their own 
> using the provided scripts. Ensure that your system can store a `h5`-file with the number of samples you select
> as these files tend to get large.

### 2. Surrogate Training
  - `h5_dataloader.py`: dataloader for passing the data from the sample `h5`-file to the trainer
  - `surrogate_utils.py`: utilities for surrogate training.
  - `model_trainer.py`: routine to run the training of the surrogate model.
    - `./checkpoints/`: location of model checkpoints.
    - `./trained_models/`: location where final trained models are saved.

### 3. Running the inversion
 - `invert_for_cuboids.py`: inversion routine for single sources.
 - `invert_multisource.py`: inversion routine for multiple sources.
 - `invert_utils.py`: utilities for inversion.
 - `multisource_utils.py`: utilities for multi-source inversion.
  
  Inversion results will be saved in `./inv_outs/` which already contains inversion examples for following cases:
 - `Debug_Cuboids_inv_outputs.npz`: 5 pre-defined cuboids for debugging and sanity checking .
 - `Random_Cuboids_inv_outputs.npz`: 100 random cuboid samples.
 - `Faults_inv_outputs.npz`: 100 random Okada-type faults (generated from `./synthetic_data_generation/random_faults_34186.mat`).
 - `Mogi_inv_outputs.npz`: 100 random Mogi-type sources (generated from `./synthetic_data_generation/mogi_random_100.mat`).


### 4. Plotting Results
 - `plot_utils.py`: utility functions for preliminary plotting of trained surrogate and inversion results.
 - `inv_plots.py`: functions to create figures of the inversion result.
 - `inference_comparison.py`: functions to create figures of forward inference result.
 - `animate_surrogate_inference.py`: script to create an animation of a cuboid parameter sweep through the inference model.
 - `parameter_misfit_relation.py`: creates a large plot relating misfit between the cuboid parameters.
 
All plots will be saved in `./figs/`.



## Software Requirements
We provide the environment in which we developed these codes in the `environment.yml` file.
The environment can be replicated in Conda by running `$ conda env create -f environment.yml` in the terminal.

It is, however, not necessary to replicate our provided environment. Rather it is required
that an environment exists at least with the packages and versions listed below.
To ensure that custom classes run properly, it is especially important
that the user's environment has the same Keras and Tensorflow versions that are listed in `environment.yml`.

The (minimal) required python packages and versions are:
- Python Version: `3.12.0`
- Keras Version: `3.6.0`
- Tensorflow Version: `2.18.0`
- SciPy Version: `1.16.0`
- NumPy Version: `2.0.2`
- joblib Version: `1.5.2` (required for fast parallel sample generation)
- h5py Version: `3.14.0` (required for I/O of large sample files)
- tqdm Version: `4.67.1` (required to display progress on long sample generation processes)

The plotting routines require:
- Matplotlib Version: `3.10.3`
- Seaborn Version: `0.13.2`
- cmcrameri Version: `1.9` (scientific colormaps)
<br/><br/>
---