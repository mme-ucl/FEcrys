# FECrys

FECrys implements statistical-mechanics and machine-learning methods for
calculating free energies of isolated molecules and molecular crystals. The
workflow combines [probabilistic generative
models](https://www.science.org/doi/10.1126/science.aaw1147) with ideas from
[targeted free-energy
perturbation](https://doi.org/10.1103/PhysRevE.65.046122).

<p align="center">
  <img width="371" height="356" alt="Overview of the FECrys workflow" src="https://github.com/user-attachments/assets/f0e55d04-9974-485b-9afd-853934083d47">
</p>

> [!NOTE]
> FECrys is currently a research codebase organised around Jupyter notebooks,
> rather than an installable Python package. Run notebooks from the repository
> checkout so that the local `O` module and bundled data can be found.

## Contents

- [`O/`](O): molecular-mechanics, normalising-flow, symmetry, and plotting code
- [`notebooks/`](notebooks): installation checks, examples, and paper workflows
- [`environment.yml`](environment.yml): recommended Conda environment
- [`requirements.txt`](requirements.txt): pip-installable Python dependencies
- [`cmp_gpu.yml`](cmp_gpu.yml): alternative GPU environment specification

## Architecture documentation

The [interactive Figure 1 code map](docs/figure-1-code-map.html) connects the
normalising-flow architecture in the [accompanying manuscript](https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-92x2f/v4)  to the modules,
classes, and functions that implement each stage. Select any component to see
its responsibility, data shape, relationships, and source location.

## Requirements

- Linux (tested on Ubuntu 20.04/24.04, RHEL 8+, CentOS Stream 9, Fedora 38+,
  and WSL2)
- [Miniforge](https://github.com/conda-forge/miniforge) or another Conda
  distribution
- Python 3.10 (selected by `environment.yml`)
- An NVIDIA GPU and compatible driver are optional

## Installation

The recommended setup uses [`environment.yml`](environment.yml). This file
tells Conda to:

- create an environment named `fecrys`;
- install Python 3.10;
- obtain scientific and molecular-modelling packages from conda-forge; and
- install the TensorFlow-related packages listed in its `pip` section.

### 1. Install Conda

If `conda --version` does not return a version number, install
[Miniforge](https://github.com/conda-forge/miniforge), then open a new terminal.
Miniforge is recommended because most FECrys dependencies are provided by
conda-forge.

### 2. Get the code

```bash
git clone https://github.com/mme-ucl/FEcrys.git
cd FEcrys
```

If you already have a checkout, open a terminal in its root directory—the one
containing `environment.yml`—instead.

### 3. Create the environment

Run:

```bash
conda env create -f environment.yml
```

Conda reads the environment name and complete dependency list from the YAML
file, resolves compatible package versions, and installs them in an isolated
environment. This may take several minutes. It does not modify the system
Python installation.

### 4. Activate the environment

```bash
conda activate fecrys
```

The terminal prompt will normally show `(fecrys)` while the environment is
active. Confirm that the expected interpreter is selected with:

```bash
python --version
conda env list
```

Python should report version 3.10, and the `fecrys` entry in the environment
list should be marked as active.

### 5. Update an existing environment

When `environment.yml` changes, synchronize an existing installation with:

```bash
conda env update -n fecrys -f environment.yml --prune
```

The `--prune` option removes packages that are no longer declared in the file,
keeping the environment aligned with the repository configuration.

### Recreate the environment

If dependency resolution or imports become inconsistent, recreate the
environment from the YAML file:

```bash
conda deactivate
conda env remove -n fecrys
conda env create -f environment.yml
conda activate fecrys
```

The current YAML configuration includes CUDA-related packages for GPU use. An
NVIDIA GPU is not required to read the code or run CPU-compatible workflows,
but GPU acceleration also requires a compatible NVIDIA driver on the host.

### Pip fallback

`requirements.txt` is provided for environments where Conda is unavailable:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Conda is preferred because some molecular-modelling dependencies are more
reliably distributed through conda-forge.

## Verify the setup

From the repository root, activate the environment and run:

```bash
python -c "from O.interface import *; print('FECrys is ready')"
```

You can also inspect the TensorFlow devices available to the environment:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

A GPU is not required for importing FECrys or exploring the notebooks, but it
is recommended for training normalising-flow models.

## Run the notebooks

Start Jupyter from the repository root:

```bash
jupyter lab
```

Begin with
[`JN_0 - installation of dependencies.ipynb`](notebooks/JN_0%20-%20installation%20of%20dependencies.ipynb)
to check the environment. The main examples are:

| Notebook | Purpose |
| --- | --- |
| `JN_1` | Reproduce the main figures from the paper |
| `JN_2` | Load a model from the paper |
| `JN_3` | Run an NPT equilibration example |
| `JN_4` | Train a model with a minimal example |
| `JN_4.5` | Train with fewer force-field evaluations |
| `JN_6` | Explore identity initialisation |

Notebook filenames include a short status marker inherited from the research
workflow. Treat notebooks marked `(n)` as work in progress.

Precomputed results and example systems are available from the associated
[Zenodo record](https://zenodo.org/records/15164990).

## Troubleshooting

### The `O` module cannot be imported

Confirm that the `fecrys` environment is active and that the command or Jupyter
server was started from the repository root:

```bash
conda activate fecrys
cd /path/to/FEcrys
python -c "from O.interface import *"
```

### TensorFlow does not detect a GPU

First check that the NVIDIA driver is visible with `nvidia-smi`, then compare
the installed driver, CUDA runtime, TensorFlow, and TensorFlow Probability
versions. CPU execution remains available when no compatible GPU is detected.

### OpenMM, OpenFF, or RDKit fails to install

Use the Conda environment rather than the pip fallback. If an existing
environment has become inconsistent, recreate it:

```bash
conda deactivate
conda env remove -n fecrys
conda env create -f environment.yml
```

When reporting a problem, include the operating system, output of `conda list`,
Python version, and—if relevant—`nvidia-smi` output.

## Citation

If FECrys is useful in your work, please cite the relevant publication:

1. E. Olehnovics, Y. M. Liu, N. Mehio, A. Y. Sheikh, M. R. Shirts, and M.
   Salvalaglio, “Lattice Free Energies of Molecular Crystals Using Normalizing
   Flow,” *ChemRxiv* (2025).
   [doi:10.26434/chemrxiv-2025-92x2f-v3](https://doi.org/10.26434/chemrxiv-2025-92x2f-v3)
2. E. Olehnovics, Y. M. Liu, N. Mehio, A. Y. Sheikh, M. R. Shirts, and M.
   Salvalaglio, “Accurate Lattice Free Energies of Packing Polymorphs from
   Probabilistic Generative Models,” *Journal of Chemical Theory and
   Computation* **21** (2025), 2244–2255.
   [doi:10.1021/acs.jctc.4c01612](https://doi.org/10.1021/acs.jctc.4c01612)
3. E. Olehnovics, Y. M. Liu, N. Mehio, A. Y. Sheikh, M. R. Shirts, and M.
   Salvalaglio, “Assessing the Accuracy and Efficiency of Free Energy
   Differences Obtained from Reweighted Flow-Based Probabilistic Generative
   Models,” *Journal of Chemical Theory and Computation* **20** (2024),
   5913–5922.
   [doi:10.1021/acs.jctc.4c00520](https://doi.org/10.1021/acs.jctc.4c00520)

## Support

Please use the repository's [GitHub issue
tracker](https://github.com/mme-ucl/FEcrys/issues) for reproducible bugs and
documentation problems.
