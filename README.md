<img width="1114" height="1068" alt="TOC_v1" src="https://github.com/user-attachments/assets/f0e55d04-9974-485b-9afd-853934083d47" />

# FECrys

The code in this repository implements stat-mech+ML methods for the efficient calculation of free energies of isolated molecules and crystalline solids using [probabilistic generative modelling](https://www.science.org/doi/10.1126/science.aaw1147) and ideas derived from [targeted free energy perturbation](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.65.046122). 


If you find this repository uesful please consider citing: 

- [Lattice Free Energies of Molecular Crystals Using Normalizing Flow](https://doi.org/10.26434/chemrxiv-2025-92x2f-v3), Edgar Olehnovics, Yifei Michelle Liu, Nada Mehio, Ahmad Y Sheikh, Michael Shirts, and Matteo Salvalaglio ChemRxiv 2025 

- [Accurate Lattice Free Energies of Packing Polymorphs from Probabilistic Generative Models]() Edgar Olehnovics, Yifei Michelle Liu, Nada Mehio, Ahmad Y. Sheikh, Michael R. Shirts, and Matteo Salvalaglio
Journal of Chemical Theory and Computation 2025 21 (5), 2244-2255 DOI: 10.1021/acs.jctc.4c01612 

- [Assessing the Accuracy and Efficiency of Free Energy Differences Obtained from Reweighted Flow-Based Probabilistic Generative Models](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00520), Edgar Olehnovics, Yifei Michelle Liu, Nada Mehio, Ahmad Y. Sheikh, Michael R. Shirts, and Matteo Salvalaglio, Journal of Chemical Theory and Computation 2024 20 (14), 5913-5922, DOI: 10.1021/acs.jctc.4c00520

---

# FECrys Installation Guide

**Last Updated:** February 2026  
**Tested on:** Ubuntu 20.04 LTS, Ubuntu 24.04 LTS, Red Hat 8+, CentOS Stream 9, Fedora 38+, WSL2  
**Python Version:** 3.10+

---

## Quick Start (Recommended)

```bash
# 1. Install Miniforge (if not already installed)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
source ~/.bashrc

# 2. Create environment from environment.yml
conda env create -f environment.yml

# 3. Activate environment
conda activate fecrys

# 4. Run verification (launch Jupyter and open JN_0)
jupyter notebook
```

---

## Detailed Installation Steps

### Prerequisites

#### Linux System Setup

##### Ubuntu (20.04 LTS, 24.04 LTS)

```bash 

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install build tools
sudo apt install -y build-essential curl wget git python3-dev
```

##### Red Hat / CentOS / Fedora (8+, 9+, 38+)

```bash
# Update system packages
sudo dnf update -y
# Or for older RHEL/CentOS 8:
# sudo yum update -y

# Install build tools and development packages
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y git curl wget python3-devel

# Ensure EPEL repository is enabled (for some optional packages)
sudo dnf install -y epel-release
```

Note: These instructions work for:
- **Red Hat Enterprise Linux (RHEL)** 8+, 9+
- **CentOS Stream** 8, 9
- **Fedora** 38+
- **Rocky Linux** 8+, 9+
- **AlmaLinux** 8+, 9+

#### GPU Support (Optional but Recommended)

For NVIDIA GPU acceleration with CUDA 12.x:

##### Ubuntu

**Ubuntu 24.04 LTS:**
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-x
```

**Ubuntu 20.04 LTS:**
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-x
```

##### Red Hat / CentOS / Fedora

For **RHEL/CentOS 9+** or **Fedora 38+**:
```bash
# Add NVIDIA CUDA repository
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
# Or for RHEL 8:
# sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# Clean and update package metadata
sudo dnf clean all
sudo dnf update -y

# Install CUDA toolkit 12.x
sudo dnf install -y cuda-toolkit-12-x

# Update LD_LIBRARY_PATH
echo 'export PATH=/usr/local/cuda-12/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

##### Verify GPU Setup

Check installation:
```bash
nvidia-smi  # Should show GPU info
nvcc --version  # Should show CUDA compiler version
```

### Install Miniforge (Conda Package Manager)

```bash
# Download latest Miniforge
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

# Make executable and run
bash Miniforge3-$(uname)-$(uname -m).sh

# Initialize conda
source ~/miniforge3/bin/activate
conda init

# Restart terminal or run:
source ~/.bashrc
```

Verify conda installation:
```bash
conda --version
conda info
```

### Method 1: Environment File (Recommended)

This is the most reproducible method:

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate fecrys

# Verify installation
python -c "from O.interface import *; print('✓ FECrys ready')"
```

**All dependencies are pinned to known-working versions.**

### Method 2: Manual Installation

If you prefer to install step-by-step:

```bash
# 1. Create environment
conda create -n fecrys python=3.10 -c conda-forge -y
conda activate fecrys

# 2. Update base packages
conda update -n base -c conda-forge conda -y

# 3. Install numerical/scientific packages
conda install -c conda-forge numpy scipy matplotlib ipython jupyter jupyterlab pandas -y

# 4. Install molecular modeling packages
conda install -c conda-forge rdkit mdtraj openmm parmed pymbar mdanalysis openmmtools -y

# 5. Install OpenFF packages
conda install -c conda-forge openff-toolkit openmmforcefields -y

# 6. Install CUDA support (check your CUDA version)
conda install -c conda-forge cudatoolkit=12.1 cudnn -y

# 7. Install deep learning packages
pip install tensorflow>=2.14 tensorflow-probability>=0.23 protobuf>=3.20

# 8. Install utilities
conda install -c conda-forge pint pyyaml -y
pip install h5py
```

---

## Verification

To verify your installation run these quick tests:

### Test 1: TensorFlow and GPU

```python
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.test.is_built_with_cuda()}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
```

### Test 2: FECrys Import

```python
from O.interface import *
print("✓ FECrys imported successfully")
```

### Test 3: Key Packages

```python
import numpy, scipy, matplotlib, rdkit, mdtraj, openmm, pymbar, mdanalysis

print("✓ All core packages available")
```

### Test 4: Check Package Versions

```bash
conda list
```

Expected packages (among others):
- rdkit
- mdtraj
- openmm
- parmed
- pymbar
- mdanalysis
- openmmtools
- tensorflow
- tensorflow-probability

---

## Common Issues and Solutions

### Issue 1: CUDA/GPU Not Detected by TensorFlow

**Solution:**
```bash
# Verify NVIDIA driver
nvidia-smi

# Reinstall TensorFlow GPU version
pip install --upgrade tensorflow

# Check CUDA toolkit
conda install -c conda-forge cudatoolkit=12.1 -y

# Verify in Python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### Issue 2: OpenMM or OpenFF Installation Fails

**Solution:**
```bash
# Make sure you're using conda-forge channel
conda install -c conda-forge openmmtools openff-toolkit openmmforcefields --upgrade -y

# If still issues, try fresh environment
conda remove --name fecrys --all
conda env create -f environment.yml
```

### Issue 3: RDKit Not Found

**Solution:**
```bash
# RDKit must be installed via conda
conda install -c conda-forge rdkit --upgrade -y
```

### Issue 4: Protobuf Version Conflict

**Solution:**
```bash
# Install specific protobuf version
pip install protobuf==3.20 --force-reinstall

# Or upgrade
pip install protobuf>=3.20
```

### Issue 5: Python Import Errors

**Solution:**
```bash
# Verify correct environment is active
conda activate fecrys
which python  # Should show path in fecrys environment

# Check Python version
python --version  # Should be 3.10+

# Verify FECrys path
cd /path/to/FECrys  # Should be in project root
python -c "from O.interface import *"
```

### Issue 6: OpenMM Simulation Runs Slowly

**Possible causes and solutions:**
```python
# Ensure GPU platform is used
import openmm as mm
print(mm.Platform.getPlatformByName('CUDA'))  # Should not error

# Check available platforms
for i in range(mm.Platform.getNumPlatforms()):
    print(mm.Platform.getPlatform(i).getName())
```

---

## Updating Packages

To update all packages while maintaining compatibility:

```bash
# Update entire environment
conda update -n fecrys --all -c conda-forge -y

# Or update specific package
conda install -c conda-forge openmm --upgrade -y
```

---

## GPU Memory Management

If you encounter GPU out-of-memory errors:

```python
import tensorflow as tf

# Option 1: Limit GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Option 2: Set fixed memory usage
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])  # 4GB limit
```

---

## Next Steps

Once installation is complete:

1. **Check the notebooks:**
   - `JN_0`: Installation verification (you are here)
   - `JN_1`: Main figures from the paper
   - `JN_2`: Loading pretrained models
   - `JN_3`: NPT equilibration examples
   - `JN_4+`: Training examples

2. **Download additional files:**
   - [Zenodo repository](https://zenodo.org/records/15164990) with pre-computed results and example systems

---

## System Information

To collect system information for troubleshooting:

```bash
#!/bin/bash
echo "=== System Info ==="
uname -a
python --version
nvidia-smi

echo "=== Conda Info ==="
conda info

echo "=== Environment Packages ==="
conda list

echo "=== Python Packages ==="
pip list

echo "=== GPU Check ==="
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Save and share this output when reporting issues.

---

## Support and Issues

For issues specific to:
- **FECrys**: Check the notebooks and README
- **OpenMM**: See https://github.com/openmm/openmm/
- **TensorFlow**: See https://github.com/tensorflow/tensorflow
- **Conda**: Run `conda doctor` for diagnostics

---

*Created February 2026 | Compatible with Ubuntu 20.04+, Python 3.10+*
