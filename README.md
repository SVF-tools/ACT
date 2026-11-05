# Abstract Constraint Transformation (ACT)

A testing and verification framework for AI models based on neural networks, built on a three-tier architecture (front-end, back-end, and pipeline), with native PyTorch support and an ACT intermediate representation (IR) that enables refinement-based precision and supports diverse model architectures, input formats, and specification types.

**Online (v0.1):** ACT now supports online use through ACT-GUI at [https://act-online.au](https://act-online.au).

## Quick Start


## 0. Preparation
Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions) and create a running environment.

```
conda env create -f environment.yml    # Install required lib packages to run ACT
conda activate act-py312 # Activate an environment (python-3.12)  # Activate the environment 
```

## 1. Clone repository
```
git clone --recursive https://github.com/SVF-tools/ACT.git
cd ACT
```

## 2. Apply and download the [Gurobi license](https://www.gurobi.com/academia/academic-program-and-licenses/) (Optional for MILP optimization)
```
cp /path/to/your/gurobi.lic ./modules/gurobi/gurobi.lic  # put gurobi.lic file in ./modules/gurobi/ directory
```

## 3. Run ACT phases
ACT uses a modular three-tier architecture. For detailed documentation on each module, see the [ACT README](act/README.md).

```
# Pipeline CLI (testing and integration)
python -m act.pipeline --help

# Front-End CLI (data and model processing)
python -m act.front_end --help

# Back-End CLI (core verification and network generation)
python -m act.back_end --help
```

## 4. Small Jupyter notebook demos
- [ACT Fuzzer example](https://github.com/SVF-tools/ACT/blob/main/ipynb/vnnlib_fuzzer.ipynb)
- [ACT Verifier example](https://github.com/SVF-tools/ACT/blob/main/ipynb/vnnlib_verifier.ipynb)
- [More](https://github.com/SVF-tools/ACT/tree/main/ipynb)

### Pubs and Docs
- Kaijie Liu and Yulei Sui. [Detecting Unsoundness in Neural Network Verifiers via Concrete–Abstract Consistency](https://openreview.net/forum?id=6vjnMGdx5i). ACM/IEEE International Conference on AI-Powered Software Engineering (AIware 2026)
- Guanqin Zhang and Yulei Sui. [Tensor-Based Batch Fuzzing with Adaptive Perturbation Scaling for Deep Neural Networks](https://arxiv.org/abs/2606.25239). 41st IEEE/ACM International Conference on Automated Software Engineering (ASE 2026)

### License
ACT is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

### Acknowledgements
This project was developed with the assistance of GitHub Copilot to enhance code readability and efficiency. AI-generated suggestions were reviewed and tested by the contributors before inclusion.
