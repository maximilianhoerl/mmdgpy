# `mmdgpy`

The `mmdgpy` Python package is an implementation of interior penalty discontinuous Galerkin (dG) schemes in 2D and 3D for flow in porous media governed by Darcy's law.

It provides a full-dimensional dG scheme for bulk problems and various mixed-dimensional dG schemes for problems in fractured porous media.
Fractures are represented by one-codimensional interfaces.
A spatially varying fracture aperture is possible.

The `mmdgpy` module is based on <nobr>`DUNE` [1]</nobr>. In particular, it relies on <nobr>`DUNE-MMesh` [2]</nobr> and <nobr>`DUNE-FEM` [4]</nobr>.

For details, we refer to [3,5].

## Installation

1. Create and activate a virtual environment (recommended).
````
python3 -m venv dune-env
source dune-env/bin/activate
````

2. Clone the repository and install `mmdgpy` and its dependencies using `pip`.
````
git clone https://github.com/maximilianhoerl/mmdgpy.git
pip install -e mmdgpy
````

## Schemes

The functions and coefficients describing a specific flow problem are to be defined in a problem class inheriting the abstract class `DGProblem` or `MMDGProblem`.

Given a grid file and a problem object, a dG scheme is created as an instance of one of the following classes:
- `DG`: a full-dimensional scheme, generally without fracture,
- `MMDG1`: a mixed-dimensional scheme corresponding to the reduced <nobr>model I</nobr> (if `contortion=True`) or the reduced <nobr>model I-R</nobr> (if `contortion=False`) <nobr>in [3,5]</nobr>,
- `MMDG2`: a mixed-dimensional scheme corresponding to the reduced <nobr>model II</nobr> (if `contortion=True`) or the reduced <nobr>model II-R</nobr> (if `contortion=False`) <nobr>in [3,5]</nobr>.

## Scripts

The directories `main` and `test` contain a few useful scripts that demonstrate how to use `mmdgpy` and can be used for testing purposes.

- `test_dg.py`: Tests the full-dimensional `DG` scheme under grid refinement with corresponding parameter file `params_dg.py`.
- `test_mmdg.py`: Tests the mixed-dimensional scheme `MMDG1` or `MMDG2` under grid refinement with corresponding parameter file `params_mmdg.py`.
- `main_comparison.py`: Solves the same flow problem with fracture using the different reduced schemes and calculates a full-dimensional reference solution. The corresponding parameter file is `params_comparison.py`.
- `main_roughness.py`: Solves a flow problem with random Gaussian fracture using different reduced schemes and calculates a full-dimensional reference solution. The corresponding parameter file is `params_roughness.py`.
- `plot_comparison.py`: Visualizes the simulation results obtained with the script `main_comparison.py` or `main_roughness.py`. To be executed with `pvpython` (`ParaView` version 5.10.1).

## Documentation

The `sphinx` documentation can be build inside the `docs` directory.
````
cd mmdgpy/docs
make html
````

## Literature
<a id="1">[1]</a>  P. Bastian et al. *"The Dune framework: Basic concepts and recent developments."* In: Comput. Math. Appl. 81 (2021), pp. 75-112.

<a id="2">[2]</a>  S. Burbulla, A. Dedner, M. Hörl, and C. Rohde. *"Dune-MMesh: The Dune Grid Module for Moving Interfaces."* In: J. Open Source Softw. 7 (74) (2022), 3959.

<a id="3">[3]</a>  S. Burbulla, M. Hörl, and C. Rohde. *"Flow in Porous Media with Fractures of Varying Aperture."* In: SIAM J. Sci. Comput. 45(4) (2023).

<a id="4">[4]</a>  A. Dedner et al. *"A generic interface for parallel and adaptive discretization schemes: abstraction principles and the DUNE-FEM module."* In: Comput. 90 (2010), pp. 165-196.

<a id="5">[5]</a>  M. Hörl. *"Flow in Porous Media with Fractures of Varying Aperture."* Master's Thesis. University of Stuttgart, 2022.
