from mmdgpy.problems.dgproblem1 import DGProblem1
from mmdgpy.problems.dgproblem2 import DGProblem2

import numpy as np
from os.path import join

################################################################################
###  parameter file for test_dg.py  ############################################
################################################################################

dim = 2
order = 1
mu0 = 10
contortion = False
trafo = None

# series of number of grid elements per directions
n_values = np.unique(
    (np.logspace(0, 2.2, 8) if dim == 2 else np.logspace(0, 1, 5)).astype(int)
)

# problem parameters
d_mean = 5e-3
d_diff = 2e-3
problem_number = 1

if problem_number == 1:
    problem = DGProblem1()
elif problem_number == 2:
    problem = DGProblem2(d_mean, d_diff)
else:
    raise ValueError(f"Invalid problem number {problem_number}")

# parameters for solver
storage = None
solver = None

# directories and file names
grid_dir = "grids"
vtk_dir = "vtk"
gridfile = join(grid_dir, "cube.dgf")
vtkfile = join(vtk_dir, "pressure")
