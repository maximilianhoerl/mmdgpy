from ufl import cos, pi
from dune.ufl import Constant
from os.path import join

from mmdgpy.problems.mmdgproblem1 import MMDGProblem1
from mmdgpy.problems.mmdgproblem2 import MMDGProblem2
from mmdgpy.problems.mmdgproblem3 import MMDGProblem3
from mmdgpy.problems.mmdgproblem4 import MMDGProblem4
from mmdgpy.problems.mmdgproblem5 import MMDGProblem5

################################################################################
###  parameter file for test_mmdg.py  ##########################################
################################################################################

dim = 2
order = 1
mu0 = 10
xi = 2.0 / 3.0

use_mmdg1 = True
contortion = True

repeat = 5 if dim == 2 else 3

# directories and file names
grid_dir = "grids"
vtk_dir = "vtk"
gridfile = join(grid_dir, "vertical.msh")
vtkfile = join(vtk_dir, "pressure")

# parameters for solver
solver = "monolithic"
iter = 100
tol = 1e-8
f_tol = 1e-8
eps = 1e-8
accelerate = False
verbose = True

# problem parameters
d0 = Constant(1e-2, name="d0")
d_mean = 5e-3
d_diff = 2e-3
d2 = lambda x: d0 + 0.5 * d0 * cos(8 * pi * x[1])
d1 = lambda x: d0 * (x[1] + 1)

problem_number = 3

if problem_number == 1:
    problem = MMDGProblem1(d0)
elif problem_number == 2:
    problem = MMDGProblem2(d0)
elif problem_number == 3:
    problem = MMDGProblem3(d1, d2, xi, c=1)
elif problem_number == 4:
    problem = MMDGProblem4(d1, d2)
elif problem_number == 5:
    problem = MMDGProblem5(d_mean, d_diff)
else:
    raise ValueError(f"Invalid problem number {problem_number}")
