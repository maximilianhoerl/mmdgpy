import numpy as np
from time import time
from os import mkdir
from os.path import exists
from mmdgpy.test.params_mmdg import *
from ufl import *
from dune.ufl import Constant
from mmdgpy.dg.mmdg2 import MMDG2
from mmdgpy.dg.mmdg1 import MMDG1
from mmdgpy.grids.grids import create_reduced_grid

################################################################################

if not exists(grid_dir):
    mkdir(grid_dir)

if not exists(vtk_dir):
    mkdir(vtk_dir)

errors, errors_bulk, errors_gamma = [], [], []
eocs, eocs_bulk, eocs_gamma = [], [], []
i0 = 0

for i in range(i0, i0 + repeat):
    print(i)
    start_time = time()

    create_reduced_grid(gridfile, h=0.5*2**-i, hf=0.5*2**-i, dim=dim)

    if use_mmdg1:
        mmdg = MMDG1(dim, order, gridfile, problem, mu0, xi, contortion)
    else:
        mmdg = MMDG2(dim, order, gridfile, problem, mu0, xi, contortion)

    mmdg.solve(solver, iter, tol, f_tol, eps, accelerate, verbose)
    mmdg.write_vtk(vtkfile, i)

    err_bulk, err_gamma, err_total = mmdg.get_error(order)
    errors_bulk += [err_bulk]
    errors_gamma += [err_gamma]
    errors += [err_total]

    if i > i0:
        eocs_bulk += \
         [ np.log( errors_bulk[i-1] / errors_bulk[i] ) / np.log(2) ]
        eocs_gamma += \
         [ np.log( errors_gamma[i-1] / errors_gamma[i] ) / np.log(2) ]
        eocs += [ np.log( errors[i-1] / errors[i] ) / np.log(2) ]

    print("\nFinished with a total run time of {0:.2f} Seconds.\n".format( \
     time() - start_time))

print("errors (bulk):", errors_bulk)
print("EOC (bulk):", eocs_bulk)
print("errors (Gamma):", errors_gamma)
print("EOC (Gamma):", eocs_gamma)
print("errors (total):", errors)
print("EOC (total):", eocs)

assert abs(eocs[-1] - order - 1) < 1e-1
assert errors[-1] < 5e-3
