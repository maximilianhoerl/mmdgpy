import numpy as np
from time import time
from os import mkdir
from os.path import exists
from params_dg import *
from mmdgpy.dg.dg import DG
from mmdgpy.grids.grids import create_dgf_grid

################################################################################

if not exists(grid_dir):
    mkdir(grid_dir)

if not exists(vtk_dir):
    mkdir(vtk_dir)

errors = []
eocs = []

for i in range(len(n_values)):
    print(i)
    start_time = time()

    create_dgf_grid(gridfile, n_values[i], dim)

    dg = DG(dim, order, gridfile, problem, mu0, contortion, trafo, storage)
    dg.solve(solver)
    dg.write_vtk(vtkfile, i)

    errors += [dg.get_error(order=10)]

    if i > 0:
        eocs += [ np.log( errors[i-1] / errors[i] ) \
         / np.log( n_values[i] / n_values[i-1] ) ]

    print("Finished with a total run time of {0:.2f} Seconds.\n".format( \
     time() - start_time))

print("errors:", errors)
print("EOC:", eocs)
