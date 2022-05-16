from time import time
from os import mkdir
from os.path import exists
from mmdgpy.dg.dg import DG
from mmdgpy.dg.mmdg2 import MMDG2
from mmdgpy.dg.mmdg1 import MMDG1
from mmdgpy.problems.problemcomparison import ProblemComparison
from mmdgpy.grids.grids import create_resolved_grid, create_reduced_grid
from params_comparison import *

################################################################################

if not exists(grid_dir):
    mkdir(grid_dir)

if not exists(vtk_dir):
    mkdir(vtk_dir)

for d0 in d0_values:
    vtkfile = vtkprefix(d0)
    print("-------------------------------------------------------------------")
    print(vtkfile)

    problem = ProblemComparison(
     gd = lambda x, dm : gd(x, dm, d0),
     gn = lambda x, dm : gn(x, dm, d0),
     boundary_dn = lambda x, dm : boundary_dn(x, dm, d0),
     k = lambda x, dm : k(x, dm, d0),
     d1 = lambda x : d1_ufl([x[0], x[1], x[2] if dim==3 else 0], d0),
     d2 = lambda x : d2_ufl([x[0], x[1], x[2] if dim==3 else 0], d0),
     gd_gamma = lambda x : gd_gamma(x, d0),
     gn_gamma = lambda x : gn_gamma(x, d0),
     boundary_dn_gamma = lambda x : boundary_dn_gamma(x, d0),
     k_gamma = lambda x : k_gamma(x, d0),
     k_gamma_perp = lambda x : k_gamma_perp(x, d0) )

    ########

    start_time = time()

    create_resolved_grid(
     file = resolved_grid,
     d1 = lambda x : d1(x, d0),
     d2 = lambda x : d2(x, d0),
     h = h_fulldim(d0),
     hf = hf_fulldim(d0),
     dim = dim )

    create_reduced_grid(
     file = reduced_grid,
     h = h_reduced(d0),
     hf = hf_reduced(d0),
     dim = dim )

    print("\nCreated grids with a run time of {0:.2f} Seconds.\n".format(
     time() - start_time))

    ########

    start_time = time()

    dg = DG(dim, order, resolved_grid, problem, mu0, storage=storage)
    dg.solve(solver_dg)
    dg.write_vtk(vtkfile + '_dg')

    print("\nSolved scheme \"dg\" with a run time of {0:.2f}"
     " Seconds.\n".format(time() - start_time))

    ########

    start_time = time()

    mmdg2_notrafo = \
     MMDG2(dim, order, reduced_grid, problem, mu0, xi, contortion=False)
    mmdg2_notrafo.solve(
     solver_mmdg, iter, tol, f_tol, eps, accelerate, verbose)
    mmdg2_notrafo.write_vtk(vtkfile + '_mmdg2_notrafo')

    print("\nSolved scheme \"mmdg2_notrafo\" with a run time of {0:.2f}"
     " Seconds.\n".format(time() - start_time))

    #######

    start_time = time()

    mmdg2_trafo = \
     MMDG2(dim, order, reduced_grid, problem, mu0, xi, contortion=True)
    mmdg2_trafo.solve(
     solver_mmdg, iter, tol, f_tol, eps, accelerate, verbose)
    mmdg2_trafo.write_vtk(vtkfile + '_mmdg2_trafo')

    print("\nSolved scheme \"mmdg2_trafo\" with a run time of {0:.2f}"
     " Seconds.\n".format(time() - start_time))

    ########

    start_time = time()

    mmdg1_notrafo = \
     MMDG1(dim, order, reduced_grid, problem, mu0, xi, contortion=False)
    mmdg1_notrafo.solve(
     solver_mmdg, iter, tol, f_tol, eps, accelerate, verbose)
    mmdg1_notrafo.write_vtk(vtkfile + '_mmdg1_notrafo')

    print("\nSolved scheme \"mmdg1_notrafo\" with a run time of {0:.2f}"
     " Seconds.\n".format(time() - start_time))

    ########

    start_time = time()

    mmdg1_trafo = \
     MMDG1(dim, order, reduced_grid, problem, mu0, xi, contortion=True)
    mmdg1_trafo.solve(
     solver_mmdg, iter, tol, f_tol, eps, accelerate, verbose)
    mmdg1_trafo.write_vtk(vtkfile + '_mmdg1_trafo')

    print("\nSolved scheme \"mmdg1_trafo\" with a run time of {0:.2f}"
     " Seconds.\n".format(time() - start_time))
