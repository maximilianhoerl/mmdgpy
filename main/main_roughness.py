import matplotlib.pyplot as plt
from time import time
from os import mkdir
from os.path import exists
import numpy as np
from mmdgpy.dg.dg import DG
from mmdgpy.dg.mmdg2 import MMDG2
from mmdgpy.dg.mmdg1 import MMDG1
from mmdgpy.problems.problemroughness import ProblemRoughness
from mmdgpy.grids.grids import create_reduced_grid, create_resolved_grid
from mmdgpy.grids.random import get_gaussian_aperture
from params_roughness import *

################################################################################

if not exists(grid_dir):
    mkdir(grid_dir)

if not exists(vtk_dir):
    mkdir(vtk_dir)

d1, d2 = get_gaussian_aperture(
    dim, mu, cov, hf_gaussian, dmin, join(vtk_dir, "aperture.npz")
)

npzfile = np.load(join(vtk_dir, "aperture.npz"))
z1, z2 = npzfile["d1"], npzfile["d2"]
d1mean = np.mean(z1)
d2mean = np.mean(z2)

y = np.linspace(0, 1, 10001)
plt.plot(0.5 - d1(y), y, "k-", 0.5 + d2(y), y, "k-")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.savefig(join(vtk_dir, "aperture.pdf"), bbox_inches="tight")

########

problem = ProblemRoughness(
    gd,
    gn,
    boundary_dn,
    k,
    lambda x: d1(x[1]),
    lambda x: d2(x[1]),
    gd_gamma,
    gn_gamma,
    boundary_dn_gamma,
    k_gamma,
    k_gamma_perp,
    q,
    q_gamma,
)

problem_dmean = ProblemRoughness(
    gd,
    gn,
    boundary_dn,
    k,
    lambda x: d1mean,
    lambda x: d2mean,
    gd_gamma,
    gn_gamma,
    boundary_dn_gamma,
    k_gamma,
    k_gamma_perp,
    q,
    q_gamma,
)

########

create_resolved_grid(
    file=resolved_grid,
    d1=lambda x: d1(x[1]),
    d2=lambda x: d2(x[1]),
    h=h_fulldim,
    hf=hf_fulldim,
    dim=dim,
)

create_reduced_grid(file=reduced_grid, h=h_reduced, hf=hf_reduced, dim=dim)

########

start_time = time()

dg = DG(dim, order, resolved_grid, problem, mu0, storage=storage)
dg.solve(solver_dg)
dg.write_vtk(vtkprefix + "_dg")

print(
    '\nSolved scheme "dg" with a run time of {0:.2f}'
    " Seconds.\n".format(time() - start_time)
)

########

start_time = time()

mmdg2_notrafo = MMDG2(dim, order, reduced_grid, problem, mu0, xi, contortion=False)
mmdg2_notrafo.solve(solver_mmdg, iter, tol, f_tol, eps, accelerate, verbose)
mmdg2_notrafo.write_vtk(vtkprefix + "_mmdg2_notrafo")

print(
    '\nSolved scheme "mmdg2_notrafo" with a run time of {0:.2f}'
    " Seconds.\n".format(time() - start_time)
)

########

start_time = time()

mmdg2_trafo = MMDG2(dim, order, reduced_grid, problem, mu0, xi, contortion=True)
mmdg2_trafo.solve(solver_mmdg, iter, tol, f_tol, eps, accelerate, verbose)
mmdg2_trafo.write_vtk(vtkprefix + "_mmdg2_trafo")

print(
    '\nSolved scheme "mmdg2_trafo" with a run time of {0:.2f}'
    " Seconds.\n".format(time() - start_time)
)

########

start_time = time()

mmdg1_notrafo = MMDG1(dim, order, reduced_grid, problem, mu0, xi, contortion=False)
mmdg1_notrafo.solve(solver_mmdg, iter, tol, f_tol, eps, accelerate, verbose)
mmdg1_notrafo.write_vtk(vtkprefix + "_mmdg1_notrafo")

print(
    '\nSolved scheme "mmdg1_notrafo" with a run time of {0:.2f}'
    " Seconds.\n".format(time() - start_time)
)

########

start_time = time()

mmdg1_trafo = MMDG1(dim, order, reduced_grid, problem, mu0, xi, contortion=True)
mmdg1_trafo.solve(solver_mmdg, iter, tol, f_tol, eps, accelerate, verbose)
mmdg1_trafo.write_vtk(vtkprefix + "_mmdg1_trafo")

print(
    '\nSolved scheme "mmdg1_trafo" with a run time of {0:.2f}'
    " Seconds.\n".format(time() - start_time)
)

########

start_time = time()

mmdg2_dmean_notrafo = MMDG2(
    dim, order, reduced_grid, problem_dmean, mu0, xi, contortion=False
)
mmdg2_dmean_notrafo.solve(solver_mmdg, iter, tol, f_tol, eps, accelerate, verbose)
mmdg2_dmean_notrafo.write_vtk(vtkprefix + "_mmdg2_dmean_notrafo")

print(
    '\nSolved scheme "mmdg2_dmean_notrafo" with a run time of {0:.2f}'
    " Seconds.\n".format(time() - start_time)
)

########

start_time = time()

mmdg2_dmean_trafo = MMDG2(
    dim, order, reduced_grid, problem_dmean, mu0, xi, contortion=True
)
mmdg2_dmean_trafo.solve(solver_mmdg, iter, tol, f_tol, eps, accelerate, verbose)
mmdg2_dmean_trafo.write_vtk(vtkprefix + "_mmdg2_dmean_trafo")

print(
    '\nSolved scheme "mmdg2_dmean_trafo" with a run time of {0:.2f}'
    " Seconds.\n".format(time() - start_time)
)

########
