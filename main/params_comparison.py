import numpy as np
from os.path import join
from ufl import *
from dune.ufl import Constant

################################################################################
###  parameter file for main_comparison.py  ####################################
################################################################################

dim = 2     # dimension
order = 1   # order of dG space
mu0 = 10    # penalty parameter
xi = 2./3.  # coupling parameter

# parameters for solver of full-dimensional scheme
storage_dg = None
solver_dg = None

# parameters for solver of reduced schemes
storage_mmdg = 'istl'
solver_mmdg = 'monolithic'
iter = 100
tol = 1e10
f_tol = 1e-8
eps = 1e-8
parameters = {"newton.linear.verbose": "true"}
accelerate = False
verbose = True

# fracture parameters
tangential = True
symm = True
d0_values = np.logspace(-1, -3, 7) if dim==2 else np.logspace(-1, -2, 4)
d_ampl = lambda d0 : d0

# grid resolution
h_fulldim = lambda d0 : 3e-2 if dim==2 else 8e-2
hf_fulldim = \
 lambda d0 : np.minimum( 0.1 * d0 if dim==2 else 0.5 * d0, 0.5 * h_fulldim(d0) )
h_reduced = lambda d0 : 3e-2 if dim==2 else 8e-2
hf_reduced = lambda d0 : 8e-3 if dim==2 else 3e-2

# directories and file names
grid_dir = 'grids'
vtk_dir = 'vtk'
resolved_grid = join(grid_dir, 'resolved.msh')
reduced_grid = join(grid_dir, 'vertical.msh')
vtkprefix = \
 lambda d0 : join(vtk_dir, 'comparison_' + ( 'tang' if tangential else 'perp' )
 + '_' + ( 'sym' if symm else 'asym' ) + f'_dim{dim}' + f'_{d0:0.1e}')

################################################################################

d1 = lambda x, d0 : d0 + 0.5 * d_ampl(d0) * ( np.sin(8 * np.pi * x[1])
 + np.sin(8 * np.pi * x[2]) )
d1_ufl = lambda x, d0 : d0 + 0.5 * d_ampl(d0) * ( sin(8 * pi * x[1])
 + sin(8 * pi * x[2]) )

if symm:
    d2 = lambda x, d0 : d0 + 0.5 * d_ampl(d0) * ( np.sin(8 * np.pi * x[1])
     + np.sin(8 * np.pi * x[2]) )
    d2_ufl = lambda x, d0 : d0 + 0.5 * d_ampl(d0) * ( sin(8 * pi * x[1])
     + sin(8 * pi * x[2]) )
else:
    d2 = lambda x, d0 : d0 - 0.5 * d_ampl(d0) * ( np.sin(8 * np.pi * x[1])
     + np.sin(8 * np.pi * x[2]) )
    d2_ufl = lambda x, d0 : d0 - 0.5 * d_ampl(d0) * ( sin(8 * pi * x[1])
     + sin(8 * pi * x[2]) )

isFractureDomain = lambda dm : 0.5 * dm * (dm - 1)

if tangential:
    gd = lambda x, dm, d0 : 4 * x[0] * ( 1. - x[0] ) * ( 1. - x[1] )
    k = lambda x, dm, d0 : Identity(dim) + isFractureDomain(dm) * Identity(dim)
    gd_gamma = \
     lambda x, d0 : conditional(x[1] < 1e-12, 1, 0) * ( 1. - (4./3.) * d0**2 )
    k_gamma = lambda x, d0 : 2. * Identity(dim)
    k_gamma_perp = lambda x, d0 : 2.

else:
    gd = lambda x, dm, d0 : 1. - x[0]
    k = lambda x, dm, d0 : \
     Identity(dim) - 0.5 * isFractureDomain(dm) * Identity(dim)
    gd_gamma = lambda x, d0 : 0.5
    k_gamma = lambda x, d0 : 0.5 * Identity(dim)
    k_gamma_perp = lambda x, d0 : 0.5

gn = lambda x, dm, d0 : 0.
boundary_dn = lambda x, dm, d0 : 1.
gn_gamma = lambda x, d0 : 0.
boundary_dn_gamma = lambda x, d0 : 1.

################################################################################
