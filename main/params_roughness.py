import numpy as np
from os.path import join
from ufl import *
from dune.ufl import Constant

################################################################################

dim = 2     # dimension
order = 1   # order of dG space
mu0 = 10  # penalty parameter
xi = 2./3.  # coupling parameter

# parameters for solver of full-dimensional scheme
storage = None
solver_dg = None

# parameters for solver of reduced schemes
solver_mmdg = 'monolithic'
iter = 5000
tol = 1e-8
f_tol = 1e-8
eps = 1e-8
accelerate = True
verbose = True

# fracture parameters
d0 = 5e-2
sigma2 = 5e-4
l = 0.1
H = 0.8
mu = d0
cov = lambda r : sigma2 * np.exp( -( r / l ) ** (2 * H) )
dmin = 1e-6
hf_gaussian = 2e-3

# grid resolution
# NOTE: schemes with contortion = True require h_reduced = hf_reduced such that
#       1 / h_reduced is an integer
h_fulldim = 3e-2
hf_fulldim = np.amin([ hf_gaussian, 0.1 * d0, 0.5 * h_fulldim ])
h_reduced = hf_gaussian
hf_reduced = h_reduced

# directories and file names
grid_dir = 'grids'
vtk_dir = 'vtk'
resolved_grid = join(grid_dir, 'resolved.msh')
reduced_grid = join(grid_dir, 'vertical.msh')
vtkprefix = join(vtk_dir, 'roughness' + f'_dim{dim}')

# problem parameters
isFractureDomain = lambda dm : 0.5 * dm * (dm - 1)
kf = 1e-2

gd = lambda x, dm : 1 - x[0]
k = lambda x, dm : (1 - isFractureDomain(dm)) * Identity(dim) \
 + isFractureDomain(dm) * kf * Identity(dim)
gd_gamma = lambda x : 1 - x[1]
k_gamma = lambda x : kf * Identity(dim)
k_gamma_perp = lambda x : kf
gn = lambda x, dm : 0.
boundary_dn = lambda x, dm : \
 conditional(x[0] < 1e-12, 1, 0) + conditional(x[0] > 1. - 1e-12, 1, 0)
gn_gamma = lambda x : 0.
boundary_dn_gamma = lambda x : 0.
q = lambda x, dm : 200 * ( tanh( 200 * ( 0.025 - sqrt(dot(x, x)) ) ) \
 - tanh( 200 * ( 0.025 - sqrt((x[0] - 1)**2 + (x[1] - 1)**2) ) ) )
q_gamma = lambda x : 0.
