from ufl import *
from dune.ufl import Constant
from dune.fem.scheme import galerkin
from dune.mmesh import iterativeSolve, monolithicSolve, skeleton, trace
from dune.fem.space import dglagrange
from dune.fem.function import integrate
from mmdgpy.dg.dg import DG

class MMDG2(DG):
    """! A discontinuous Galerkin scheme in d=2,3 dimensions with a reduced
        fracture. Aperture gradients are negelected.
    """
    # Attributes:
    ## @var igridview
    # The interface grid view.
    ## @var ispace
    # The interface dG space.
    ## @var x_gamma
    # The interface spatial coordinate.
    ## @var phi_gamma
    # The interface test function.
    ## @var n_gamma
    # The interface facet normal.
    ## @var ph_gamma
    # The interface numerical solution.
    ## @var b_gamma
    # The bilinear form of interface contributions.
    ## @var l_gamma
    # The linear form of interface contributions.
    ## @var c_bulk
    # The bilinear form of coupling terms (with bulk test function).
    ## @var c_gamma
    # The bilinear form of coupling terms (with interface test function).

    def __init__(self, dim, order, gridfile, problem, mu0, xi=2./3., \
     contortion=False, trafo=None):
        """! The constructor.

            @param dim  (int) The bulk dimension dim=2,3.
            @param order  (int) The order of accuracy >= 1 of the dG method.
            @param gridfile  A .msh grid file.
            @param problem  A problem class implementing the interface
                            MMDGProblem.
            @param mu0  (int) A penalty parameter that must be chosen
                        sufficiently large.
            @param xi  (optional) A coupling parameter. The default value is
                       2/3.
            @param contortion  (bool, optional) A boolean that indicates whether
                               the domain is to be contorted according to a
                               given transformation. The default value is False.
            @param trafo  (optional) A transformation function depending on the
                          spatial coordinate and the domain marker that
                          determines the contortion of the domain. By default
                          the transformation of the given problem is used.
        """

        super().__init__(dim, order, gridfile, problem, mu0, contortion, trafo)

        self.igridview = self.omega.hierarchicalGrid.interfaceGrid
        self.ispace = dglagrange(self.igridview, order=order)
        self.x_gamma = SpatialCoordinate(self.ispace)

        mu_gamma = Constant(mu0, name="mu0_gamma") * (order + 1) \
         * (order + dim - 1) / MaxFacetEdgeLength(self.ispace.cell())

        p_gamma = TrialFunction(self.ispace)
        self.phi_gamma = TestFunction(self.ispace)
        self.n_gamma = FacetNormal(self.ispace.cell())
        self.ph_gamma = self.ispace.interpolate(0, name="pressureGamma")

        self.b_bulk -= self.chi_gamma * self.iterms * dS

        self.b_gamma = dot(self.problem.k_gamma(self.x_gamma) \
         * grad(self.problem.d(self.x_gamma) * p_gamma), grad(self.phi_gamma)) \
         * dx
        self.b_gamma += mu_gamma * jump(p_gamma) * jump(self.phi_gamma) * dS
        self.b_gamma -= \
         jump(self.phi_gamma) * dot(avg(self.problem.k_gamma(self.x_gamma) \
         * grad(self.problem.d(self.x_gamma) * p_gamma)), self.n_gamma('+')) \
         * dS
        self.b_gamma -= jump(self.problem.d(self.x_gamma) * p_gamma) \
         * dot(avg(self.problem.k_gamma(self.x_gamma) * grad(self.phi_gamma)), \
         self.n_gamma('+')) * dS

        if self.problem.boundary_dn_gamma(self.x_gamma) != 0:
            self.b_gamma += mu_gamma * p_gamma * self.phi_gamma \
             * self.problem.boundary_dn_gamma(self.x_gamma) * ds
            self.b_gamma -= \
             self.phi_gamma * dot(self.problem.k_gamma(self.x_gamma) \
             * grad(self.problem.d(self.x_gamma) * p_gamma), self.n_gamma) \
             * self.problem.boundary_dn_gamma(self.x_gamma) * ds
            self.b_gamma -= self.problem.d(self.x_gamma) * p_gamma \
             * dot(self.problem.k_gamma(self.x_gamma) * grad(self.phi_gamma), \
             self.n_gamma) * self.problem.boundary_dn_gamma(self.x_gamma) * ds

        self.l_gamma = 0.

        if self.problem.q_gamma(self.x_gamma) != 0:
            self.l_gamma += \
             self.problem.q_gamma(self.x_gamma) * self.phi_gamma * dx

        if self.problem.gd_gamma(self.x_gamma) != 0 and \
         self.problem.boundary_dn_gamma(self.x_gamma) != 0:
            self.l_gamma += mu_gamma * self.problem.gd_gamma(self.x_gamma) \
             * self.phi_gamma * self.problem.boundary_dn_gamma(self.x_gamma) * ds
            self.l_gamma -= \
             self.problem.d(self.x_gamma) * self.problem.gd_gamma(self.x_gamma)\
             * dot(self.problem.k_gamma(self.x_gamma) * grad(self.phi_gamma), \
             self.n_gamma) * self.problem.boundary_dn_gamma(self.x_gamma) * ds

        if self.problem.boundary_dn_gamma(self.x_gamma) != 1 and \
         self.problem.gn_gamma(self.x_gamma) != 0:
            self.l_gamma += \
             self.problem.gn_gamma(self.x_gamma) * self.phi_gamma \
             * ( 1. - self.problem.boundary_dn_gamma(self.x_gamma) ) * ds

        alpha = problem.k_gamma_perp(self.x) / self.problem.d(self.x)('+')
        alpha_gamma = \
         self.problem.k_gamma_perp(self.x_gamma) / self.problem.d(self.x_gamma)
        beta = 4 * alpha / ( 2 * xi - 1 )
        beta_gamma = 4 * alpha_gamma / ( 2 * xi - 1 )

        if contortion:
            grad_d_plus = self.problem.grad_d_i(self.x, self.n('+'))('+')
            grad_d_minus = self.problem.grad_d_i(self.x, self.n('-'))('+')

            self.c_bulk = self.chi_gamma * self.phi('-') * \
             ( 0.5 * beta * ( avg(self.p) - \
             skeleton(self.ph_gamma, grid=self.omega)('+') ) - \
             self.problem.leftright(self.n('-'),-1,1) * alpha * jump(self.p) ) \
             / sqrt( 1. + dot(grad_d_minus, grad_d_minus) ) * dS
            self.c_bulk += self.chi_gamma * self.phi('+') * \
             ( 0.5 * beta * ( avg(self.p) - \
             skeleton(self.ph_gamma, grid=self.omega)('+') ) - \
             self.problem.leftright(self.n('+'),-1,1) * alpha * jump(self.p) ) \
              / sqrt( 1. + dot(grad_d_plus, grad_d_plus) ) * dS

        else:
            self.c_bulk = \
             alpha * jump(self.p) * jump(self.phi) * self.chi_gamma * dS
            self.c_bulk += beta * \
             ( avg(self.p) - skeleton(self.ph_gamma, grid=self.omega)('+') ) \
             * avg(self.phi) * self.chi_gamma * dS

        self.c_gamma = beta_gamma * \
         ( p_gamma - avg(trace(self.ph, self.igridview)) ) * self.phi_gamma * dx


    def solve(self, solver='monolithic', iter=100, tol=1e-8, f_tol=1e-8, \
     eps=1e-8, accelerate=False, verbose=True):
        """! Solves the discontinuous Galerkin problem.

            @param solver  (optional) One of the following solvers. The default
                           solver is 'monolithic'.
                           - 'monolithic' for dune.mmesh.monolithicSolve,
                           - 'iterative' for dune.mmesh.iterativeSolve,
                           - 'bulk_only' to only to solve the bulk problem with
                             exact solution for the fracture,
                           - 'fracture_only' to only solve the fracture problem
                             with exact solution for the bulk.

            @param iter  (optional) The maximum number of iterations. Only
                         relevant for 'monolithicSolve' and 'iterativeSolve'.
                         The default value is 100.
            @param tol  (optional) The objective residual of the iteration step
                        in the infinity norm. Only relevant for
                        'monolithicSolve' and 'iterativeSolve'. The defaut value
                        is 1e-8.
            @param f_tol  (optional) The objective residual of the function
                          value in the infinity norm. Only relevant for
                          'monolithicSolve'. The default value is 1e-8.
            @param eps  (optional) The step size for finite differences. Only
                        relevant for 'monolithicSolve'. The default value is
                        1e-8.
            @param accelerate  (bool, optional): Boolean that indicates whether
                               to use a vector formulation of the fix-point
                               iteration. Only relevant for 'iterativeSolve'.
                               The default value is False.
            @param verbose  (bool, optional): Boolean that indicates whether the
                            residuum is printed for each iteration. Only
                            relevant for 'monolithicSolve' and 'iterativeSolve'.
                            The default value is True.
        """
        scheme = galerkin([self.b_bulk + self.c_bulk == self.l_bulk], \
         solver=("suitesparse","umfpack"))
        scheme_gamma = galerkin([self.b_gamma + self.c_gamma == self.l_gamma], \
         solver=("suitesparse","umfpack"))

        if solver == 'monolithic':
            monolithicSolve(schemes=(scheme, scheme_gamma), \
             targets=(self.ph, self.ph_gamma), iter=iter, tol=tol, f_tol=f_tol,\
             eps=eps, verbose=verbose)

        elif solver == 'iterative':
            iterativeSolve(schemes=(scheme, scheme_gamma), \
             targets=(self.ph, self.ph_gamma), iter=iter, tol=tol, \
             verbose=verbose, accelerate=accelerate)

        elif solver == 'bulk_only':
            self.ph_gamma.interpolate(self.problem.p_gamma(self.x_gamma))
            scheme.solve(target=self.ph)

        elif solver == 'fracture_only':
            self.ph.interpolate(self.problem.p(self.x, self.dm))
            scheme_gamma.solve(target=self.ph_gamma)

        else:
            raise ValueError('Unknown solver: ' + solver)


    def write_vtk(self, filename="pressure", filenumber=0):
        """! Writes out the solution to VTk. Remember to call solve() first.

            @param filename  (optional) A filename. The default value is
                             'pressure'.
            @param filenumber  (int, optional) A file number if a series of
                               problems is solved. The defaut value is 0.
        """
        super().write_vtk(filename, filenumber)

        pointdata = {"p_Gamma": self.ph_gamma,
         "d1": self.problem.d1(self.x_gamma),
         "d2": self.problem.d2(self.x_gamma),
         "d": self.problem.d(self.x_gamma),
         "grad_d1": self.problem.grad_d1(self.x_gamma),
         "grad_d2": self.problem.grad_d2(self.x_gamma)}

        try:
            pointdata.update(\
             {"exact_gamma": self.problem.p_gamma(self.x_gamma), \
             "error_gamma": self.ph_gamma - self.problem.p_gamma(self.x_gamma)})
        except:
            pass

        self.igridview.writeVTK(filename + "Gamma" + str(filenumber), \
         pointdata=pointdata, nonconforming=True, \
         subsampling=self.ispace.order-1)


    def get_error(self, order):
        """! Returns the L2-error for the bulk, fracture and total problem.
            Requires that an exact solution is implemented for the given
            problem.

            @param order  (int) The order of integration.
        """
        error_bulk = super().get_error(order)
        error_gamma = integrate(self.igridview, \
         (self.ph_gamma - self.problem.p_gamma(self.x_gamma)) ** 2, order=order)
        error_total = sqrt(error_bulk ** 2 + error_gamma)
        error_gamma = sqrt(error_gamma)

        return error_bulk, error_gamma, error_total
