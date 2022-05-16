from ufl import *
from ufl.checks import is_globally_constant
from dune.mmesh import trace, normals
from mmdgpy.dg.mmdg2 import MMDG2

class MMDG1(MMDG2):
    """! A discontinuous Galerkin scheme in d=2,3 dimensions with a reduced
        fracture of variable aperture. Aperture gradients are included.
    """

    def __init__(self, dim, order, gridfile, problem, mu0, xi=2./3., \
     contortion=True, trafo=None):
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
        super().__init__(dim, order, gridfile, problem, mu0, xi=xi, \
         contortion=contortion, trafo=trafo)

        if not ( is_globally_constant(problem.d1(self.x_gamma)) and \
         is_globally_constant(problem.d2(self.x_gamma)) ):

            inormal = normals(self.igridview)

            self.b_gamma -= trace(self.ph, self.igridview)('+') * \
             dot(grad(problem.d_i(self.x_gamma, inormal)), \
             problem.k_gamma(self.x_gamma) * grad(self.phi_gamma)) * dx
            self.b_gamma -= trace(self.ph, self.igridview)('-') * \
             dot(grad(problem.d_i(self.x_gamma, -inormal)), \
             problem.k_gamma(self.x_gamma) * grad(self.phi_gamma)) * dx

            trp = trace(self.ph, self.igridview, restrictTo='+')
            trm = trace(self.ph, self.igridview, restrictTo='-')

            self.b_gamma += \
             jump(self.phi_gamma) * dot(problem.k_gamma(self.x_gamma) * \
             0.5 * trp('+') * \
             grad(problem.d_i(self.x_gamma, inormal('+')))('+') + \
             0.5 * trp('-') * \
             grad(problem.d_i(self.x_gamma, inormal('-')))('-'), \
             self.n_gamma('+')) * dS
            self.b_gamma += \
             jump(self.phi_gamma) * dot(problem.k_gamma(self.x_gamma) * \
             0.5 * trm('+') * \
             grad(problem.d_i(self.x_gamma, -inormal('+')))('+') + \
             0.5 * trm('-') * \
             grad(problem.d_i(self.x_gamma, -inormal('-')))('-'), \
             self.n_gamma('+')) * dS

            if self.problem.boundary_dn_gamma(self.x_gamma) != 0:
                self.b_gamma += self.phi_gamma * trp \
                 * dot(grad(problem.d_i(self.x_gamma, inormal)), self.n_gamma) \
                 * self.problem.boundary_dn_gamma(self.x_gamma) * ds
                self.b_gamma += self.phi_gamma * trm \
                 * dot(grad(problem.d_i(self.x_gamma, -inormal)), self.n_gamma)\
                 * self.problem.boundary_dn_gamma(self.x_gamma) * ds
