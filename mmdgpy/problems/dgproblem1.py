from ufl import *
from mmdgpy.problems.dgproblem import DGProblem

class DGProblem1(DGProblem):
    """ A simple test problem without fracture.
    """

    def p(self, x, dm):
        """ The exact solution p(x) = x*x.

            :param x: The spatial coordinate.
            :param dm: A domain marker.
        """
        return sin(2. * pi * x[0]) * cos(2. * pi * x[1])


    def q(self, x, dm):
        """ The source term q.

            :param x: The spatial coordinate.
            :param dm: A domain marker.
        """
        return -div( self.k(x,dm) * grad(self.p(x,dm)) )


    def gd(self, x, dm):
        """ The pressure at the boundary (Dirichlet condition).

            :param x: The spatial coordinate.
            :param dm: A domain marker.
        """
        return self.p(x, dm)


    def gn(self, x, dm):
        """ The velocity at the boundary of the bulk domain (Neumann
            condition).

            :param x: The spatial coordinate.
            :param dm: A domain marker.
        """
        return 2. * pi * ( 1. + x[1] * x[1] ) * sin(2. * pi * x[1])


    def boundary_dn(self, x, dm):
        """! Returns 1 for a Dirichlet condition and 0 for a Neumann condition
            on the boundary of the bulk domain.

            :param x: The spatial coordinate.
            :param dm: A domain marker.
        """
        return conditional(x[1] > 1. - 1e-12, 0, 1)


    def k(self, x, dm):
        """ The permeability matrix k = diag( 1 + x(i)^2 ).

            :param x: The spatial coordinate.
            :param dm: A domain marker.
        """
        dim = x.ufl_shape[0]

        k = []
        for i in range(dim):
            row = [0] * dim
            row[i] = 1 + x[i] * x[i]
            k += [row]

        return as_matrix(k)
