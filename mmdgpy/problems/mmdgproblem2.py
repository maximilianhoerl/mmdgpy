from ufl import *
from dune.ufl import Constant
from mmdgpy.problems.mmdgproblemvertical import MMDGProblemVertical

class MMDGProblem2(MMDGProblemVertical):
    """! A test problem for a fracture with variable aperture. For xi=1, the
        problem is an exact solution of the model in MMDG2 (no contortion).
        The bulk domain is Omega=[0,1]^dim with fracture Gamma={x_0 = 0.5}.
    """
    # Attributes:
    ## @var dmax
    # The maximum aperture of the fracture.

    def __init__(self, dmax):
        """! The constructor.

            @param dmax  The maximum aperture of the fracture.
        """
        self.dmax = Constant(dmax, name="dmax")


    def p(self, x, dm):
        """! The exact solution p for the bulk domain.

            p1 = cosh(sqrt(2)*(2*x_1-1)) * cos(4*x_2),
            p2 = cosh(sqrt(2)*(2*x_1+1)) * cos(4*x_2).

            @param x  The spatial coordinate.
            @param dm  A domain marker.
        """
        return cos(4 * x[1]) * ( (1 - dm) * cosh(sqrt(2) * (2 * x[0] - 1)) \
         + dm * cosh(sqrt(2) * (2 * x[0] + 1)) )


    def q(self, x, dm):
        """! The source term q for the bulk domain.

            @param x  The spatial coordinate.
            @param dm  A domain marker.
        """
        return 8 * self.p(x, dm)


    def gd(self, x, dm):
        """! The pressure at the boundary of the bulk domain (Dirichlet
            condition).

            @param x  The spatial coordinate.
            @param dm  A domain marker.
        """
        return self.p(x, dm)


    def gn(self, x, dm):
        """! The velocity at the boundary of the bulk domain (Neumann
            condition).

            @param x  The spatial coordinate.
            @param dm  A domain marker.
        """
        return 0.


    def boundary_dn(self, x, dm):
        """! Returns 1 for a Dirichlet condition and 0 for a Neumann condition
            on the boundary of the bulk domain.

            @param x  The spatial coordinate.
            @param dm  A domain marker.
        """
        return 1.


    def k(self, x, dm):
        """! The permeability matrix k for the bulk domain.

            @param x  The spatial coordinate.
            @param dm  A domain marker.
        """
        dim = x.ufl_shape[0]
        return Identity(dim)


    def p_gamma(self, x):
        """! The exact solution p_gamma inside the reduced fracture.

            @param x  The spatial coordinate.
        """
        return cos(4 * x[1])


    def q_gamma(self, x):
        """! The source term q_gamma inside the reduced fracture.

            @param x  The spatial coordinate.
        """
        return -32 * self.d(x) * sin(4 * x[1]) \
         - 2 * sqrt(2) * sinh(2 * sqrt(2)) * self.p_gamma(x)


    def gd_gamma(self, x):
        """! The pressure at the boundary of the reduced fracture (Dirichlet
            condition).

            @param x  The spatial coordinate.
        """
        return self.p_gamma(x)


    def gn_gamma(self, x):
        """! The velocity at the boundary of the reduced fracture (Neumann
            condition).

            @param x  The spatial coordinate.
        """
        return 0.


    def boundary_dn_gamma(self, x):
        """! Returns 1 for a Dirichlet condition and 0 for a Neumann condition
            on the boundary of the reduced fracture.

            @param x  The spatial coordinate.
        """
        return 1.


    def k_gamma(self, x):
        """! The (tangential) permeability matrix k_gamma inside the reduced
            fracture.

            @param x  The spatial coordinate.
        """
        dim = x.ufl_shape[0]
        return Identity(dim)


    def k_gamma_perp(self, x):
        """! The normal permeability for the reduced fracture.

            @param x  The spatial coordinate.
        """
        return sqrt(2) * self.d(x) / tanh(sqrt(2))


    def d(self, x):
        """! The total aperture d of the fracture.

            @param x  The spatial coordinate.
        """
        return self.dmax * exp(-4 * x[1])
