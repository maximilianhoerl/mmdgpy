from ufl import *
from dune.ufl import Constant
from mmdgpy.problems.mmdgproblemvertical import MMDGProblemVertical


class MMDGProblem1(MMDGProblemVertical):
    """A test problem for a fracture with constant aperture. For xi=0.75, the
    problem is an exact solution of the model in MMDG2 (no contortion).
    The bulk domain is Omega=[0,1]^dim with fracture Gamma={x_0 = 0.5}. The
    problem is formulated in analogy to Ex. 6.3 in P. F. Antonietti et al.
    (2019): SIAM J. Sci. Comput. 41(1), A109–A138.

    :ivar d0: The constant aperture of the fracture.
    """

    def __init__(self, d0):
        """The constructor.

        :param float d0: The constant aperture of the fracture.
        """
        self.d0 = Constant(d0, name="d0")

    def p(self, x, dm):
        """The exact solution p for the bulk domain.

        p_1 = sin(4*x_0) * cos(pi*x_1),
        p_2 = cos(4*x_0) * cos(pi*x_1).

        :param x: The spatial coordinate.
        :param dm: A domain marker.
        """
        return (1 - dm) * sin(4 * x[0]) * cos(pi * x[1]) + dm * cos(4 * x[0]) * cos(
            pi * x[1]
        )

    def q(self, x, dm):
        """The source term q for the bulk domain.

        :param x: The spatial coordinate.
        :param dm: A domain marker.
        """
        return (16 + pi**2) * self.p(x, dm)

    def gd(self, x, dm):
        """The pressure at the boundary of the bulk domain (Dirichlet
        condition).

        :param x: The spatial coordinate.
        :param dm: A domain marker.
        """
        return self.p(x, dm)

    def gn(self, x, dm):
        """The velocity at the boundary of the bulk domain (Neumann condition).

        :param x: The spatial coordinate.
        :param dm: A domain marker.
        """
        return 0.0

    def boundary_dn(self, x, dm):
        """Returns 1 for a Dirichlet condition and 0 for a Neumann condition
        on the boundary of the bulk domain.

        :param x: The spatial coordinate.
        :param dm: A domain marker.
        """
        return 1.0

    def k(self, x, dm):
        """The permeability matrix k for the bulk domain.

        :param x: The spatial coordinate.
        :param dm: A domain marker.
        """
        dim = x.ufl_shape[0]
        return Identity(dim)

    def p_gamma(self, x):
        """The exact solution p_gamma inside the reduced fracture.

        :param x: The spatial coordinate.
        """
        return 0.75 * (cos(2.0) + sin(2.0)) * cos(pi * x[1])

    def q_gamma(self, x):
        """The source term q_gamma inside the reduced fracture.

        :param x: The spatial coordinate.
        """
        return (0.75 * self.d0 * pi**2 + 4) * (cos(2.0) + sin(2.0)) * cos(pi * x[1])

    def gd_gamma(self, x):
        """The pressure at the boundary of the reduced fracture (Dirichlet
        condition).

        :param x: The spatial coordinate.
        """
        return self.p_gamma(x)

    def gn_gamma(self, x):
        """The velocity at the boundary of the reduced fracture (Neumann
        condition).

        :param x: The spatial coordinate.
        """
        return -0.75 * pi * (cos(2.0) + sin(2.0)) * sin(pi * x[1])

    def boundary_dn_gamma(self, x):
        """Returns 1 for a Dirichlet condition and 0 for a Neumann condition
        on the boundary of the reduced fracture.

        :param x: The spatial coordinate.
        """
        return conditional(x[1] > 1.0 - 1e-12, 0, 1)

    def k_gamma(self, x):
        """The (tangential) permeability matrix k_gamma inside the reduced
        fracture.

        :param x: The spatial coordinate.
        """
        dim = x.ufl_shape[0]
        return Identity(dim)

    def k_gamma_perp(self, x):
        """The normal permeability for the reduced fracture.

        :param x: The spatial coordinate.
        """
        return 2 * self.d0

    def d(self, x):
        """The total aperture d of the fracture.

        :param x: The spatial coordinate.
        """
        return self.d0
