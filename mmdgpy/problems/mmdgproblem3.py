from ufl import as_matrix, cosh, div, dot, exp, grad, Identity, sinh
from dune.ufl import Constant
from mmdgpy.problems.mmdgproblemvertical import MMDGProblemVertical


class MMDGProblem3(MMDGProblemVertical):
    """A test problem for a fracture with arbitrary variable aperture. The
    problem is an exact solution of the model in MMDG1 (with
    contortion). The bulk domain is Omega=[0,1]^dim with fracture
    Gamma={x_0 = 0.5}.

    :ivar d1: The aperture of the fracture left to the interface.
    :ivar d2: The aperture of the fracture right to the interface.
    :ivar xi: The coupling parameter.
    :ivar c: A constant larger than sup( abs(d2.dx(1)) ).
    """

    def __init__(self, d1, d2, xi, c=1):
        """The constructor.

        :param d1: The aperture of the fracture left to the interface.
        :param d2: The aperture of the fracture right to the interface.
        :param float xi: The coupling parameter.
        :param float c: A constant larger than sup( abs(d2.dx(1)) ).
        """
        self.d1 = d1
        self.d2 = d2
        self.xi = Constant(xi, name="xi")
        self.c = Constant(c, name="c")

    def p(self, x, dm):
        """The exact solution p for the bulk domain.

        :param x: The spatial coordinate.
        :param dm: A domain marker.
        """
        return (
            (1 - dm) * sinh(x[0] + self.d1(x) - 0.5)
            + dm * cosh(x[0] - self.d2(x) - 0.5)
        ) * exp(x[1])

    def q(self, x, dm):
        """The source term q for the bulk domain.

        :param x: The spatial coordinate.
        :param dm: A domain marker.
        """
        return -div(self.k(x, dm) * grad(self.p(x, dm)))

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

        k1 = []
        for i in range(dim):
            row = [0] * dim
            row[i] = (self.c + self.grad_d2(x)[1]) / (
                1.0 + dot(self.grad_d1(x), self.grad_d1(x))
            )
            k1 += [row]

        k1 = as_matrix(k1)
        k2 = Identity(dim)

        return (1 - dm) * k1 + dm * k2

    def p_gamma(self, x):
        """The exact solution p_gamma inside the reduced fracture.

        :param x: The spatial coordinate.
        """
        return (0.5 + (self.xi - 0.5) * (1 + 2.0 / self.c * self.grad_d2(x)[1])) * exp(
            x[1]
        )

    def q_gamma(self, x):
        """The source term q_gamma inside the reduced fracture.

        :param x: The spatial coordinate.
        """
        return (
            -div(self.k_gamma(x) * grad(self.d(x) * self.p_gamma(x)))
            + (self.c + 2 * self.grad_d2(x)[1]) * exp(x[1])
            + div(exp(x[1]) * self.k_gamma(x) * self.grad_d2(x))
        )

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
        return 0.0

    def boundary_dn_gamma(self, x):
        """Returns 1 for a Dirichlet condition and 0 for a Neumann condition
        on the boundary of the reduced fracture.

        :param x: The spatial coordinate.
        """
        return 1.0

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
        return 0.5 * self.c * self.d(x)

    def d(self, x):
        """The total aperture d of the fracture.

        :param x: The spatial coordinate.
        """
        return self.d1(x) + self.d2(x)
