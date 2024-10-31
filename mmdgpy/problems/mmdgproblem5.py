from ufl import *
from dune.ufl import Constant
from mmdgpy.problems.mmdgproblemvertical import MMDGProblemVertical


class MMDGProblem5(MMDGProblemVertical):
    """! The reduced problem corresponding to DGProblem2. Note that this problem
    is NO exact solution of any of the implemented reduced models.

    :ivar d_mean: The mean aperture of the fracture.
    :ivar d_diff: The maximum aperture fluctuation of the fracture.
    """

    def __init__(self, d_mean, d_diff):
        """The constructor.

        :param float d_mean: The mean aperture of the fracture.
        :param float d_diff: The maximum aperture fluctuation of the
            fracture.
        """
        self.d_mean = Constant(d_mean, name="d_mean")
        self.d_diff = Constant(d_diff, name="d_diff")

    def p(self, x, dm):
        """The exact solution p for the bulk domain.

        :param x: The spatial coordinate.
        :param dm: A domain marker.
        """
        p1 = 2 * (x[0] + self.d(x)) * exp(-self.d(x))
        p2 = 2 * x[0] * exp(self.d(x))

        return (1 - dm) * p1 + dm * p2

    def q(self, x, dm):
        """The source term q for the bulk domain.

        :param x: The spatial coordinate.
        :param dm: A domain marker.
        """
        q1 = (
            -2.0
            / (1 - self.d(x))
            * exp(-self.d(x))
            * (
                (1 - x[0] - self.d(x))
                * (
                    self._d_prime_prime(x)
                    + (self._d_prime(x) ** 2) * self.d(x) / (1 - self.d(x))
                )
                - self._d_prime(x) ** 2
            )
        )
        q2 = (
            -2.0
            * x[0]
            / (1 + self.d(x))
            * exp(self.d(x))
            * (
                self._d_prime_prime(x)
                + (self._d_prime(x) ** 2) * self.d(x) / (1 + self.d(x))
            )
        )

        return (1 - dm) * q1 + dm * q2

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

        k1, k2 = [], []
        for i in range(dim):
            row1, row2, rowf = [0] * dim, [0] * dim, [0] * dim
            row1[i] = 1.0 / (1 - self.d(x)) if i > 0 else 1
            row2[i] = 1.0 / (1 + self.d(x)) if i > 0 else 1
            k1 += [row1]
            k2 += [row2]

        return (1 - dm) * as_matrix(k1) + dm * as_matrix(k2)

    def p_gamma(self, x):
        """The exact solution p_gamma inside the reduced fracture.

        :param x: The spatial coordinate.
        """
        return (1 + self.d(x)) / self.d(x) * sinh(self.d(x))

    def q_gamma(self, x):
        """The source term q_gamma inside the reduced fracture.

        :param x: The spatial coordinate.
        """
        return -1 * (4 + self._d_prime_prime(x)) * sinh(self.d(x))

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

        k = []
        for i in range(dim):
            row = [0] * dim
            row[i] = 0 if i == 0 else 1
            k += [row]

        return as_matrix(k)

    def k_gamma_perp(self, x):
        """The normal permeability for the reduced fracture.

        :param x: The spatial coordinate.
        """
        return 1.0 / (1 + self.d(x))

    def d(self, x):
        """The total aperture d of the fracture.

        :param x: The spatial coordinate.
        """
        return self.d_mean + 0.5 * self.d_diff * cos(8 * pi * x[1])

    def _d_prime(self, x):
        """The first derivative of the aperture function d.

        :param x: The spatial coordinate.
        """
        return self.d(x).dx(1)

    def _d_prime_prime(self, x):
        """The second derivative of the aperture function d.

        :param x: The spatial coordinate.
        """
        return self._d_prime(x).dx(1)
