from ufl import *
from dune.ufl import Constant
from mmdgpy.problems.dgproblem import DGProblem


class DGProblem2(DGProblem):
    """A test problem with exact solution and a cosine-shaped
    full-dimensional fracture.

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
        """The exact solution p.

        :param x: The spatial coordinate.
        :param dm: A domain marker.
        """
        p1 = 2 * (x[0] + self.d(x)) * exp(-self.d(x))
        p2 = 2 * x[0] * exp(self.d(x))
        pf = (1 + self.d(x)) * exp(2 * x[0] - 1)

        return conditional(
            x[0] < 0.5 * (1 - self.d(x)),
            p1,
            conditional(x[0] > 0.5 * (1 + self.d(x)), p2, pf),
        )

    def q(self, x, dm):
        """The source term q.

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
        qf = -1 * (4 + self._d_prime_prime(x)) * exp(2 * x[0] - 1)

        return conditional(
            x[0] < 0.5 * (1 - self.d(x)),
            q1,
            conditional(x[0] > 0.5 * (1 + self.d(x)), q2, qf),
        )

    def gd(self, x, dm):
        """The pressure at the boundary (Dirichlet condition).

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
        """The permeability matrix k.

        :param x: The spatial coordinate.
        :param dm: A domain marker.
        """
        dim = x.ufl_shape[0]

        k1, k2, kf = [], [], []
        for i in range(dim):
            row1, row2, rowf = [0] * dim, [0] * dim, [0] * dim
            row1[i] = 1.0 / (1 - self.d(x)) if i > 0 else 1
            row2[i] = 1.0 / (1 + self.d(x)) if i > 0 else 1
            rowf[i] = 1.0 / (1 + self.d(x)) if i == 0 else 1
            k1 += [row1]
            k2 += [row2]
            kf += [rowf]

        return conditional(
            x[0] < 0.5 * (1 - self.d(x)),
            as_matrix(k1),
            conditional(x[0] > 0.5 * (1 + self.d(x)), as_matrix(k2), as_matrix(kf)),
        )

    def d(self, x):
        """The aperture function of the fracture.

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
