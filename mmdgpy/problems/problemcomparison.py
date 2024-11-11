from mmdgpy.problems.mmdgproblemvertical import MMDGProblemVertical


class ProblemComparison(MMDGProblemVertical):
    """A class to create custom problems on the domain [0,1]^dim with a
    fracture along the interface x0=0.5 to be solved by a discontinuous
    Galerkin scheme. The source terms are zero so that the flow is only
    driven by the choice of boundary conditions.

    :ivar _gd: The pressure at the boundary of the bulk domain (Dirichlet
        condition).
    :ivar _gn: The velocity at the boundary of the bulk domain (Neumann
        condition).
    :ivar _boundary_dn: Indicator function to distinguish between Dirichlet
        (1) and Neumann (0) conditions on the boundary of the bulk domain.
    :ivar _k: The permeability matrix k for the bulk domain.
    :ivar _d1: The aperture d_1 of the fracture left to the interface.
    :ivar _d2: The aperture d_2 of the fracture right to the interface.
    :ivar _gd_gamma: The pressure at the boundary of the reduced fracture
        (Dirichlet condition).
    :ivar _gn_gamma: The velocity at the boundary of the reduced fracture
        (Neumann condition).
    :ivar _boundary_dn_gamma: Indicator function to distinguish between
        Dirichlet (1) and Neumann (0) conditions on the boundary of the
        reduced fracture.
    :ivar _k_gamma: The (tangential) permeability matrix k_gamma inside the
        reduced fracture.
    :ivar _k_gamma_perp: The normal permeability for the reduced fracture.
    """

    def __init__(
        self,
        gd,
        gn,
        boundary_dn,
        k,
        d1,
        d2,
        gd_gamma,
        gn_gamma,
        boundary_dn_gamma,
        k_gamma,
        k_gamma_perp,
    ):
        """The constructor.

        :param gd: The pressure at the boundary of the bulk domain
            (Dirichlet condition).
        :param gn: The velocity at the boundary of the bulk domain (Neumann
            condition).
        :param boundary_dn: Indicator function to distinguish between
            Dirichlet (1) and Neumann (0) conditions on the boundary of the
            bulk domain.
        :param k: The permeability matrix k for the bulk domain.
        :param d1: The aperture d_1 of the fracture left to the interface.
        :param d2: The aperture d_2 of the fracture right to the interface.
        :param gd_gamma: The pressure at the boundary of the reduced
            fracture (Dirichlet condition).
        :param gn_gamma: The velocity at the boundary of the reduced
            fracture (Neumann condition).
        :param boundary_dn_gamma: Indicator function to distinguish between
            Dirichlet (1) and Neumann (0) conditions on the boundary of the
            reduced fracture.
        :param k_gamma: The (tangential) permeability matrix k_gamma inside
            the reduced fracture.
        :param k_gamma_perp: The normal permeability for the reduced
            fracture.
        """
        self._gd = gd
        self._gn = gn
        self._boundary_dn = boundary_dn
        self._k = k
        self._d1 = d1
        self._d2 = d2
        self._gd_gamma = gd_gamma
        self._gn_gamma = gn_gamma
        self._boundary_dn_gamma = boundary_dn_gamma
        self._k_gamma = k_gamma
        self._k_gamma_perp = k_gamma_perp

    def p(self, x, dm):
        """The exact solution is not implemented.

        :param x: The spatial coordinate.
        :param dm: A domain marker.
        :raises NotImplementedError:
        """
        raise NotImplementedError

    def q(self, x, dm):
        """The source term q for the bulk domain.

        :param x: The spatial coordinate.
        :param dm: A domain marker.
        """
        return x[0] - x[0]

    def gd(self, x, dm):
        """The pressure at the boundary of the bulk domain (Dirichlet
        condition).

        :param x: The spatial coordinate.
        :param dm: A domain marker.
        """
        return self._gd(x, dm)

    def gn(self, x, dm):
        """The velocity at the boundary of the bulk domain (Neumann condition).

        :param x: The spatial coordinate.
        :param dm: A domain marker.
        """
        return self._gn(x, dm)

    def boundary_dn(self, x, dm):
        """Returns 1 for a Dirichlet condition and 0 for a Neumann condition
        on the boundary of the bulk domain.

        :param x: The spatial coordinate.
        :param dm: A domain marker.
        """
        return self._boundary_dn(x, dm)

    def k(self, x, dm):
        """The permeability matrix k for the bulk domain.

        :param x: The spatial coordinate.
        :param dm: A domain marker.
        """
        return self._k(x, dm)

    def p_gamma(self, x):
        """The exact solution is not implemented.

        :param x: The spatial coordinate.
        :raises NotImplementedError:
        """
        raise NotImplementedError

    def q_gamma(self, x):
        """The source term q_gamma inside the reduced fracture.

        :param x: The spatial coordinate.
        """
        return 0.0

    def gd_gamma(self, x):
        """The pressure at the boundary of the reduced fracture (Dirichlet
        condition).

        :param x: The spatial coordinate.
        """
        return self._gd_gamma(x)

    def gn_gamma(self, x):
        """The velocity at the boundary of the reduced fracture (Neumann
        condition).

        :param x: The spatial coordinate.
        """
        return self._gn_gamma(x)

    def boundary_dn_gamma(self, x):
        """Returns 1 for a Dirichlet condition and 0 for a Neumann condition
        on the boundary of the reduced fracture.

        :param x: The spatial coordinate.
        """
        return self._boundary_dn_gamma(x)

    def k_gamma(self, x):
        """The (tangential) permeability matrix k_gamma inside the reduced
        fracture.

        :param x: The spatial coordinate.
        """
        return self._k_gamma(x)

    def k_gamma_perp(self, x):
        """The normal permeability for the reduced fracture.

        :param x: The spatial coordinate.
        """
        return self._k_gamma_perp(x)

    def d(self, x):
        """The total aperture d of the fracture.

        :param x: The spatial coordinate.
        """
        return self.d1(x) + self.d2(x)

    def d1(self, x):
        """The aperture d_1 of the fracture left to the interface.

        :param x: The spatial coordinate.
        """
        return self._d1(x)

    def d2(self, x):
        """The aperture d_2 of the fracture right to the interface.

        :param x: The spatial coordinate.
        """
        return self._d2(x)
