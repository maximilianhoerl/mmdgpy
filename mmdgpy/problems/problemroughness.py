from ufl import *
from dune.ufl import Constant
from dune.fem.space import lagrange, finiteVolume
from mmdgpy.problems.problemcomparison import ProblemComparison


class ProblemRoughness(ProblemComparison):
    """A class to create custom problems on the domain [0,1]^dim with a
    fracture along the interface x0=0.5 to be solved by a discontinuous
    Galerkin scheme. The aperture functions d1, d2 are not given in ufl
    representation so that we provide methods to interpolate them on a the
    given grid.

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
    :ivar _q: The source term q for the bulk domain.
    :ivar _q_gamma: The source term q_gamma inside the reduced fracture.
    :ivar _bulk_d1: The aperture d_1 of the fracture left to the interface
        in bulk coordinates.
    :ivar _bulk_d2: The aperture d_2 of the fracture right to the interface
        in bulk coordinates.
    :ivar _i_d1: The aperture d_1 of the fracture left to the interface in
        interface coordinates.
    :ivar _i_d2: The aperture d_2 of the fracture right to the interface in
        interface coordinates.
    :ivar _bulk_grad_d1: The gradient of the aperture function d1 in bulk
        coordinates.
    :ivar _bulk_grad_d2: The gradient of the aperture function d2 in bulk
        coordinates.
    :ivar _i_grad_d1:
        The gradient of the aperture function d1 in interface coordinates.
    :ivar _i_grad_d2: The gradient of the aperture function d2 in interface
        coordinates.
    :ivar _grid: The grid view of a dG method.
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
        q,
        q_gamma,
    ):
        """The constructor.

        :param gd: The pressure at the boundary of the bulk domain
            (Dirichlet condition).
        :param gn: The velocity at the boundary of the bulk domain
            (Neumann condition).
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
        :param q: The source term q for the bulk domain.
        :param q_gamma: The source term q_gamma inside the reduced fracture.
        """
        super().__init__(
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
        )
        self._q = q
        self._q_gamma = q_gamma

    def q(self, x, dm):
        """The source term q for the bulk domain.

        :param x: The spatial coordinate.
        :param dm: A domain marker.
        """
        return self._q(x, dm)

    def q_gamma(self, x):
        """The source term q_gamma inside the reduced fracture.

        :param x: The spatial coordinate.
        """
        return self._q_gamma(x)

    def d1(self, x):
        """The aperture d_1 of the fracture left to the interface.

        :param x: The spatial coordinate.
        """
        if x.ufl_domain().topological_dimension() == self._grid.dimension:
            return self._bulk_d1
        else:
            return self._i_d1

    def d2(self, x):
        """The aperture d_2 of the fracture right to the interface.

        :param x: The spatial coordinate.
        """
        if x.ufl_domain().topological_dimension() == self._grid.dimension:
            return self._bulk_d2
        else:
            return self._i_d2

    def grad_d1(self, x):
        """The gradient of the aperture function d1.

        :param x: The spatial coordinate.
        """
        if x.ufl_domain().topological_dimension() == self._grid.dimension:
            return self._bulk_grad_d1
        else:
            return self._i_grad_d1

    def grad_d2(self, x):
        """The gradient of the aperture function d1.

        :param x:  The spatial coordinate.
        """
        if x.ufl_domain().topological_dimension() == self._grid.dimension:
            return self._bulk_grad_d2
        else:
            return self._i_grad_d2

    def initialize(self, grid):
        """Interpolates the aperture functions d1, d2 and their gradients
        on the given grid since they are not given in an ufl representation.

        :param grid: The grid view of a dG method.
        """
        self._grid = grid
        bulkspace = lagrange(self._grid, order=1)
        ispace = lagrange(self._grid.hierarchicalGrid.interfaceGrid, order=1)
        bulkvectorspace = finiteVolume(grid, dimRange=grid.dimension)
        ivectorspace = finiteVolume(
            grid.hierarchicalGrid.interfaceGrid, dimRange=grid.dimension
        )

        self._bulk_d1 = bulkspace.function(name="bulk_d1")
        self._bulk_d2 = bulkspace.function(name="bulk_d2")
        self._i_d1 = ispace.function(name="i_d1")
        self._i_d2 = ispace.function(name="i_d2")

        self._interpolate(self._d1, self._bulk_d1)
        self._interpolate(self._d2, self._bulk_d2)
        self._interpolate(self._d1, self._i_d1)
        self._interpolate(self._d2, self._i_d2)

        self._bulk_grad_d1 = bulkvectorspace.interpolate(
            grad(self._bulk_d1), name="bulk_grad_d1"
        )
        self._bulk_grad_d2 = bulkvectorspace.interpolate(
            grad(self._bulk_d2), name="bulk_grad_d2"
        )
        self._i_grad_d1 = ivectorspace.interpolate(grad(self._i_d1), name="i_grad_d1")
        self._i_grad_d2 = ivectorspace.interpolate(grad(self._i_d2), name="i_grad_d2")

    def update(self, grid):
        """Updates the aperture functions and gradients after a transformation
        of the grid.

        :param grid: The grid view of a dG method.
        """
        self._grid = grid

        bulkspace = lagrange(grid, order=1)
        d1_tmp = self._bulk_d1.as_numpy.copy()
        d2_tmp = self._bulk_d2.as_numpy.copy()
        self._bulk_d1 = bulkspace.function(name="bulk_d1new")
        self._bulk_d2 = bulkspace.function(name="bulk_d2new")
        self._bulk_d1.as_numpy[:] = d1_tmp
        self._bulk_d2.as_numpy[:] = d2_tmp

        bulkvectorspace = finiteVolume(grid, dimRange=grid.dimension)
        self._bulk_grad_d1 = bulkvectorspace.interpolate(
            grad(self._bulk_d1), name="bulk_grad_d1"
        )
        self._bulk_grad_d2 = bulkvectorspace.interpolate(
            grad(self._bulk_d2), name="bulk_grad_d2"
        )

    def _interpolate(self, fct_np, fct_ufl):
        """Interpolates a function that is not given in ufl representation.

        :param fct_np: A numpy function to be interpolated.
        :param fct_ufl: The interpolated ufl function.
        """
        grid = fct_ufl.gridView

        idxSet = grid.indexSet
        for v in grid.vertices:
            idx = idxSet.index(v)
            cnt = v.geometry.center
            fct_ufl.as_numpy[:][idx] = fct_np([cnt[0], cnt[1], 0.0])
