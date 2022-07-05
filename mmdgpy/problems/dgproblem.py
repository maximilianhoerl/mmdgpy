from abc import abstractmethod, ABC

class DGProblem(ABC):
    """ An abstract interface class for a Darcy flow problem

            -div(k*grad(p)) = q    in Omega,
                          p = g    on bd(Omega).

        The problem is to be solved by a discontinuous Galerkin scheme without
        reduced fracture, i.e., by an instance of the class DG.
    """

    @abstractmethod
    def p(self, x, dm):
        """ The exact solution p for the bulk domain.

            :param x: The spatial coordinate.
            :param dm: A domain marker.
        """
        pass


    @abstractmethod
    def q(self, x, dm):
        """ The source term q for the bulk domain.

            :param x: The spatial coordinate.
            :param dm: A domain marker.
        """
        pass


    @abstractmethod
    def gd(self, x, dm):
        """ The pressure at the boundary of the bulk domain (Dirichlet
            condition).

            :param x: The spatial coordinate.
            :param dm: A domain marker.
        """
        pass


    @abstractmethod
    def gn(self, x, dm):
        """ The velocity at the boundary of the bulk domain (Neumann condition).

            :param x: The spatial coordinate.
            :param dm: A domain marker.
        """
        pass


    @abstractmethod
    def boundary_dn(self, x, dm):
        """ Returns 1 for a Dirichlet condition and 0 for a Neumann condition
            on the boundary of the bulk domain.

            :param x: The spatial coordinate.
            :param dm: A domain marker.
        """
        pass


    @abstractmethod
    def k(self, x, dm):
        """ The permeability matrix k for the bulk domain.

            :param x: The spatial coordinate.
            :param dm: A domain marker.
        """
        pass


    def initialize(self, grid):
        """ Initializes the problem. By default, no initialization is required.
            However, if a member function is not available in UFL
            representation, this method can be overwritten accordingly to obtain
            a representation as interpolated function in a lagrange space with
            respect to the grid of a dG scheme.

            :param grid: The grid view of a dG method.
        """
        pass


    def update(self, grid):
        """ Updates the problem. By default, updating is not required.
            If necessary, this method can be overwritten accordingly to to be
            applied after a grid transformation.

            :param grid: The grid view of a dG method.
        """
        pass
