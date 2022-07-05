from ufl import *
from abc import abstractmethod
from ufl.checks import is_globally_constant
from mmdgpy.problems.dgproblem import DGProblem

class MMDGProblem(DGProblem):
    """ An abstract interface class for a Darcy-type problem with reduced
        fracture that is to be solved by a discontinuous Galerkin scheme, i.e.,
        by an instance of the class MMDG1 or MMDG2.
    """

    @abstractmethod
    def p_gamma(self, x):
        """ The exact solution p_gamma inside the reduced fracture.

            :param x: The spatial coordinate.
        """
        pass


    @abstractmethod
    def q_gamma(self, x):
        """ The source term q_gamma inside the reduced fracture.

            :param x: The spatial coordinate.
        """
        pass


    @abstractmethod
    def gd_gamma(self, x):
        """ The pressure at the boundary of the reduced fracture (Dirichlet
            condition).

            :param x: The spatial coordinate.
        """
        pass


    @abstractmethod
    def gn_gamma(self, x):
        """ The velocity at the boundary of the reduced fracture (Neumann
            condition).

            :param x: The spatial coordinate.
        """
        pass


    @abstractmethod
    def boundary_dn_gamma(self, x):
        """ Returns 1 for a Dirichlet condition and 0 for a Neumann condition
            on the boundary of the reduced fracture.

            :param x: The spatial coordinate.
        """
        pass


    @abstractmethod
    def k_gamma(self, x):
        """ The (tangential) permeability matrix k_gamma inside the reduced
            fracture.

            :param x: The spatial coordinate.
        """
        pass


    @abstractmethod
    def k_gamma_perp(self, x):
        """ The normal permeability for the reduced fracture.

            :param x: The spatial coordinate.
        """
        pass


    @abstractmethod
    def d(self, x):
        """ The total aperture d of the fracture.

            :param x: The spatial coordinate.
        """
        pass


    def d1(self, x):
        """ The aperture d_1 of the fracture left to the interface.

            :param x: The spatial coordinate.
        """
        return 0.5 * self.d(x)


    def d2(self, x):
        """ The aperture d_2 of the fracture right to the interface.

            :param x: The spatial coordinate.
        """
        return 0.5 * self.d(x)


    def grad_d1(self, x):
        """ The gradient of the aperture function d1.

            :param x: The spatial coordinate.
        """
        if is_globally_constant(self.d1(x)):
            dim = x.ufl_shape[0]
            return as_vector([0] * dim)

        return grad(self.d1(x))


    def grad_d2(self, x):
        """ The gradient of the aperture function d1.

            :param x: The spatial coordinate.
        """
        if is_globally_constant(self.d2(x)):
            dim = x.ufl_shape[0]
            return as_vector([0] * dim)

        return grad(self.d2(x))


    def d_i(self, x, normal):
        """ Returns the aperture d1 or d2 depending on the orientation of the
            normal.

            :param x: The spatial coordinate.
            :param normal: A bulk facet normal on the interface.
        """
        return self.leftright(normal, self.d1(x), self.d2(x))


    def grad_d_i(self, x, normal):
        """ Returns the gradient of the aperture d1 or d2 depending on the
            orientation of the normal.

            :param x: The spatial coordinate.
            :param normal: A bulk facet normal on the interface.
        """
        return self.leftright(normal, self.grad_d1(x), self.grad_d2(x))


    @abstractmethod
    def leftright(self, normal, left, right):
        """ Returns the expression 'left' or 'right' depending on the
            orientation of the normal.

            :param normal: A bulk facet normal on the interface.
            :param left: An expression to return depending on the orientation of
                the normal.
            :param right: An expression to return depending on the orientation
                of the normal.
        """
        pass


    @abstractmethod
    def trafo(self, x, dm):
        """ A transformation function that determines the contortion of the
            domain.

            :param x: The spatial coordinate.
            :param dm: A domain marker.
        """
        pass
