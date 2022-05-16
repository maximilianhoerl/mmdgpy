from ufl import *
from mmdgpy.problems.mmdgproblem import MMDGProblem

class MMDGProblemVertical(MMDGProblem):
    """! An abstract interface class for a Darcy-type problem with
        domain [0,1]^dim and reduced fracture along the interface x0=0.5, that
        is to be solved by a discontinuous Galerkin scheme, i.e., by an instance
        of the class MMDG2 or MMDG1.
    """

    def leftright(self, normal, left, right):
        """! Returns the expression 'left' or 'right' depending on the
            orientation of the normal.

            @param normal  A bulk facet normal on the interface.
            @param left  An expression to return depending on the orientation of
                         the normal.
            @param right  An expression to return depending on the orientation
                          of the normal.
        """
        dim = normal.ufl_shape[0]
        e1 = [0] * dim
        e1[0] = 1

        return conditional(dot(normal, as_vector(e1)) > 0, left, right)


    def trafo(self, x, dm):
        """! A transformation function that determines the contortion of the
            domain.

            @param x  The spatial coordinate.
            @param dm  A domain marker.
        """
        dim = x.ufl_shape[0]
        e1 = [0] * dim
        e1[0] = 1

        return x + 2 * ( (dm - 1) * self.d1(x) * x[0] \
         + dm * self.d2(x) * ( 1 - x[0] ) ) * as_vector(e1)
