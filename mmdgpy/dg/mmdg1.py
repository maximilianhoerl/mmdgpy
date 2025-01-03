from ufl import dx, dS, ds, grad, dot, jump
from ufl.checks import is_globally_constant
from dune.mmesh import trace, normals
from mmdgpy.dg.mmdg2 import MMDG2


class MMDG1(MMDG2):
    """A discontinuous Galerkin scheme in d=2,3 dimensions with a reduced
    fracture of variable aperture. Aperture gradients are included.

    :ivar MMDGProblem problem: A problem implementing the interface
        MMDGProblem.
    :ivar omega: The bulk grid view.
    :ivar space: The bulk dG space.
    :ivar velocity_space: The bulk velocity dG space.
    :ivar x: The bulk spatial coordinate.
    :ivar dm: A domain marker.
    :ivar chi_gamma: An interface indicator.
    :ivar p: The bulk trial function.
    :ivar phi: The bulk test function.
    :ivar n: The bulk facet normal.
    :ivar ph: The bulk numerical solution.
    :ivar iterms: An integrand on internal bulk facets.
    :ivar b_bulk: The bilinear form of bulk contributions.
    :ivar l_bulk: The linear form of bulk contributions.
    :ivar storage: The underlying linear algebra backend.
    :ivar uh: The bulk velocity numerical solution.
    :ivar igridview: The interface grid view.
    :ivar ispace: The interface dG space.
    :ivar x_gamma: The interface spatial coordinate.
    :ivar phi_gamma: The interface test function.
    :ivar n_gamma: The interface facet normal.
    :ivar ph_gamma: The interface numerical solution.
    :ivar b_gamma: The bilinear form of interface contributions.
    :ivar l_gamma: The linear form of interface contributions.
    :ivar c_bulk: The bilinear form of coupling terms (with bulk test
        function).
    :ivar c_gamma: The bilinear form of coupling terms (with interface test
        function).
    :ivar inormal: The normal vector on the interface.
    """

    def __init__(
        self,
        dim,
        order,
        gridfile,
        problem,
        mu0,
        xi=2.0 / 3.0,
        contortion=True,
        trafo=None,
    ):
        """The constructor.

        :param int dim: The bulk dimension dim=2,3.
        :param int order: The order of accuracy >= 1 of the dG method.
        :param str gridfile: A .msh grid file.
        :param MMDGProblem problem: A problem implementing the interface
            MMDGProblem.
        :param float mu0: A penalty parameter that must be chosen
            sufficiently large.
        :param float xi: A coupling parameter. Defaults to 2/3.
        :param bool contortion: A boolean that indicates whether the domain
            is to be contorted according to a given transformation. Defaults
            to False.
        :param trafo: A transformation function depending on the spatial
            coordinate and the domain marker that determines the contortion
            of the domain. By default the transformation of the given
            problem is used.
        """
        super().__init__(
            dim,
            order,
            gridfile,
            problem,
            mu0,
            xi=xi,
            contortion=contortion,
            trafo=trafo,
        )

        if not (
            is_globally_constant(problem.d1(self.x_gamma))
            and is_globally_constant(problem.d2(self.x_gamma))
        ):

            self.inormal = normals(self.igridview)

            self.b_gamma -= (
                trace(self.ph, self.igridview)("+")
                * dot(
                    grad(problem.d_i(self.x_gamma, self.inormal)),
                    problem.k_gamma(self.x_gamma) * grad(self.phi_gamma),
                )
                * dx
            )
            self.b_gamma -= (
                trace(self.ph, self.igridview)("-")
                * dot(
                    grad(problem.d_i(self.x_gamma, -self.inormal)),
                    problem.k_gamma(self.x_gamma) * grad(self.phi_gamma),
                )
                * dx
            )

            trp = trace(self.ph, self.igridview, restrictTo="+")
            trm = trace(self.ph, self.igridview, restrictTo="-")

            self.b_gamma += (
                jump(self.phi_gamma)
                * dot(
                    problem.k_gamma(self.x_gamma)
                    * 0.5
                    * trp("+")
                    * grad(problem.d_i(self.x_gamma, self.inormal("+")))("+")
                    + 0.5
                    * trp("-")
                    * grad(problem.d_i(self.x_gamma, self.inormal("-")))("-"),
                    self.n_gamma("+"),
                )
                * dS
            )
            self.b_gamma += (
                jump(self.phi_gamma)
                * dot(
                    problem.k_gamma(self.x_gamma)
                    * 0.5
                    * trm("+")
                    * grad(problem.d_i(self.x_gamma, -self.inormal("+")))("+")
                    + 0.5
                    * trm("-")
                    * grad(problem.d_i(self.x_gamma, -self.inormal("-")))("-"),
                    self.n_gamma("+"),
                )
                * dS
            )

            if self.problem.boundary_dn_gamma(self.x_gamma) != 0:
                self.b_gamma += (
                    self.phi_gamma
                    * trp
                    * dot(grad(problem.d_i(self.x_gamma, self.inormal)), self.n_gamma)
                    * self.problem.boundary_dn_gamma(self.x_gamma)
                    * ds
                )
                self.b_gamma += (
                    self.phi_gamma
                    * trm
                    * dot(grad(problem.d_i(self.x_gamma, -self.inormal)), self.n_gamma)
                    * self.problem.boundary_dn_gamma(self.x_gamma)
                    * ds
                )

    def _calculate_velocity(self):
        """Internal method to calculate the velocity field after solving the scheme for the
        pressure.
        """
        self.uh = self.velocity_space.interpolate(
            -self.problem.k(self.x, self.dm) * grad(self.ph), name="velocity"
        )

        self.uh_gamma_perp = self.ispace.interpolate(
            self.problem.k_gamma_perp(self.x_gamma)
            / self.problem.d(self.x_gamma)
            * jump(trace(self.ph, self.igridview)),
            name="velocityGammaPerp",
        )

        self.uh_gamma_tang = self.velocity_ispace.interpolate(
            -self.problem.k_gamma(self.x_gamma)
            / self.problem.d(self.x_gamma)
            * (
                grad(self.problem.d(self.x_gamma) * self.ph_gamma)
                - trace(self.ph, self.igridview)("+")
                * grad(self.problem.d_i(self.x_gamma, self.inormal))
                - trace(self.ph, self.igridview)("-")
                * grad(self.problem.d_i(self.x_gamma, -self.inormal))
            ),
            name="velocityGammaTang",
        )
