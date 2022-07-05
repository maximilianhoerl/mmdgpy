from ufl import *
from dune.ufl import Constant
from dune.grid import reader
from dune.mmesh import mmesh, domainMarker, interfaceIndicator, moveMesh
from dune.fem import parameter, adapt
from dune.fem.space import dglagrange, lagrange
from dune.fem.scheme import galerkin
from dune.fem.function import integrate, uflFunction
from dune.fem.view import geometryGridView

class DG:
    """ A discontinuous Galerkin scheme in d=2,3 dimensions without reduced
        fracture.
    """
    # Attributes:
    ## @var problem
    # A problem implementing the interface DGProblem.
    ## @var omega
    # The bulk grid view.
    ## @var space
    # The bulk dG space.
    ## @var x
    # The bulk spatial coordinate.
    ## @var dm
    # A domain marker.
    ## @var chi_gamma
    # An interface indicator.
    ## @var p
    # The bulk trial function.
    ## @var phi
    # The bulk test function.
    ## @var n
    # The bulk facet normal.
    ## @var ph
    # The bulk numerical solution.
    ## @var iterms
    # An integrand on internal bulk facets.
    ## @var b_bulk
    # The bilinear form of bulk contributions.
    ## @var l_bulk
    # The linear form of bulk contributions.
    ## @var storage
    # The underlying linear algebra backend.

    def __init__(self, dim, order, gridfile, problem, mu0, contortion=False, \
     trafo=None, storage=None):
        """ The constructor.

        Args:
            dim (int): The dimension dim=2,3.
            order (int): The order of accuracy >= 1 of the dG method.
            gridfile (str): A grid file (dgf or msh).
            problem (DGProblem): A problem implementing the interface DGProblem.
            mu0 (int): A penalty parameter that must be chosen
                sufficiently large.
            contortion (bool, optional): A boolean that indicates whether
                the domain is to be contorted according to a
                given transformation. The default value is False.
            trafo (optional): A transformation function depending on the
                spatial coordinate and the domain marker that
                determines the contortion of the domain. The default
                value is None.
            storage (optional): The underlying linear algebra backend
                (None or 'istl'). Defaults to None.
        """
        if gridfile.split(".")[-1] == "msh":
            grid_reader = reader.gmsh
        elif gridfile.split(".")[-1] == "dgf":
            grid_reader = reader.dgf
        else:
            raise Exception("ERROR: unknown grid type")

        self.omega = mmesh((grid_reader, gridfile), dim)
        self.problem = problem
        self.problem.initialize(self.omega)

        if contortion:
            if trafo == None:
                trafo = self.problem.trafo

            vectorSpace = dglagrange(self.omega, dimRange=dim, order=order+1)
            position = vectorSpace.interpolate( \
             trafo(SpatialCoordinate(vectorSpace), domainMarker(self.omega)), \
             name="position")
            self.omega = geometryGridView(position)
            self.problem.update(self.omega)

        self.storage = storage
        self.space = dglagrange(self.omega, order=order, storage=self.storage)
        self.x = SpatialCoordinate(self.space)
        self.dm = domainMarker(self.omega)
        self.chi_gamma = interfaceIndicator(\
         self.omega.hierarchicalGrid.interfaceGrid, grid=self.omega)

        self.p = TrialFunction(self.space)
        self.phi = TestFunction(self.space)
        self.n = FacetNormal(self.space.cell())
        self.ph = self.space.interpolate(0, name="pressure")

        mu = Constant(mu0, name="mu0") * (order + 1) \
         * (order + dim) / MaxFacetEdgeLength(self.space.cell())

        self.iterms = mu * jump(self.p) * jump(self.phi) - jump(self.phi) \
         * dot(avg(self.problem.k(self.x, self.dm) * grad(self.p)), \
         self.n('+')) - jump(self.p) * dot(avg(self.problem.k(self.x, self.dm) \
         * grad(self.phi)), self.n('+'))

        self.b_bulk = dot(self.problem.k(self.x, self.dm) * grad(self.p), \
         grad(self.phi)) * dx
        self.b_bulk += self.iterms * dS

        self.b_bulk += \
         mu * self.p * self.phi * self.problem.boundary_dn(self.x, self.dm) * ds
        self.b_bulk -= self.p * dot(self.problem.k(self.x, self.dm) \
         * grad(self.phi), self.n) * self.problem.boundary_dn(self.x, self.dm) \
         * ds
        self.b_bulk -= self.phi * dot(self.problem.k(self.x, self.dm) \
         * grad(self.p), self.n) * self.problem.boundary_dn(self.x, self.dm) * ds

        self.l_bulk = 0.

        if self.problem.q(self.x, self.dm) != 0:
            self.l_bulk += self.problem.q(self.x, self.dm) * self.phi * dx

        if self.problem.gd(self.x, self.dm) != 0:
            self.l_bulk += mu * self.problem.gd(self.x, self.dm) * self.phi \
             * self.problem.boundary_dn(self.x, self.dm) * ds
            self.l_bulk -= self.problem.gd(self.x, self.dm) \
             * dot(self.problem.k(self.x, self.dm) * grad(self.phi), self.n) \
             * self.problem.boundary_dn(self.x, self.dm) * ds

        if self.problem.boundary_dn(self.x, self.dm) != 1 and \
         self.problem.gn(self.x, self.dm) != 0:
            self.l_bulk += self.problem.gn(self.x, self.dm) * self.phi \
             * ( 1. - self.problem.boundary_dn(self.x, self.dm) ) * ds


    def solve(self, solver=None):
        """! Solves the discontinuous Galerkin problem.

            @param solver  (optional): One of the following solvers. The default
                           solver is 'umfpack' (if storage=None) or 'minres'
                           (if storage='istl').
                           - 'umfpack' (requires storage=None),
                           - 'cg' (requires storage='istl'),
                           - 'minres' (requires storage='istl').
        """
        parameter.append({"fem.verboserank": 0})

        if self.storage == "istl" and solver in [None, "cg", "minres"]:
            if solver==None:
                solver = "minres"
            scheme = galerkin([self.b_bulk == self.l_bulk],
             solver=solver,
             parameters={"newton.verbose": "true",
              "newton.linear.preconditioning.method": "ilu"})

        elif self.storage == None and solver in [None, "umfpack"]:
            scheme = galerkin([self.b_bulk == self.l_bulk], \
             solver=("suitesparse","umfpack"), \
             parameters={"newton.verbose": "true"})

        else:
            raise ValueError("Invalid solver or storage type!")

        scheme.solve(target=self.ph)


    def write_vtk(self, filename="pressure", filenumber=0):
        """! Writes out the solution to VTk. Remember to call solve() first.

            @param filename  (optional) A filename. The default value is
                             'pressure'.
            @param filenumber  (int, optional) A file number if a series of
                               problems is solved. The defaut value is 0.
        """
        uh = -self.problem.k(self.x, self.dm) * grad(self.ph)
        pointdata = {"p": self.ph, "u": uh}

        try:
            p_exact = self.problem.p(self.x, self.dm)
            u_exact = -self.problem.k(self.x, self.dm) \
             * grad(self.problem.p(self.x, self.dm))

            pointdata.update({"p_exact": self.problem.p(self.x, self.dm), \
             "p_error": p_exact - self.ph, "u_exact": u_exact, \
             "u_error": u_exact - uh})
        except:
            pass

        self.omega.writeVTK(filename + str(filenumber), pointdata=pointdata,\
         nonconforming=True, subsampling=self.space.order-1)


    def get_error(self, order):
        """! Returns the L2-error of the solution. Requires that an exact
            solution is implemented for the given problem.

            @param order  (int) The order of integration.
        """
        return sqrt(integrate(self.omega, \
         (self.ph - self.problem.p(self.x, self.dm)) ** 2, order=order))
