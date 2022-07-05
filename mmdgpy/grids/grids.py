import gmsh
import numpy as np

################################################################################

def create_dgf_grid(file, n, dim):
    """ Writes a dgf grid file for the unit square [0,1]^dim.

        :param int n: The number of grid elements per direction.
        :param int dim: The dimension of the domain.
    """
    file = open(file, "w")
    file.write("DGF\n\nInterval\n")

    # write lower left corner
    for i in range(0, dim):
        file.write(str(0) + "\t")
    file.write("\n")

    # write upper right corner
    for i in range(0, dim):
        file.write(str(1) + "\t")
    file.write("\n")

    # write number of grid elements
    for i in range(0, dim):
        file.write(str(n) + "\t")
    file.write("\n#\n\nSimplex\n#")

    file.close()

    return

################################################################################

def create_reduced_grid(file, h, hf, dim):
    """ Creates a msh grid file for the unit square [0,1]^dim with a vertical
        interface at x0=0.5.

        :param float h: A characteristic grid width for the bulk.
        :param float hf: A characteristic grid width at the interface.
        :param int dim: The dimension of the domain.
        :raises ValueError: If not (dim=2 or dim=3).
    """
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 8)

    gmsh.model.add(file)

    geo = gmsh.model.geo

    p1 = geo.addPoint(  0, 0, 0, h, 1)
    p2 = geo.addPoint(0.5, 0, 0, hf, 2)
    p3 = geo.addPoint(  1, 0, 0, h, 3)
    p4 = geo.addPoint(  1, 1, 0, h, 4)
    p5 = geo.addPoint(0.5, 1, 0, hf, 5)
    p6 = geo.addPoint(  0, 1, 0, h, 6)

    l1 = geo.addLine(p1, p2, 1)
    l2 = geo.addLine(p2, p3, 2)
    l3 = geo.addLine(p3, p4, 3)
    l4 = geo.addLine(p4, p5, 4)
    l5 = geo.addLine(p5, p6, 5)
    l6 = geo.addLine(p6, p1, 6)
    lf1 = geo.addLine(p2, p5, 31)

    geo.addCurveLoop([1, 31, 5, 6], 1)
    geo.addCurveLoop([2, 3, 4, -31], 2)

    geo.addPlaneSurface([1], 0)
    geo.addPlaneSurface([2], 1)

    if dim == 3:
        p11 = geo.addPoint(  0, 0, 1, h, 11)
        p12 = geo.addPoint(0.5, 0, 1, hf, 12)
        p13 = geo.addPoint(  1, 0, 1, h, 13)
        p14 = geo.addPoint(  1, 1, 1, h, 14)
        p15 = geo.addPoint(0.5, 1, 1, hf, 15)
        p16 = geo.addPoint(  0, 1, 1, h, 16)

        l11 = geo.addLine(p11, p12, 11)
        l12 = geo.addLine(p12, p13, 12)
        l13 = geo.addLine(p13, p14, 13)
        l14 = geo.addLine(p14, p15, 14)
        l15 = geo.addLine(p15, p16, 15)
        l16 = geo.addLine(p16, p11, 16)
        l21 = geo.addLine(p1, p11, 21)
        l22 = geo.addLine(p3, p13, 22)
        l23 = geo.addLine(p4, p14, 23)
        l24 = geo.addLine(p6, p16, 24)
        lf2 = geo.addLine(p5, p15, 32)
        lf3 = geo.addLine(p15, p12, 33)
        lf4 = geo.addLine(p12, p2, 34)

        geo.addCurveLoop([11, -33, 15, 16], 3)
        geo.addCurveLoop([12, 13, 14, 33], 4)
        geo.addCurveLoop([1, -34, -11, -21], 5)
        geo.addCurveLoop([2, 22, -12, 34], 6)
        geo.addCurveLoop([-5, 32, 15, -24], 7)
        geo.addCurveLoop([-4, 23, 14, -32], 8)
        geo.addCurveLoop([6, 21, -16, -24], 9)
        geo.addCurveLoop([-3, 22, 13, -23], 10)
        geo.addCurveLoop([31, 32, 33, 34], 11)

        for i in range(2, 11):
            geo.addPlaneSurface([i+1], i)

        geo.addSurfaceLoop([0, 2, 4, 6, 8, 10], 1)
        geo.addVolume([1], 0)
        geo.addSurfaceLoop([1, 3, 5, 7, 9, 10], 2)
        geo.addVolume([2], 1)

    elif dim != 2:
        raise ValueError('Invalid dimension!')

    geo.synchronize()

    gmsh.model.mesh.generate(dim=dim)
    gmsh.write(file)
    gmsh.finalize()

################################################################################

def create_resolved_grid(file, d1, d2, h, hf, dim):
    """ Creates a msh grid file for the unit square [0,1]^dim with a resolved
        vertical fracture.

        :param d1: The aperture left of the interface x0=0.5.
        :param d2: The aperture right of the interface x0=0.5.
        :param float h: A characteristic grid width for the bulk.
        :param float hf: A characteristic grid with at the interface.
        :param int dim: The dimension of the domain.
        :raises ValueError: If not (dim=2 or dim=3).
    """
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("General.Verbosity", 0)

    gmsh.model.add(file)

    geo = gmsh.model.geo

    hf = 1. / np.ceil(1. / hf)
    pos = np.arange(0., 1. + 1e-12, hf)
    N = len(pos)

    if dim == 2:
        bottomleft = geo.addPoint(0, 0, 0, h, 1)
        bottomright = geo.addPoint(1, 0, 0, h, 2)
        topleft = geo.addPoint(0, 1, 0, h, 3)
        topright = geo.addPoint(1, 1, 0, h, 4)

        points_gamma1 = []
        points_gamma2 = []

        for i in range(N):
            points_gamma1.append(geo.addPoint(
             0.5 - d1([0.5, pos[i], 0]), pos[i], 0, hf, 100 + 2 * i))
            points_gamma2.append(geo.addPoint(
             0.5 + d2([0.5, pos[i], 0]), pos[i], 0, hf, 100 + 2 * i + 1))

        outline_left = geo.addLine(bottomleft, topleft, 1)
        outline_right = geo.addLine(bottomright, topright, 2)
        outline_bottomleft = geo.addLine(bottomleft, points_gamma1[0], 3)
        outline_bottommid = \
         geo.addLine(points_gamma1[0], points_gamma2[0], 4)
        outline_bottomright = \
         geo.addLine(points_gamma2[0], bottomright, 5)
        outline_topleft = geo.addLine(topleft, points_gamma1[-1], 6)
        outline_topmid = \
         geo.addLine(points_gamma1[-1], points_gamma2[-1], 7)
        outline_topright = \
         geo.addLine(points_gamma2[-1], topright, 8)

        lines_gamma1 = []
        lines_gamma2 = []

        for i in range(N-1):
            lines_gamma1.append(geo.addLine(
             points_gamma1[i], points_gamma1[i+1], 100 + 2 * i))
            lines_gamma2.append(geo.addLine(
             points_gamma2[N-i-1], points_gamma2[N-i-2], 100 + 2 * i + 1))

        geo.addCurveLoop(lines_gamma1 + [-6, -1, 3], 1)
        geo.addCurveLoop(lines_gamma2 + [5, 2, -8], 2)
        geo.addCurveLoop(lines_gamma1 + [7] + lines_gamma2 + [-4], 3)

        geo.addPlaneSurface([1], 0) # Omega_1
        geo.addPlaneSurface([2], 1) # Omega_2
        geo.addPlaneSurface([3], 2) # Omega_f

    elif dim == 3:
        p1 = geo.addPoint(0,0,0,h)
        p2 = geo.addPoint(1,0,0,h)
        p3 = geo.addPoint(0,1,0,h)
        p4 = geo.addPoint(1,1,0,h)
        p5 = geo.addPoint(0,0,1,h)
        p6 = geo.addPoint(1,0,1,h)
        p7 = geo.addPoint(0,1,1,h)
        p8 = geo.addPoint(1,1,1,h)

        points_gamma1 = np.empty((N,N), dtype=int)
        points_gamma2 = np.empty((N,N), dtype=int)
        lines_gamma1_x2 = np.empty((N-1,N), dtype=int)
        lines_gamma1_x3 = np.empty((N,N-1), dtype=int)
        lines_gamma2_x2 = np.empty((N-1,N), dtype=int)
        lines_gamma2_x3 = np.empty((N,N-1), dtype=int)
        curves_gamma1 = np.empty((N-1,N-1), dtype=int)
        curves_gamma2 = np.empty((N-1,N-1), dtype=int)
        surf_gamma1 = np.empty((N-1,N-1), dtype=int)
        surf_gamma2 = np.empty((N-1,N-1), dtype=int)

        for i in range(N):
            for j in range(N):
                points_gamma1[i,j] = geo.addPoint(
                 0.5 - d1([0.5, pos[i], pos[j]]), pos[i], pos[j], hf)
                points_gamma2[i,j] = geo.addPoint(
                 0.5 + d2([0.5, pos[i], pos[j]]), pos[i], pos[j], hf)

        for i in range(N-1):
            for j in range(N):
                lines_gamma1_x2[i,j] = \
                 geo.addLine(points_gamma1[i,j], points_gamma1[i+1,j])
                lines_gamma2_x2[i,j] = \
                 geo.addLine(points_gamma2[i,j], points_gamma2[i+1,j])

        for i in range(N):
            for j in range(N-1):
                lines_gamma1_x3[i,j] = \
                 geo.addLine(points_gamma1[i,j], points_gamma1[i,j+1])
                lines_gamma2_x3[i,j] = \
                 geo.addLine(points_gamma2[i,j], points_gamma2[i,j+1])

        l13 = geo.addLine(p1, p3)
        l15 = geo.addLine(p1, p5)
        l1f = geo.addLine(p1, points_gamma1[0,0])
        l24 = geo.addLine(p2, p4)
        l26 = geo.addLine(p2, p6)
        l37 = geo.addLine(p3, p7)
        l3f = geo.addLine(p3, points_gamma1[N-1,0])
        l57 = geo.addLine(p5, p7)
        l5f = geo.addLine(p5, points_gamma1[0,N-1])
        l48 = geo.addLine(p4, p8)
        l68 = geo.addLine(p6, p8)
        l7f = geo.addLine(p7, points_gamma1[N-1,N-1])
        lf2 = geo.addLine(points_gamma2[0,0], p2)
        lf4 = geo.addLine(points_gamma2[N-1,0], p4)
        lf6 = geo.addLine(points_gamma2[0,N-1], p6)
        lf8 = geo.addLine(points_gamma2[N-1,N-1], p8)
        lf_12 = geo.addLine(points_gamma1[0,0], points_gamma2[0,0])
        lf_34 = geo.addLine(points_gamma1[N-1,0], points_gamma2[N-1,0])
        lf_56 = geo.addLine(points_gamma1[0,N-1], points_gamma2[0,N-1])
        lf_78 = geo.addLine(points_gamma1[N-1,N-1], points_gamma2[N-1,N-1])

        c1357 = geo.addCurveLoop([l13, l37, -l57, -l15])
        c13f = geo.addCurveLoop(
         np.concatenate(([-l3f, -l13, l1f], lines_gamma1_x2[:,0])))
        c15f = geo.addCurveLoop(
         np.concatenate(([-l5f, -l15, l1f], lines_gamma1_x3[0,:])))
        c2468 = geo.addCurveLoop([l24, l48, -l68, -l26])
        c24f = geo.addCurveLoop(
         np.concatenate(([lf4, -l24, -lf2], lines_gamma2_x2[:,0])))
        c26f = geo.addCurveLoop(
         np.concatenate(([lf6, -l26, -lf2], lines_gamma2_x3[0,:])))
        c37f = geo.addCurveLoop(
         np.concatenate(([-l7f, -l37, l3f], lines_gamma1_x3[N-1,:])))
        c48f = geo.addCurveLoop(
         np.concatenate(([lf8, -l48, -lf4], lines_gamma2_x3[N-1,:])))
        c57f = geo.addCurveLoop(
         np.concatenate(([-l7f, -l57, l5f], lines_gamma1_x2[:,N-1])))
        c68f = geo.addCurveLoop(
         np.concatenate(([lf8, -l68, -lf6], lines_gamma2_x2[:,N-1])))
        cf_1234 = geo.addCurveLoop(np.concatenate((lines_gamma1_x2[:,0],
         [lf_34], -np.flip(lines_gamma2_x2[:,0]), [-lf_12])))
        cf_1256 = geo.addCurveLoop(np.concatenate((lines_gamma1_x3[0,:],
         [lf_56], -np.flip(lines_gamma2_x3[0,:]), [-lf_12])))
        cf_3478 = geo.addCurveLoop(np.concatenate((lines_gamma1_x3[N-1,:],
         [lf_78], -np.flip(lines_gamma2_x3[N-1,:]), [-lf_34])))
        cf_5678 = geo.addCurveLoop(np.concatenate((lines_gamma1_x2[:,N-1],
         [lf_78], -np.flip(lines_gamma2_x2[:,N-1]), [-lf_56])))

        s1357 = geo.addPlaneSurface([c1357])
        s13f = geo.addPlaneSurface([c13f])
        s15f = geo.addPlaneSurface([c15f])
        s2468 = geo.addPlaneSurface([c2468])
        s24f = geo.addPlaneSurface([c24f])
        s26f = geo.addPlaneSurface([c26f])
        s37f = geo.addPlaneSurface([c37f])
        s48f = geo.addPlaneSurface([c48f])
        s57f = geo.addPlaneSurface([c57f])
        s68f = geo.addPlaneSurface([c68f])
        sf_1234 = geo.addPlaneSurface([cf_1234])
        sf_1256 = geo.addPlaneSurface([cf_1256])
        sf_3478 = geo.addPlaneSurface([cf_3478])
        sf_5678 = geo.addPlaneSurface([cf_5678])

        for i in range(N-1):
            for j in range(N-1):
                curves_gamma1[i,j] = geo.addCurveLoop([ lines_gamma1_x2[i,j],
                 lines_gamma1_x3[i+1,j], -lines_gamma1_x2[i,j+1],
                 -lines_gamma1_x3[i,j] ])
                curves_gamma2[i,j] = geo.addCurveLoop([ lines_gamma2_x2[i,j],
                 lines_gamma2_x3[i+1,j], -lines_gamma2_x2[i,j+1],
                 -lines_gamma2_x3[i,j] ])
                surf_gamma1[i,j] = geo.addPlaneSurface([curves_gamma1[i,j]])
                surf_gamma2[i,j] = geo.addPlaneSurface([curves_gamma2[i,j]])

        sl1 = geo.addSurfaceLoop(
         np.concatenate(([s1357, s13f, s15f, s37f, s57f], surf_gamma1.flat)))
        sl2 = geo.addSurfaceLoop(
         np.concatenate(([s2468, s24f, s26f, s48f, s68f], surf_gamma2.flat)))
        slf = geo.addSurfaceLoop(
         np.concatenate(([sf_1234, sf_1256, sf_3478, sf_5678],
         surf_gamma1.flat, surf_gamma2.flat)))

        geo.addVolume([sl1], 0) # Omega_1
        geo.addVolume([sl2], 1) # Omega_2
        geo.addVolume([slf], 2) # Omega_f

    else:
        raise ValueError('Invalid dimension!')

    geo.synchronize()

    gmsh.model.mesh.generate(dim=dim)
    gmsh.write(file)
    gmsh.finalize()

################################################################################
