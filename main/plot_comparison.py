from paraview.simple import *
from os import remove, mkdir
from os.path import join, exists
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vtk as vtk
from vtk.util.numpy_support import vtk_to_numpy
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
from scipy.interpolate import griddata, interp1d

################################################################################

dim = 2 # dimension
roughness = False

vtk_dir = 'vtk'
output_dir = 'plot'

if roughness: # fracture for data from "roughness series"
    npzfile = np.load(join(vtk_dir, 'aperture.npz'))
    xh, z1, z2 = npzfile['xh'], npzfile['d1'], npzfile['d2']
    d1_1d, d2_1d = interp1d(xh, z1), interp1d(xh, z2)
    d1 = lambda x, d0 : d1_1d(x[1])
    d2 = lambda x, d0 : d2_1d(x[1])

    d0_values = [5e-2]
    fileseries = 'roughness' + f'_dim{dim}'
    files = [fileseries]

else: # fracture for data from "comparison series"
    tangential = True
    symm = True
    d0_values = np.logspace(-1, -3, 7) if dim==2 else np.logspace(-1, -2, 4)
    d_ampl = lambda d0 : d0

    d1 = lambda x, d0 : d0 + 0.5 * d_ampl(d0) * ( np.sin(8 * np.pi * x[1])
     + np.sin(8 * np.pi * x[2]) )

    if symm:
        d2 = lambda x, d0 : d0 + 0.5 * d_ampl(d0) * ( np.sin(8 * np.pi * x[1])
         + np.sin(8 * np.pi * x[2]) )
    else:
        d2 = lambda x, d0 : d0 - 0.5 * d_ampl(d0) * ( np.sin(8 * np.pi * x[1])
         + np.sin(8 * np.pi * x[2]) )

    fileseries = 'comparison_' + ( 'tang' if tangential else 'perp' ) + '_' \
     + ( 'sym' if symm else 'asym' ) + f'_dim{dim}'
    files = [ ( fileseries + f'_{d0:0.1e}' ) for d0 in d0_values ]

# plot parameters
legend_entries = ['reference', 'model I', 'model I-R', 'model II', 'model II-R',
 r'model II (mean $d$)', 'model II-R (mean $d$)']
line_styles = [(0, (1, 1)), (0, (5, 3)), (0, (3, 3, 1, 3)),
 (0, (3, 3, 1, 3, 1, 3)), (0, (5, 5)), (0, (1, 2)), '-']
colors = ['crimson', 'limegreen', 'dodgerblue', 'darkorange', 'orchid',
 'saddlebrown', 'k']
marker_styles = ['x', 'o', '+', 'd', 'p', '8', '*']

################################################################################

def get_point_data(reader, value):
    index = -1
    sp = value.split(':')
    if len(sp) == 2:
        value = sp[0]
        index = int(sp[1])

    point_scalar_idx = -1
    pd = reader.GetOutput().GetPointData()
    for i in range(pd.GetNumberOfArrays()):
        if(value == pd.GetArrayName(i)):
            point_scalar_idx = i

    if (point_scalar_idx < 0):
        print('Value not found:' + value)
        return None
    else:
        point_scalar_vtk_array = \
         reader.GetOutput().GetPointData().GetArray(point_scalar_idx)
        pscalar_array = vtk_to_numpy(point_scalar_vtk_array)

    if index != -1:
        return pscalar_array[:, index]
    else:
        return pscalar_array


def get_triangulation(reader, fulldim=True):
    nodes_vtk_array = reader.GetOutput().GetPoints().GetData()
    nodes_numpy_array = vtk_to_numpy(nodes_vtk_array)
    x, y, z = \
     nodes_numpy_array[:, 0], nodes_numpy_array[:, 1], nodes_numpy_array[:, 2]

    tri_vtk_array = reader.GetOutput().GetCells().GetData()
    triangles = vtk_to_numpy(tri_vtk_array)

    ntri = reader.GetOutput().GetNumberOfCells()
    ia = np.zeros(ntri)
    ib = np.zeros(ntri)
    ic = np.zeros(ntri)

    offset = 0
    for i in range(0, ntri):
        n_points = triangles[offset]
        if n_points == 2:
            offset += 3
        else:
            ia[i] = triangles[offset + 1]
            ib[i] = triangles[offset + 2]
            ic[i] = triangles[offset + 3]
            offset += 4

    triangles = np.vstack((ia, ib, ic))
    triangles = triangles.T

    return x, y, z, triangles


def export_legend(legend, filename, expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def nanavg(*arrs):
    s0 = np.shape(arrs[0])

    nan_add = np.zeros_like(arrs[0])
    nan_cnt = np.zeros_like(arrs[0])
    nan_all = np.full_like(arrs[0], True, dtype=bool)

    for arr in arrs:
        assert np.shape(arr) == s0, "Arrays must have the same shape!"

        b = np.isnan(arr)

        nan_add += np.where(b, 0, arr)
        nan_cnt += b.astype(float)
        nan_all = nan_all & b

    nan_cnt[nan_all] = np.nan
    nan_add /= ( len(arrs) - nan_cnt )

    return nan_add

################################################################################

if not exists(output_dir):
    mkdir(output_dir)

error = np.zeros((len(legend_entries) - 1, len(d0_values)))

h = 1. / (1000. if dim==2 else 200.)
y = np.linspace(0, 1, num=1001 if dim==2 else 201, endpoint=True)
z = [0.] if dim==2 else y
ydata, zdata = np.meshgrid(y, z)

for k in range(len(d0_values)):
    filename = files[k]
    d0 = d0_values[k]

    vtp_files = [filename + '_mmdg1_trafoGamma0',
     filename + '_mmdg1_notrafoGamma0',
     filename + '_mmdg2_trafoGamma0',
     filename + '_mmdg2_notrafoGamma0']

    if roughness:
        vtp_files += [filename + '_mmdg2_dmean_trafoGamma0',
         filename + '_mmdg2_dmean_notrafoGamma0']

    print('\n' + filename)

    ############################################################################
    # read and integrate reference solution

    p = []

    vtk_file = join(vtk_dir, filename + '_dg0.vtu')
    vtk_reader = XMLUnstructuredGridReader(FileName=vtk_file)
    SetActiveSource(vtk_reader)
    plot_over_line = PlotOverLine()
    plot_over_line.Resolution = 100 if dim==2 else 10

    # calculate averaged pressure (reference)
    for y, z in zip(ydata.flat, zdata.flat):
        d1_ = d1([0.5, y, z], d0_values[k])
        d2_ = d2([0.5, y, z], d0_values[k])
        leftpoint = [0.5 - d1_, y, z]
        rightpoint = [0.5 + d2_, y, z]
        plot_over_line.Point1 = leftpoint
        plot_over_line.Point2 = rightpoint
        integrate_variables = IntegrateVariables(Input=plot_over_line)
        p += [integrate_variables.PointData['p'].GetRange()[0] / ( d1_ + d2_ )]

    ydata, zdata, p = np.array(ydata), np.array(zdata), np.array(p)

    if dim==3:
        p = np.reshape(p, ydata.shape)

    cutoff = np.where(p>1e3)
    n = np.size(cutoff)
    if n != 0:
        print(f'Warning: Ignoring {n} entr' + ('y' if n==1 else 'ies') + \
         ' of the reference solution')
        p[cutoff] = np.nan

    while np.isnan(p).any():
        p_avg = np.empty_like(p)

        if dim==2:
            p_avg[1:-1] = nanavg(p[2:], p[:-2])
            p_avg[0] = p[1]
            p_avg[-1] = p[-2]

        elif dim==3:
            p_avg[1:-1,1:-1] = \
             nanavg(p[2:,1:-1], p[:-2,1:-1], p[1:-1,2:], p[1:-1,:-2])
            p_avg[1:-1,0] = nanavg(p[2:,0], p[:-2,0], p[1:-1,1])
            p_avg[1:-1,-1] = nanavg(p[2:,-1], p[:-2,-1], p[1:-1,-2])
            p_avg[0,1:-1] = nanavg(p[0,2:], p[0,:-2], p[1,1:-1])
            p_avg[-1,1:-1] = nanavg(p[-1,2:], p[-1,:-2], p[-2,1:-1])
            p_avg[0,0] = nanavg(p[0,1], p[1,0])
            p_avg[-1,0] = nanavg(p[-1,1], p[-2,0])
            p_avg[0,-1] = nanavg(p[0,-2], p[1,-1])
            p_avg[-1,-1] = nanavg(p[-2,-1], p[-1,-2])

        else:
            raise ValueError("Invalid dimension!")

        idx = np.argwhere(np.isnan(p))
        p[idx] = p_avg[idx]

    ############################################################################
    # plots

    plt.rcParams.update({'font.size': 14})
    cmap = 'rainbow'

    if dim==2:
        ########################################################################
        # plot full-dimensional reference solution

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(vtk_file)
        reader.Update()

        fig_p, ax_p = plt.subplots()
        fig_u, ax_u = plt.subplots()

        p_data = get_point_data(reader, 'p')
        u_data = get_point_data(reader, 'u')

        x_tri, y_tri, z_tri, triangles = get_triangulation(reader)
        triangulation = Triangulation(x=x_tri, y=y_tri, triangles=triangles)

        v = np.linspace(0., 1., 100)
        xi, yi = np.meshgrid(v, v)
        ui_x = griddata((x_tri, y_tri), u_data[:,0], (xi, yi), method='cubic')
        ui_y = griddata((x_tri, y_tri), u_data[:,1], (xi, yi), method='cubic')
        ui_abs = np.sqrt(np.square(ui_x) + np.square(ui_y))

        im_u = ax_u.streamplot(xi, yi, ui_x, ui_y, density=2, arrowsize=0.7, \
         arrowstyle='->', color=ui_abs, cmap=cmap, linewidth=1,
         norm=Normalize(np.amin(ui_abs), np.amax(ui_abs)))
        fig_u.colorbar(im_u.lines)

        p_grid = griddata((x_tri, y_tri), p_data, (xi, yi), method='cubic')
        im_p = \
         ax_p.tricontourf(xi.flat, yi.flat, p_grid.flat, cmap=cmap, levels=51)
        fig_p.colorbar(im_p)

        y = np.linspace(0.,1.,201)
        x = np.full_like(y, 0.5)
        z = np.zeros_like(y)
        X = np.r_[[x],[y],[z]]
        d1_ = d1(X, d0_values[k])
        d2_ = d2(X, d0_values[k])
        ax_p.plot(0.5 - d1_, y, 'k-', 0.5 + d2_, y, 'k-', lw=0.75)
        ax_u.plot(0.5 - d1_, y, 'k-', 0.5 + d2_, y, 'k-', lw=0.75)

        ax_p.set_xlabel(r'$x_1$')
        ax_p.set_ylabel(r'$x_2$')
        ax_u.set_xlabel(r'$x_1$')
        ax_u.set_ylabel(r'$x_2$')
        ax_u.set_xlim([0.,1.])
        ax_u.set_ylim([0.,1.])

        fig_p.savefig(join(output_dir, filename + '_fulldim_pressure.pdf'), \
         bbox_inches='tight')
        fig_u.savefig(join(output_dir, filename + '_fulldim_velocity.pdf'), \
         bbox_inches='tight')
        plt.close(fig_p)
        plt.close(fig_u)

        ########################################################################
        # plot comparison and calculate error

        fig_sol, ax_sol = plt.subplots()
        fig_err, ax_err = plt.subplots()

        ax_sol.plot(ydata.flat, p, linestyle=line_styles[-1], color=colors[-1])

        csv_file = join(vtk_dir, 'tmp.csv')

        for i in range(len(vtp_files)):
            vtp_reader = \
             XMLPolyDataReader(FileName=join(vtk_dir, vtp_files[i] + '.vtp'))
            SetActiveSource(vtp_reader)
            plot_over_line = PlotOverLine()
            plot_over_line.Resolution = len(p) - 1
            plot_over_line.Point1 = [0.5, 0, 0]
            plot_over_line.Point2 = [0.5, 1, 0]
            SaveData(csv_file, plot_over_line, Precision=15,
             FieldAssociation='Point Data')

            csv_reader = pd.read_csv(csv_file)
            y = csv_reader['Points:1'].values
            p_gamma = csv_reader['p_Gamma'].values

            ax_sol.plot(y, p_gamma, linestyle=line_styles[i], color=colors[i])
            ax_err.plot(y, p - p_gamma, linestyle=line_styles[i],
             color=colors[i])
            error[i,k] = np.sqrt(h * np.nansum(np.square(p - p_gamma)))

        remove(csv_file)

        ax_sol.set_xlabel(r'$x_2$')
        ax_sol.set_ylabel(r'pressure  $p_\Gamma (x_2 )$')
        fig_sol.savefig(join(output_dir, filename + '_pressure.pdf'), \
         bbox_inches='tight')

        ax_err.set_xlabel(r'$x_2$')
        ax_err.set_ylabel(\
         r'error  $p_\Gamma^\mathrm{ref.}(x_2 ) - p_\Gamma (x_2 )$')
        fig_err.savefig(join(output_dir, filename + '_error.pdf'), \
         bbox_inches='tight')

        plt.close(fig_sol)
        plt.close(fig_err)

    ############################################################################

    elif dim==3:
        ########################################################################
        # read data and calculate error

        p_gamma_data = []
        triangulation = []

        for i in range(len(vtp_files)):
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(join(vtk_dir, vtp_files[i] + '.vtu'))
            reader.Update()

            x_tri, y_tri, z_tri, triangles = get_triangulation(reader)
            triangulation += \
             [ Triangulation(x=y_tri, y=z_tri, triangles=triangles) ]

            p_gamma_data += [ get_point_data(reader, 'p_Gamma') ]
            p_gamma = griddata(
             (y_tri, z_tri), p_gamma_data[i], (ydata, zdata), method='cubic')

            error[i,k] = \
             np.sqrt((h**2) * np.nansum(np.square(p - p_gamma).flat))

        ########################################################################
        # plot averaged reference and reduced solutions

        vmin, vmax = np.amin(p_gamma_data), np.amax(p_gamma_data)
        vmin = np.minimum(vmin, np.amin(p))
        vmax = np.maximum(vmax, np.amax(p))

        fig_p_gamma = plt.figure()
        ax_p_gamma = fig_p_gamma.gca()
        im_p_gamma = ax_p_gamma.tricontourf(ydata.flat, zdata.flat, p.flat,
         cmap='viridis', levels=np.linspace(vmin,vmax,51))
        fig_p_gamma.colorbar(im_p_gamma, format='%.2f')
        ax_p_gamma.set_xlabel(r'$x_2$')
        ax_p_gamma.set_ylabel(r'$x_3$')
        fig_p_gamma.savefig(join(output_dir,
         filename + '_reference_p_gamma.pdf'), bbox_inches='tight')
        plt.close(fig_p_gamma)

        for i in range(len(vtp_files)):
            fig_p_gamma = plt.figure()
            ax_p_gamma = fig_p_gamma.gca()
            im_p_gamma = ax_p_gamma.tricontourf(triangulation[i],
             p_gamma_data[i], cmap='viridis', levels=np.linspace(vmin,vmax,51))
            fig_p_gamma.colorbar(im_p_gamma, format='%.2f')
            ax_p_gamma.set_xlabel(r'$x_2$')
            ax_p_gamma.set_ylabel(r'$x_3$')
            fig_p_gamma.savefig(join(output_dir, vtp_files[i] + '.pdf'),
             bbox_inches='tight')
            plt.close(fig_p_gamma)

    else:
        raise ValueError("Invalid dimension!")

################################################################################
# plot L2 error

if len(files) > 1:
    fig_l2, ax_l2 = plt.subplots()

    for i in range(len(vtp_files)):
        print(error[i,:])
        ax_l2.loglog(d0_values, error[i,:], linestyle=line_styles[i], \
         marker=marker_styles[i], color=colors[i], fillstyle='none', \
         markersize=10)

    ax_l2.set_xlabel(r'$d_0$')
    ax_l2.set_ylabel(r'$L^2$ error')
    fig_l2.savefig(join(output_dir, fileseries + '_L2.pdf'), \
     bbox_inches='tight')
    plt.close(fig_l2)

# only to obtain legend as seperate plot
fig_legend, ax_legend = plt.subplots()
ax_legend.plot([1],[1], linestyle=line_styles[-1], color=colors[-1])

for i in range(len(vtp_files)):
    ax_legend.plot([1],[1], linestyle=line_styles[i], \
     marker=marker_styles[i], color=colors[i], fillstyle='none', markersize=10)

legend = ax_legend.legend(legend_entries, ncol=1, loc='best', \
 bbox_to_anchor=(2., 2.), framealpha=1, frameon=True)
export_legend(legend, join(output_dir, 'legend.pdf'))
