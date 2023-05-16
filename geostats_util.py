import numpy as np
import pandas as pd
from geostatspy import geostats
from scipy.interpolate import interp1d
import os
# %% Multi-point statistcis

def snesim(df_,xcol,ycol,zcol,vcol, nreal, NX, NY, NZ, NX_ti, NY_ti, NZ_ti, training_image, seed = 77777, num_category = 2, shale_ratio = 0.85,  output_file = 'snesim.out'):
        
    # make hard data file
    x = df_[xcol]
    y = df_[ycol]
    z = df_[zcol]
    v = df_[vcol]
    df_temp = pd.DataFrame({"X": x, "Y": y, "Z": z, "Var": v})
    Dataframe2GSLIB("data_temp.dat", df_temp)

    # make training image file
    with open("train.dat", "w") as f:
        f.write("train image \n")
        f.write("1 \n")
        f.write("value \n")
        for i in training_image.flatten():
            f.write(f"{int(i)} \n")
    
    with open("snesim.par", "w") as f:
        f.write("              Parameters for sneszim                                         \n")
        f.write("              ********************                                         \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETER:                                                        \n")
        f.write("data_temp.dat                 -file with data                              \n")
        f.write("1  2  3  4                    -  columns for X,Y,Z,vr,wt,sec.var.          \n")
        f.write(f"{num_category}               - number of categories                       \n")
        f.write(f"{str(np.arange(2)).split('[')[1].split(']')[0]} - number of categories    \n")
        f.write(f"{np.round(shale_ratio,3)} {np.round(1-shale_ratio,3)}                - (target) global pdf    \n")      
        f.write("0               -  use (target) vertical proportions (0=no, 1=yes)\n")
        f.write("vertprop.dat    - file with target vertical proportions\n")
        f.write("0               - servosystem parameter (0=no correction)\n")        
        f.write("-1              - debugging level: 0,1,2,3\n")
        f.write("snesim.dbg      - debugging file\n")
        f.write(f"{output_file}\n")
        f.write(f"{nreal} - number of realizations to generate \n")
        f.write(f"{NX} 0.5 1                   - nx,xmn,xsiz          \n")
        f.write(f"{NY} 0.5 1                             - ny,ymn,ysiz          \n")
        f.write(f"{NZ} 0.5 1                   - nz,zmn,zsiz          \n")
        f.write(f"{seed}                    - random number seed    \n")
        f.write("26                             \n")
        f.write("10                            \n")
        f.write("0 0                            \n")
        f.write("1 1                           \n")
        f.write("localprop.dat                     \n")
        f.write("0                             \n")
        f.write("rot_aff.dat                       \n")
        f.write("3               - number of affinity categories                       \n")
        f.write("1 1 1           - affinity factors (X,Y,Z) icat=1                         \n")
        f.write("1 0.5 1         - affinity factors (X,Y,Z) icat=2                          \n")
        f.write("1 2 1           - affinity factors (X,Y,Z) icat=3                          \n")
        f.write("5               - number of multiple grids                          \n")
        f.write("train.dat       - file with training image                          \n")
        f.write(f"{NX_ti} {NY_ti} {NZ_ti}       - training image dimensions: nxtr, nytr, nztr       \n")
        f.write("1               - column for training variable                          \n")
        f.write("10 10 5         - maximum search radii (hmax,hmin,hvert)                          \n")
        f.write("0 0 0           - angles for search ellipsoid (amax,amin,avert)                \n")

    os.system("snesim.exe < snesim.par")
    sim_array = GSLIB2ndarray_3D(output_file, 0 , nreal, NX, NY, NZ)
    return sim_array[0]


# %% two-points statistics
def sgsim_3d(nreal, df_, xcol, ycol, zcol, vcol, Val_range, nx_cells, ny_cells, nz, hsiz, vsiz, hmn_max,
             hmn_med, zmn_ver, seed, var, output_file):
    """Sequential Gaussian simulation, 2D wrapper for sgsim from GSLIB (.exe
    must be available in PATH or working directory).


    """
    x = df_[xcol]
    y = df_[ycol]
    z = df_[zcol]
    v = df_[vcol]
    var_min = Val_range[0]
    var_max = Val_range[1]
    df_temp = pd.DataFrame({"X": x, "Y": y, "Z": z, "Var": v})
    Dataframe2GSLIB("data_temp.dat", df_temp)

    nug = var["nug"]
    nst = var["nst"]
    it1 = var["it1"]
    cc1 = var["cc1"]
    azi1 = var["azi1"]
    dip1 = var["dip1"]
    hmax1 = var["hmax1"]
    hmed1 = var["hmed1"]
    hmin1 = var["hmin1"]
    it2 = var["it2"]
    cc2 = var["cc2"]
    azi2 = var["azi2"]
    dip2 = var["dip2"]
    hmax2 = var["hmax2"]
    hmed2 = var["hmed2"]
    hmin2 = var["hmin2"]
    max_range = max(hmax1, hmax2)
    max_range_v = hmin1
    hctab = int(max_range / hsiz) * 2 + 1
    vctab = int(max_range_v / vsiz) * 2 + 1

    with open("sgsim.par", "w") as f:
        f.write("              Parameters for SGSIM                                         \n")
        f.write("              ********************                                         \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETER:                                                        \n")
        f.write("data_temp.dat                 -file with data                              \n")
        f.write("1  2  3  4  0  0              -  columns for X,Y,Z,vr,wt,sec.var.          \n")
        f.write("-1.0e21 1.0e21                -  trimming limits                           \n")
        f.write("1                             -transform the data (0=no, 1=yes)            \n")
        f.write("none.trn                      -  file for output trans table               \n")
        f.write("0                             -  consider ref. dist (0=no, 1=yes)          \n")
        f.write("none.dat                      -  file with ref. dist distribution          \n")
        f.write("1  0                          -  columns for vr and wt                     \n")
        f.write(str(var_min) + " " + str(var_max) + "   zmin,zmax(tail extrapolation)       \n")
        f.write("1   " + str(var_min) + "      -  lower tail option, parameter              \n")
        f.write("1   " + str(var_max) + "      -  upper tail option, parameter              \n")
        f.write("0                             -debugging level: 0,1,2,3                    \n")
        f.write("nonw.dbg                      -file for debugging output                   \n")
        f.write(str(output_file) + "           -file for simulation output                  \n")
        f.write(str(nreal) + "                 -number of realizations to generate          \n")
        f.write(str(nx_cells) + " " + str(hmn_max) + " " + str(hsiz) + "                          \n")
        f.write(str(ny_cells) + " " + str(hmn_med) + " " + str(hsiz) + "                          \n")
        f.write(str(nz) + " " + str(zmn_ver) + " " + str(vsiz) + "                          \n")
        f.write(str(seed) + "                  -random number seed                          \n")
        f.write("0     8                       -min and max original data for sim           \n")
        f.write("10                            -number of simulated nodes to use            \n")
        f.write("1                             -assign data to nodes (0=no, 1=yes)          \n")
        f.write("1     3                       -multiple grid search (0=no, 1=yes),num      \n")
        f.write("0                             -maximum data per octant (0=not used)        \n")
        f.write(
            str(max_range) + " " + str(max_range) + " " + str(max_range_v) + " -maximum search  (hmax,hmin,vert) \n")
        f.write(str(azi1) + "   0.0   0.0       -angles for search ellipsoid                 \n")
        f.write(str(hctab) + " " + str(hctab) + " " + str(vctab)  + " -size of covariance lookup table        \n")
        f.write("1     0.60   1.0              - ktype: 0=SK,1=OK,2=LVM,3=EXDR,4=COLC        \n")
        f.write("none.dat                      -  file with LVM, EXDR, or COLC variable     \n")
        f.write("4                             -  column for secondary variable             \n")
        f.write(str(nst) + " " + str(nug) + "  -nst, nugget effect                          \n")
        f.write(str(it1) + " " + str(cc1) + " " + str(azi1) + "  " + str(dip1) + " 0.0 -it,cc,ang1,ang2,ang3\n")
        f.write(
            " " + str(hmax1) + "    " + str(hmed1) + "             " + str(
                hmin1) + "  - a_hmax, a_hmin, a_vert        \n")
        f.write(
            str(it2) + " " + str(cc2) + "   " + str(azi2) + "               " + str(
                dip2) + " 0.0 -it,cc,ang1,ang2,ang3\n")
        f.write(" " + str(hmax2) + " " + str(hmed2) + " " + str(hmin2) + " - a_hmax, a_hmin, a_vert        \n")

    os.system("sgsim.exe sgsim.par")
    sim_array = GSLIB2ndarray_3D(output_file, 0, nreal, nx_cells, ny_cells, nz)
    return sim_array[0]
# %% two-points statistics - indicator
def sisim_3d(nreal, df_, xcol, ycol, zcol, vcol, nx_cells, ny_cells, nz, hsiz, vsiz, hmn_max,
             hmn_med, zmn_ver, seed, var, output_file):
    """Sequential Gaussian simulation, 2D wrapper for sgsim from GSLIB (.exe
    must be available in PATH or working directory).
    """
    x = df_[xcol]
    y = df_[ycol]
    z = df_[zcol]
    v = df_[vcol]
    var_min = v.values.min()
    var_max = v.values.max()
    df_temp = pd.DataFrame({"X": x, "Y": y, "Z": z, "Var1": v, "Var2": 1-v})
    Dataframe2GSLIB("data_temp.dat", df_temp)

    nug = var["nug"]
    nst = var["nst"]
    it1 = var["it1"]
    cc1 = var["cc1"]
    azi1 = var["azi1"]
    dip1 = var["dip1"]
    hmax1 = var["hmax1"]
    hmed1 = var["hmed1"]
    hmin1 = var["hmin1"]
    it2 = var["it2"]
    cc2 = var["cc2"]
    azi2 = var["azi2"]
    dip2 = var["dip2"]
    hmax2 = var["hmax2"]
    hmed2 = var["hmed2"]
    hmin2 = var["hmin2"]
    max_range = max(hmax1, hmax2)
    max_range_v = 3
    hctab = int(max_range / hsiz) * 2 + 1

    with open("sisim.par", "w") as f:
        f.write("              Parameters for SISIM                                         \n")
        f.write("              ********************                                         \n")
        f.write("                                                                           \n")
        f.write("START OF PARAMETER:                                                        \n")
        f.write("0                                                                          \n")
        f.write("2                             -  Number of categories                      \n")
        f.write("0 1                           -  Categories                                \n")
        f.write("0.5 0.5                       -  Global CDF                                \n")
        f.write("data_temp.dat                 -  file with data                            \n")        
        f.write("1   2   3   4                 -   columns for X,Y,Z, and variable          \n")
        f.write("none.dat                      -  file with soft data                       \n")
        f.write("1  2  3  4  5  0              -  columns for X,Y,Z,vr,wt,sec.var.          \n")
        f.write("0                             -   Markov-Bayes simulation (0=no,1=yes)     \n")
        f.write("0.61  0.54                    -      calibration B(z) values               \n")
        f.write("-1.0e21 1.0e21                -  trimming limits                           \n")
        
        f.write("0.0   30.0                    -  minimum and maximum data value            \n")
        f.write("1      0.0                    -   lower tail option and parameter          \n")
        f.write("1      1.0                    -   middle     option and parameter          \n")
        f.write("1     30.0                    -   upper tail option and parameter          \n")
        f.write("none.dat                                             \n")
        f.write("3  0                          -  columns for vr and wt                     \n")
        f.write("0                             -debugging level: 0,1,2,3                    \n")
        f.write("sisim.dbg                     -file for debugging output                   \n")
        f.write("sisim.out                     -file for simulation output                  \n")
        f.write(str(nreal) + "                 -number of realizations to generate          \n")
        f.write(str(nx_cells) + " " + str(hmn_max) + " " + str(hsiz) + "                    \n")
        f.write(str(ny_cells) + " " + str(hmn_med) + " " + str(hsiz) + "                    \n")
        f.write(str(nz) + " " + str(zmn_ver) + " " + str(vsiz) + "                          \n")
        f.write(str(seed) + "                  -random number seed                          \n")
        f.write("20                            -number of simulated nodes to use            \n")
        f.write("20                            -number of simulated nodes to use            \n")
        f.write("3                            -number of simulated nodes to use             \n")
        f.write("1                             -assign data to nodes (0=no, 1=yes)          \n")
        f.write("0     3                       -multiple grid search (0=no, 1=yes),num      \n")
        f.write("0                             -maximum data per octant (0=not used)        \n")
        f.write(
            str(max_range) + " " + str(max_range) + " " + str(max_range_v) + " -maximum search  (hmax,hmin,vert) \n")
        f.write(str(azi1) + "   0.0   0.0       -angles for search ellipsoid                 \n")
        f.write(str(hctab) + " " + str(hctab) + " 1 -size of covariance lookup table        \n")
        f.write("0    2.5                      -0=full IK, 1=median approx. (cutoff)         \n")
        f.write("1                             -0=SK, 1=OK                                   \n")
        # Facies 2 (variogram)
        f.write(str(nst) + " " + str(nug) + "  -nst, nugget effect                          \n")
        f.write(str(it1) + " " + str(cc1) + " " + str(azi1) + "  " + str(dip1) + " 0.0 -it,cc,ang1,ang2,ang3\n")
        f.write(
            " " + str(hmax1) + "    " + str(hmed1) + "             " + str(
                hmin1) + "  - a_hmax, a_hmin, a_vert        \n")
        # Facies 3 (variogram)
        f.write(str(nst) + " " + str(nug) + "  -nst, nugget effect                          \n")
        f.write(str(it1) + " " + str(cc1) + " " + str(azi1) + "  " + str(dip1) + " 0.0 -it,cc,ang1,ang2,ang3\n")
        f.write(
            " " + str(hmax1) + "    " + str(hmed1) + "             " + str(
                hmin1) + "  - a_hmax, a_hmin, a_vert        \n")
    os.system("sisim.exe sisim.par")
    sim_array = GSLIB2ndarray_3D(output_file, 0, nreal, nx_cells, ny_cells, nz)
    return sim_array[0]

def create_sgs_model(dataframe, vario_dictionary, Val_name, Val_range, Num_real, horizon_grid_size = 1, vertical_grid_size = 1, grid_dim = [64,64,7], grid_mn=[0, 0, 0], seed = 77777):
    variogram = geostats.make_variogram_3D(**vario_dictionary)
    sgs_model = {
        "nreal": Num_real,
        "df_": dataframe,
        "Val_range": Val_range,
        "xcol": "X",
        "ycol": "Y",
        "zcol": "Z",
        "vcol": Val_name,
        "nx_cells": grid_dim[0],
        "ny_cells": grid_dim[1],
        "nz":  grid_dim[2],
        "hsiz": horizon_grid_size,
        "vsiz": vertical_grid_size,
        "hmn_max": grid_mn[0],
        "hmn_med": grid_mn[1],
        "zmn_ver": grid_mn[2],
        "seed": seed,
        "var": variogram,
        "output_file": "sgsim3d.out"
    }
    return sgs_model


def create_sis_model(dataframe, vario_dictionary, Val_name, Num_real, horizon_grid_size = 1, vertical_grid_size = 1, grid_dim = [64,64,7], grid_mn=[0, 0, 0], seed = 77777):
    variogram = geostats.make_variogram_3D(**vario_dictionary)
    sis_model = {
        "nreal": Num_real,
        "df_": dataframe,
        "xcol": "X",
        "ycol": "Y",
        "zcol": "Z",
        "vcol": Val_name,
        "nx_cells": grid_dim[0],
        "ny_cells": grid_dim[1],
        "nz":  grid_dim[2],
        "hsiz": horizon_grid_size,
        "vsiz": vertical_grid_size,
        "hmn_max": grid_mn[0],
        "hmn_med": grid_mn[1],
        "zmn_ver": grid_mn[2],
        "seed": seed,
        "var": variogram,
        "output_file": "sisim.out"
    }
    return sis_model

def create_snesim_model(dataframe, training_image, Val_name, Num_real,num_category =2, shale_ratio = 0.5, grid_dim = [64,64,7], grid_mn=[0.5, 0.5, 0.5], ti_grid_dim = [64,64,7], seed = 77777,horizon_grid_size = 1, vertical_grid_size = 1):
    dataframe ['X_i'] = np.round((dataframe ['X'] - grid_mn[0]) / horizon_grid_size)+0.5
    dataframe ['Y_j'] = np.round((dataframe ['Y'] - grid_mn[1]) / horizon_grid_size)+0.5
    dataframe ['Z_k'] = np.round((dataframe ['Z'] - grid_mn[2]) /  vertical_grid_size)+0.5
    
    snesim_model = {
        "nreal": Num_real,
        "df_": dataframe,
        "xcol": "X_i",
        "ycol": "Y_j",
        "zcol": "Z_k",
        "vcol": Val_name,
        "NX": grid_dim[0],
        "NY": grid_dim[1],
        "NZ":  grid_dim[2],
        "NX_ti": ti_grid_dim[0],
        "NY_ti": ti_grid_dim[1],
        "NZ_ti":  ti_grid_dim[2],
        "seed": seed,
        "num_category": num_category,
        "shale_ratio": shale_ratio,
        "training_image": training_image,
        "output_file": "snesim.out"
    }
    return snesim_model

def sgs_realizations(sgs_model_dict):
    sim = sgsim_3d(**sgs_model_dict)
    return sim

def sis_realizations(sis_model_dict):
    sim = sisim_3d(**sis_model_dict)
    return sim

def snesim_realizations(snemic_model_dict):
    sim = snesim(**snemic_model_dict)
    return sim

# %% GSLIB from Dr. Pyrcz's script
def GSLIB2ndarray_3D(data_file, kcol,nreal, nx, ny, nz):
    """Convert GSLIB Geo-EAS file to a 1D or 2D numpy ndarray for use with
    Python methods

    :param data_file: file name
    :param kcol: name of column which contains property
    :param nreal: Number of realizations
    :param nx: shape along x dimension
    :param ny: shape along y dimension
    :param nz: shape along z dimension
    :return: ndarray, column name
    """
    if nz > 1 and ny > 1:
        array = np.ndarray(shape = (nreal, nz, ny, nx), dtype=float, order="F")
    elif ny > 1:
        array = np.ndarray(shape=(nreal, ny, nx), dtype=float, order="F")
    else:
        array = np.zeros(nreal, nx)

    with open(data_file) as f:
        head = [next(f) for _ in range(2)]  # read first two lines
        line2 = head[1].split()
        ncol = int(line2[0])  # get the number of columns

        for icol in range(ncol):  # read over the column names
            head = next(f)
            if icol == kcol:
                col_name = head.split()[0]
        for ineal in range(nreal):		
            if nz > 1 and ny > 1:
                for iz in range(nz):
                    for iy in range(ny):
                        for ix in range(nx):
                            head = next(f)
                            array[ineal][iz][ny-1-iy][ix] = head.split()[kcol]    					
            elif ny > 1:
                for iy in range(ny):
                    for ix in range(0, nx):
                        head = next(f)
                        array[ineal][ny-1-iy][ix] = head.split()[kcol]
            else:
                for ix in range(nx):
                    head = next(f)
                    array[ineal][ix] = head.split()[kcol]
    return array, col_name
# %% GSLIB from Dr. Pyrcz's script
def Dataframe2GSLIB(data_file, df):
    """Convert pandas DataFrame to a GSLIB Geo-EAS file for use with GSLIB
    methods.

    :param data_file: file name
    :param df: dataframe
    :return: None
    """
    ncol = len(df.columns)
    nrow = len(df.index)

    with open(data_file, "w") as f:
        f.write(data_file + "\n")
        f.write(str(ncol) + "\n")

        for icol in range(ncol):
            f.write(df.columns[icol] + "\n")
        for irow in range(nrow):
            for icol in range(ncol):
                f.write(str(df.iloc[irow, icol]) + " ")
            f.write("\n")
# %% CDF mapping if needed
def CDF_mapping(Original, Projected):

    Original = Original.flatten() + np.random.normal(scale = 0.00001, size = Original.flatten().shape[0])
    Projected = Projected.flatten()
    
    Original_sort=np.sort(Original); 
    Original_idx = np.zeros(Original.shape[0])
    
    for ii in range(0,Original.shape[0]):
        Original_idx[np.array( Original==Original_sort[ii], dtype=bool)] = ii

    Projected_sort=np.sort(Projected)
    Original_CDF=(Original_idx+1)/float(Original_idx.max()+2)

    Projected_CDF=(np.arange(len(Projected_sort))+1)/float(len(Projected_sort)+2)
    
    f = interp1d(Projected_CDF.T,Projected_sort.T,bounds_error=False) # CDF mapping via linear interpolation

    Original_CDF[Original_CDF<=Projected_CDF.min()] = Projected_CDF.min()
    Original_CDF[Original_CDF>=Projected_CDF.max()] = Projected_CDF.max()
    
    Temp = f(Original_CDF)  
    
    Projected_result =Temp
    return np.array(Projected_result)

# %%
