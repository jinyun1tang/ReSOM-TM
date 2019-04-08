import numpy as np
from netCDF4 import Dataset


def load_forcing(nc_infile):
    """
    load environmental forcing data from the specified nc_infile
    """
    watsat=0.4554859
    forcing=Dataset(nc_infile,"r")
    h2osoi=forcing.variables["h2osoi_liq"][:]
    airvol=forcing.variables["air_vol"][:]
    tsoi=forcing.variables["tsoi"][:]
    forcing.close()
    dzsoi=0.1
    rh2osoi_vol=h2osoi[0::2]/(dzsoi*1.e3*watsat)
    reff_vol=rh2osoi_vol+airvol[0::2]/watsat
    reff_vol[reff_vol>1.0]=1.
    return rh2osoi_vol,reff_vol,tsoi[0::2]
