import numpy as np
from netCDF4 import Dataset
import math

def load_forcing(nc_infile,ftype, dels=None):
    """
    load environmental forcing data from the specified nc_infile
    """
    watsat=0.4554859

    if ftype=='const':
        rh2osoi_vol=np.ones(8760)*0.35
        reff_vol=np.ones(8760)
        tsoil=np.ones(8760)*293.
    elif ftype=='tcyclic':
        rh2osoi_vol=np.ones(8760)*0.35
        reff_vol=np.ones(8760)
        time=np.linspace(0.,365.,8760)
        if dels is None:
            del1=1.
            del2=0.
        else:
            del1=dels[0]
            del2=dels[1]        
        tsoil=290-del1*10.*np.cos(2.*math.pi/365.*time)+del2*8.*np.sin(2.*math.pi*time)
    else:
        forcing=Dataset(nc_infile,"r")
        h2osoi=forcing.variables["h2osoi_liq"][:]
        airvol=forcing.variables["air_vol"][:]
        tsoi=forcing.variables["tsoi"][:]
        forcing.close()
        dzsoi=0.1
        rh2osoi_vol=h2osoi[0::2]/(dzsoi*1.e3*watsat)
        reff_vol=rh2osoi_vol+airvol[0::2]/watsat
        reff_vol[reff_vol>1.0]=1.
        tsoil=tsoi[0::2]
    return rh2osoi_vol,reff_vol,tsoil
