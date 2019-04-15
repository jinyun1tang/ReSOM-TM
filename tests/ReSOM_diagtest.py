import matplotlib.pyplot as plt


import ReSOM.resom_diagnose as rediag
import ReSOM.resom_forcing as reforc


ftype,dels='tcyclic',[1.,1.]


nc_file='/Users/jinyuntang/work/github/ReSOM-TM/sample_forcing.nc'
rh2osoi_vol,reff_vol, tsoil=reforc.load_forcing(nc_file, ftype, dels)

tsoil_day=rediag.get_daily_ts(tsoil)

plt.plot(tsoil_day)

plt.show()
