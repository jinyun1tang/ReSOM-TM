import numpy as np

import ReSOM.resom_micdyn as rmicdyn
import ReSOM.resom_ode as rode
import ReSOM.resom_para as resom_para
import ReSOM.resom_mathlib as remath
import ReSOM.resom_forcing as reforc
import ReSOM.resom_diagnose as rediag


#model initialization
dtime=3600.0   #time step size
nsteps=24*365*4  #number of integration steps

varid=resom_para.varid()
reid=resom_para.reactionid(varid)
resompar=resom_para.resomPar(varid)
substrate_input=np.zeros(varid.norgsubstrates)
pct_sand=50.0
pct_clay=20.0
envpar=resom_para.envPar()

envpar.chb, envpar.sat,envpar.psisat,envpar.ksat=remath._clapp_hornberg_par(pct_sand, pct_clay)

dels=[]
ftype='const'
#ftype,dels='tcyclic',[1.,1.]
#ftype=''
#define input data
nc_file='/Users/jinyuntang/work/github/ReSOM-TM/sample_forcing.nc'
rh2osoi_vol,reff_vol, tsoil=reforc.load_forcing(nc_file, ftype, dels)
vmsoi=rh2osoi_vol*envpar.sat
veffpore=reff_vol*envpar.sat

fpoly=0.7
cinput_flx=5.e-5
substrate_input[varid.polymer_pom-varid.beg_orgsubstrates]=cinput_flx*fpoly
substrate_input[varid.monomer_pom-varid.beg_orgsubstrates]=cinput_flx*(1.-fpoly)
#declare state variable array for each day
ystates=np.zeros((24,varid.ntvars))
for j in range(varid.nmicrobes):
    ystates[0,varid.beg_microbeX+j]=1.e-3
    ystates[0,varid.beg_microbeV+j]=1.e-3
    ystates[0,varid.beg_enzyme+j]=1.e-2
ystates[0,varid.beg_polymer]=10.
ystates[0,varid.beg_monomer]=1.

#obtain the reaction matrix for the bulk reactions
csc_matrixp, csc_matrixd, csc_matrixs=rmicdyn.set_reaction_matrix(varid, reid,resompar)
jj=0
First=True
ystates0=np.copy(ystates[jj,:])
#total_mass=rediag.total_cmass_sum(0,ystates0,varid)

import time

start = time.time()

resom_para.update_microbial_par(varid,resompar)
for nn in range(nsteps):
    #obtain the right forcing index
    tn=nn%8760
    vmsoit=np.max([vmsoi[tn],0.01])
    ystates[jj,:]=np.copy(ystates0)
    #add external input
    ystates0[varid.mics_cum_cresp_co2]=0.0
    ystates0[varid.beg_mics_cummonomer:varid.end_mics_cummonomer+1]=0.0
    resom_para.update_kinetics_par(varid, resompar, tsoil[tn], envpar, vmsoit,veffpore[tn])
    ystates0=rmicdyn.resom_exinput(dtime, substrate_input, varid, ystates0)
    #run microbial model dynamic core
    rrates0,mic_umonomer, rCO2_phys, newcell, newEnz, phyMortCell, mobileX=\
        rmicdyn.resom_dyncore(ystates0, dtime, varid, reid, resompar, vmsoit)
    #update substrates from depolymerization and monomer uptake
    y=np.copy(ystates0[0:varid.nbvars])
    ystates[jj,0:varid.nbvars],rrates=rode.bgc_integrate_sparse(varid.nbvars, reid.nbreactions, dtime,\
        csc_matrixp, csc_matrixd, csc_matrixs, rrates0, y)
    #update microbes
    y=np.copy(ystates[jj,:])
    ystates[jj,:]=rmicdyn.updates_microbes(varid, reid, resompar, y, dtime, rrates0,rrates, \
    	mic_umonomer,rCO2_phys,newcell,newEnz,phyMortCell,mobileX)
    y=np.copy(ystates[jj,:])
    ystates[jj,:]=rmicdyn.diffusion_gases(varid,resompar,y,dtime)
    if jj==23:
        if First:
            ystatesf=np.copy(ystates)
        else:
            ystatesf=np.concatenate((ystatesf,ystates))
        First=False
    ystates0=np.copy(ystates[jj,:])
    #total_mass=rediag.total_cmass_sum(nn,ystates0,varid)

    jj=jj+1
    if jj>=24:
        jj=np.mod(jj,24)

end = time.time()
print ('time=%f seconds\n'%(end - start))

import matplotlib
import matplotlib.pyplot as plt


#print ystatesf[0,:]
#print ystatesf[1,:]

tt=np.linspace(0,nsteps-1,nsteps)*dtime/86400.
ax1=plt.subplot(4, 1, 1)
ax1.plot(tt,ystatesf[:,varid.beg_polymer])

ax1.legend(['Polymer'])
ax2=plt.subplot(4, 1, 2)

ax2.plot(tt,ystatesf[:,varid.beg_microbeV])
ax2.plot(tt,ystatesf[:,varid.beg_microbeX])
ax2.legend(['MicrobeV','MicrobeX'])
ax3=plt.subplot(4, 1, 3)
ax3.plot(tt,ystatesf[:,varid.beg_monomer])
ax3.plot(tt,ystatesf[:,varid.beg_enzyme])
ax3.legend(['Monomer','Enzyme'])
ax4=plt.subplot(4, 1, 4)
daily_co2=rediag.get_daily_ts(ystatesf[:,varid.mics_cum_cresp_co2])/dtime
ax4.plot(np.arange(len(daily_co2)),daily_co2)
#ax4.plot(tt[:26000:-1],ystatesf[:26000:-1,varid.beg_monomer])
#ax4.plot(ystatesf[25000:25600,varid.mics_cum_cresp_co2]/dtime)
ax4.legend(['CO2 respiration'])
#ax2.plot(tt,ystatesf[:,varid.beg_microbeX])
#ax2.legend(['CO2','microbeX'])
plt.show()
