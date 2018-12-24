import numpy as np

import resom_micdyn as rmicdyn
import resom_ode as rode

#model initialization
dtime=3600.0
nsteps=24
varid=rmicdyn.varid()
reid=rmicdyn.reactionid(varid)
substrate_input=np.zeros(varid.norgsubstrates)
resompar=rmicdyn.resomPar(varid)
ystates=np.zeros(varid.ntvars)

fpoly=0.7
substrate_input[varid.polymer_pom-varid.beg_orgsubstrates]=1.e-5*fpoly
substrate_input[varid.monomer_pom-varid.beg_orgsubstrates]=1.e-5*(1.-fpoly)
#declare state variable array for each day
ystates=np.zeros((24,varid.ntvars))
for j in range(varid.nmicrobes):
    ystates[0,varid.beg_microbeX+j]=1.e-3
    ystates[0,varid.beg_microbeV+j]=1.e-3
co2_flx=np.zeros(24)
#obtain the reaction matrix for the bulk reactions
csc_matrixp, csc_matrixd, csc_matrixs=rmicdyn.set_reaction_matrix(varid, reid,resompar)
jj=0
First=True

for nn in range(nsteps):
    ystates0=np.copy(ystates[jj,:])
    #add external input
    ystates0[varid.mics_cum_cresp_co2]=0.0
    ystates0[varid.beg_mics_cummonomer:varid.end_mics_cummonomer+1]=0.0
    ystates0=rmicdyn.resom_exinput(dtime, substrate_input, varid, ystates0)
    #run microbial model dynamic core
    rrates0,mic_umonomer, rCO2_phys, newcell, newEnz, phyMortCell, mobileX=\
        rmicdyn.resom_dyncore(ystates0, varid, reid, resompar)
    #update substrates from depolymerization and monomer uptake
    #rrates=np.array([y[0],y[1]*y[1],y[1]*y[2]])
    y=ystates0[0:varid.nbvars]
    ystates[jj,0:varid.nbvars],rrates=rode.bgc_integrate_sparse(varid.nbvars, reid.nbreactions, dtime,\
        csc_matrixp, csc_matrixd, csc_matrixs, rrates0, y)
    #update microbes
    y=ystates[jj,:]
    ystates[jj,:]=rmicdyn.updates_microbes(varid, reid, resompar, y, dtime, rrates0,rrates, \
    	mic_umonomer,rCO2_phys,newcell,newEnz,phyMortCell,mobileX)
    y=ystates[jj,:]
    ystates[jj,:]=rmicdyn.diffusion_gases(varid,resompar,y,dtime)
    co2_flx[jj]=(ystates[jj,varid.co2]-y[varid.co2])/dtime
    if jj==23:
        if First:
            ystatesf=np.copy(ystates)
            co2_flxf=np.copy(co2_flx)
        else:
            ystatesf=np.concatenate((ystatesf,ystates))
            co2_flxf=np.concatenate((co2_flxf,co2_flx))
        First=False
    jj=np.mod(jj+1,24)
import matplotlib
import matplotlib.pyplot as plt

print ystatesf[0,:]
print ystatesf[1,:]

tt=range(nsteps)
print ystatesf[:,varid.beg_microbeV]
print co2_flxf
plt.plot(tt,ystatesf[:,varid.beg_polymer])
plt.plot(tt,ystatesf[:,varid.beg_monomer])
plt.legend(['Polymer','Monomer'])
plt.show()
