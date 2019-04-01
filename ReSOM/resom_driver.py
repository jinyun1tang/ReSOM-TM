import numpy as np

import ReSOM.resom_micdyn as rmicdyn
import ReSOM.resom_ode as rode
import ReSOM.resom_para as resom_para
import ReSOM.resom_mathlib as remath
#model initialization
dtime=3600.0   #time step size
nsteps=24*365  #number of integration steps
varid=resom_para.varid()
reid=resom_para.reactionid(varid)
resompar=resom_para.resomPar(varid)
substrate_input=np.zeros(varid.norgsubstrates)
pct_sand=50.0
pct_clay=20.0
envpar=resom_para.envPar()

envpar.chb, envpar.sat,envpar.psisat,envpar.ksat=remath._clapp_hornberg_par(pct_sand, pct_clay)

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
ystates0=np.copy(ystates[jj,:])

tsoil=np.zeros(nsteps)+298.
vmsoi=np.zeros(nsteps)+0.3
veffpore=np.zeros(nsteps)+0.5
import time

start = time.time()

for nn in range(nsteps):
    ystates[jj,:]=ystates0
    #add external input
    ystates0[varid.mics_cum_cresp_co2]=0.0
    ystates0[varid.beg_mics_cummonomer:varid.end_mics_cummonomer+1]=0.0
    resom_para.update_kinetics_par(varid, resompar, tsoil[nn], envpar, vmsoi[nn],veffpore[nn])
    ystates0=rmicdyn.resom_exinput(dtime, substrate_input, varid, ystates0)
    #run microbial model dynamic core
    rrates0,mic_umonomer, rCO2_phys, newcell, newEnz, phyMortCell, mobileX=\
        rmicdyn.resom_dyncore(ystates0, dtime, varid, reid, resompar)
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
    co2_flx[jj]=(ystates[jj,varid.co2]-y[varid.co2])/dtime
    if jj==23:
        if First:
            ystatesf=np.copy(ystates)
            co2_flxf=np.copy(co2_flx)
        else:
            ystatesf=np.concatenate((ystatesf,ystates))
            co2_flxf=np.concatenate((co2_flxf,co2_flx))
        First=False
    ystates0=np.copy(ystates[jj,:])
    jj=jj+1
    if jj>=24:
        jj=np.mod(jj,24)

end = time.time()
print ('time=%f seconds\n'%(end - start))

import matplotlib
import matplotlib.pyplot as plt


#print ystatesf[0,:]
#print ystatesf[1,:]

tt=range(nsteps)
ax1=plt.subplot(2, 1, 1)
ax1.plot(tt,ystatesf[:,varid.beg_polymer])
ax1.plot(tt,ystatesf[:,varid.beg_monomer])
ax1.legend(['Polymer','Monomer'])
ax2=plt.subplot(2, 1, 2)
ax2.plot(tt,ystatesf[:,varid.beg_enzyme])
#ax2.plot(tt,ystatesf[:,varid.beg_microbeX])
#ax2.legend(['CO2','microbeX'])
plt.show()
