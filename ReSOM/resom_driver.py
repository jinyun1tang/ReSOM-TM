import numpy as np

import resom_micdyn as rmicdyn


#model initialization
dtime=3600.0
nsteps=120
varid=rmicdyn.varid()
reid=resom_mic.reid(varid)
substrate_input=np.zeros(varid.nsubstartes)
ystates=np.zeros(varid.ntvars)


substrate_input[varid.beg_polymer-varid.beg_substrates]=1.e-5*0.6
substrate_input[varid.beg_monomer-varid.beg_substrates]=1.e-5*0.4
#declare state variable array for each day
ystates=np.zeros((24,varid.ntvars))
#obtain the reaction matrix for the bulk reactions
csc_matrixp, csc_matrixd, csc_matrixs=resom_mic.set_reaction_matrix(varid, reid,resompar)
jj=0
First=True
ystates0=np.copy(ystates[0,:])
for nn in range(nsteps):
    #add external input
    ystates0[varid.mics_cum_cresp_co2]=0.0
    ystates0[varid.beg_mics_cummonomer:varid.end_mics_cummonomer+1]=0.0
    ystates[jj,:]=rmicdyn.resom_exinput(dtime, substrate_input, varid, ystates0)
    ystates0=np.copy(ystates[jj,:])
    #run microbial model dynamic core
    rrates0,mic_umonomer, rCO2_phys, newcell, newEnz, phyMortCell, mobileX=\
        rmicdyn.resom_dyncore(ystates0, varid, reid, resomPar)
    #update substrates from depolymerization and monomer uptake
    #rrates=np.array([y[0],y[1]*y[1],y[1]*y[2]])
    y=ystates0[0:varid.nbvars]
    ystates[0:varid.nbvars],rrates=rode.bgc_integrate_sparse(varid.nbvars, reid.nbreactions, dtime,\
        csc_matrixp, csc_matrixd, csc_matrixs, rrates0, y)    
    #update microbes

    if jj==23:
        if First:
            ystatesf=np.copy(ystates)
        else:
            ystatesf=np.concatenate((ystatesf,ystates))
        First=False
    jj=np.mod(jj+1,24)
import matplotlib
import matplotlib.pyplot as plt

print ystatesf[0,:]
print ystatesf[1,:]

tt=range(nsteps)
print ystatesf[:,varid.beg_polymer]

plt.plot(tt,ystatesf[:,varid.beg_polymer])
plt.plot(tt,ystatesf[:,varid.beg_monomer])
plt.legend(['Polymer','Monomer'])
plt.show()
