import numpy as np

import resom_micdyn as rmicdyn


#model initialization
dtime=3600.0
nsteps=120
varid=rmicdyn.varid()

substrate_input=np.zeros(varid.nsubstartes)
ystates=np.zeros(varid.nvars)


substrate_input[varid.beg_polymer-varid.beg_substrates]=1.e-5*0.6
substrate_input[varid.beg_monomer-varid.beg_substrates]=1.e-5*0.4

ystates=np.zeros((24,varid.nvars))
jj=0
First=True
ystates0=np.copy(ystates[0,:])
for nn in range(nsteps):
    ystates[jj,:]=rmicdyn.resom_exinput(dtime, substrate_input, varid, ystates0)
    ystates0=np.copy(ystates[jj,:])
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
