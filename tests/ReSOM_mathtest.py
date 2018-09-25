import numpy as np

import ReSOM.resom_mathlib as rmath


d=rmath._quadbigger(1.0,1.0,-2.0)

print 'd=%f'%d


x=[3,.4,6.,3,1.,0.8,15.]
y=[8.,3,1.,4,4.,8.,9.1]
print rmath.get_rerr(x,y)


print "test the ode integrator"

csm=np.array([[-0.04, 0., 1.e4],[0.04,-3.e7,-1.e4],[0.,3.e7,0.]])
csm_d=np.copy(csm)
csm_p=np.copy(csm)
csm_d[csm>0.0]=0.0
csm_p[csm<0.0]=0.0

y=np.array([1.0,1.e-14,1.e-14])

yt=np.copy(y)
dtime=0.01
t=0.
while True:
    rrates=np.array([y[0],y[1]*y[1],y[1]*y[2]])
    yn=rmath.bgc_integrate(3, 3, dtime, csm_p, csm_d, csm, rrates, y)
    yt=np.concatenate((yt, yn), axis=0)
    y=yn
    t=t+1

    if t>=100000.:
        break

yt=np.reshape(yt,(yt.shape[0]/3,3))


import matplotlib
import matplotlib.pyplot as plt

tv=np.arange(yt.shape[0])*dtime

plt.semilogx(tv,yt[:,0])
plt.semilogx(tv,yt[:,1]*1.e3)
plt.semilogx(tv,yt[:,2])

plt.legend(['y1','y2*1.e3','y3'])


plt.show()
