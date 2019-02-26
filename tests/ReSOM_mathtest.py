import numpy as np

from ReSOM import resom_mathlib as rmath

import ReSOM.resom_ode as rode

from scipy.sparse import csr_matrix, csc_matrix

d=rmath._quadbigger(1.0,1.0,-2.0)

print ('d=%f'%d)


x=[3,.4,6.,3,1.,0.8,15.]
y=[8.,3,1.,4,4.,8.,9.1]
print(rmath.get_rerr(x,y))


print ("test the ode integrator")

csm=np.array([[-0.04, 0., 1.e4],[0.04,-3.e7,-1.e4],[0.,3.e7,0.]])
csm_d=np.copy(csm)
csm_p=np.copy(csm)
csm_d[csm>0.0]=0.0
csm_p[csm<0.0]=0.0

csc_csm_p=csc_matrix(csm_p)
csc_csm_d=csc_matrix(csm_d)
csc_csm=csc_matrix(csm)



import time

start = time.time()

y=np.array([1.0,1.e-14,1.e-14])

yt=np.copy(y)
dtime=0.01
t=0.

while True:
    rrates=np.array([y[0],y[1]*y[1],y[1]*y[2]])
    yn=rode.bgc_integrate(3, 3, dtime, csm_p, csm_d, csm, rrates, y)
    yt=np.concatenate((yt, yn), axis=0)
    y=yn
    t=t+1
    if t>=100000.:
        break

yt=np.reshape(yt,(int(yt.shape[0]/3),3))
end = time.time()
print(end - start)


start = time.time()

y=np.array([1.0,1.e-14,1.e-14])

yt=np.array([0.0,0.,0.])
dtime=0.01
t=0.
yt[:]=y
while True:
    rrates=np.array([y[0],y[1]*y[1],y[1]*y[2]])
    yn,rrates=rode.bgc_integrate_sparse(3, 3, dtime, csc_csm_p, csc_csm_d, csc_csm, rrates, y)
    yt=np.concatenate((yt, yn), axis=0)
    y=yn
    t=t+1
    if t>=100000.:
        break

yt=np.reshape(yt,(int(yt.shape[0]/3),3))
end = time.time()
print(end - start)

import matplotlib
import matplotlib.pyplot as plt

tv=np.arange(yt.shape[0])*dtime

plt.semilogx(tv,yt[:,0])
plt.semilogx(tv,yt[:,1]*1.e3)
plt.semilogx(tv,yt[:,2])

plt.legend(['y1','y2*1.e3','y3'])


plt.show()


d=np.array([[1., 0., 2.],[0., 3., 0.],[4., 5., 6.]])
d1=csr_matrix(d)
v=np.array([1.,1.,1.])
d2=rmath.csr_matmul(d1.data, d1.indices,d1.indptr, 3, v)

print (d2)
print (d1.dot(v))

d3=csc_matrix(d)
print (d3.dot(v))
