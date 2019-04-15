import numpy as np


import ReSOM.resom_mathlib as remath
from ReSOM.constants import Rgas



enz_n=np.array([270.,350.,452.,850.])

N_CH=6.
Delta_H_s=5330.
tsoil=np.arange(273.0,340.)
nel=len(tsoil)
fact=np.zeros(nel)
fact1=np.zeros(nel)
fact2=np.zeros(nel)
fact3=np.zeros(nel)

for j in range(nel):
    fact[j]=remath._fact(tsoil[j], enz_n[0], N_CH, Delta_H_s, Rgas)
    fact1[j]=remath._fact(tsoil[j], enz_n[1], N_CH, Delta_H_s, Rgas)
    fact2[j]=remath._fact(tsoil[j], enz_n[2], N_CH, Delta_H_s, Rgas)
    fact3[j]=remath._fact(tsoil[j], enz_n[3], N_CH, Delta_H_s, Rgas)

import matplotlib.pyplot as plt

plt.plot(tsoil, fact)
plt.plot(tsoil, fact1)
plt.plot(tsoil, fact2)
plt.plot(tsoil, fact3)

plt.legend(['270','350','452','850'])
plt.xlabel('Temperature (K)')
plt.ylabel('Active fraction of enzymes')

plt.show()
