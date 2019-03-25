#cython: language_level=2
import numpy as np
cimport numpy as np
import cython
from libc.math cimport abs, M_PI
ctypedef np.float64_t dtype_t


cdef inline double max(double a, double b): return a if a >= b else b

class NotQuadEq(Exception):
  def __str__(self): return "NotQuadEq"

class NoValidSolution(Exception):
  def __str__(self): return "NoValidSolution"

class IncorrectFlux(Exception):
  def __str__(self): return "IncorrectFlux"

def _bunsen_o2(double temp_kelvin):
  """
    compute bunsen solubility for o2
  """
  cdef:
    double henry
    double bunsen
  henry=1.3e-3*np.exp(-1500.0*(1.0/temp_kelvin-1.0/298.15))
  bunsen=henry*temp_kelvin/12.2
  return bunsen

cdef inline double _get_o2diffusg(double temp):
  """
    compute gaseous o2 diffusivity, m2/s
  """
  return 1.61e-5*(temp/273.0)**1.82

cdef inline double _get_o2diffusw(double temp):
  """
    compute aqueous o2 diffusivity, m2/s
  """
  return 2.4e-9*temp/298.0

def _cal_o2_diffbw(double temp, double phig, double phiw,
  double taug, double tauw, double bo2):
  """
    return bulk and aqueous diffusivity of o2
    temp: temperature, kelvin
    taug: gaseous tortuosity
    tauw: aqueous tortuosity
    bo2: bunsen coefficient
  """
  cdef double diff_o2g, diffo2_w, diffo2_b
  diffo2_g=_get_o2diffusg(temp)
  diffo2_w=_get_o2diffusw(temp)
  diffo2_b=diffo2_g*taug*phig+diffo2_w*tauw*phiw*bo2
  return diffo2_b,diffo2_w

def _conds_o2(double dz, double taug, double tauw, double Ras,
  double tsoil, double vmsoi, double phig):
  """
  compute the conductance for soil-air o2 exchange
  """
  cdef:
    double bo2, diffo2_b, diffo2_w
    double conds
  bo2=_bunsen_o2(tsoil)
  diffo2_b,diffo2_w=_cal_o2_diffbw(tsoil, phig, vmsoi,
    taug, tauw, bo2)

  conds=1./(dz*(0.5*dz/diffo2_b+Ras))
  return conds


def _quadbigger(double a,double b,double c):
  """
    obtain the bigger solution of the quadratic equation
    ax^2 + bx + c
  """
  cdef:
    double delta
    double sol

  if a == 0.0 :
    raise NotQuadEq()
  delta = b*b - 4.0 * a * c
  if delta < 0.0:
    raise NoValidSolution()
  if a > 0.0 :
    sol=(-b+np.sqrt(delta))/(2.0*a)
  else:
    sol=(-b-np.sqrt(delta))/(2.0*a)
  return sol



cdef inline bint is_activek(double kd):
  """
    determine if kd is an active value
  """
  kd_infty=1.e10
  return (kd > 0. and kd < 0.9 * kd_infty)

@cython.boundscheck(False)
@cython.wraparound(False)
def supeca(np.ndarray[dtype_t,ndim=1] E, np.ndarray[dtype_t,ndim=1] S1, np.ndarray[dtype_t,ndim=1] S2,
  np.ndarray[dtype_t,ndim=2] K1, np.ndarray[dtype_t,ndim=2] K2, Py_ssize_t nE, Py_ssize_t nS1, Py_ssize_t nS2):
  """
    comupute enzyme-sbustrate complexes using supeca kinetics
  """

  cdef int k, j1, j2
  cdef double G12k, denorm

  if nE !=len(E) or nS1 != len(S1) or nS2 != len(S2):
    raise ValueError("check input array size")

  Fc1, Fr1, Fc2, Fr2, GS1E, GS2E=supecaflx(E, S1, S2, K1, K2, nE, nS1, nS2)
  cijk=np.zeros((nE, nS1, nS2))
  for k in range(nE):
    Fc12k=Fc1[k] + Fc2[k]
    for j1 in range(nS1):
      if is_activek(K1[k,j1]):
        for j2 in range(nS2):
          if is_activek(K2[k,j2]):
            G12k=GS1E[k,j1] + GS2E[k,j2]
            denorm=GS1E[k,j1]*GS2E[k,j2]*Fc12k/G12k + Fc12k \
                - (Fc1[k] * GS2E[k,j2]+GS1E[k,j1]*Fc2[k] - GS1E[k,j1] * GS2E[k,j2])/G12k
            cijk[k,j1,j2]=E[k]*S1[j1]/K1[k,j1] * S2[j2]/K2[k,j2] / denorm
  return cijk


def supecaflx(np.ndarray[dtype_t,ndim=1] E, np.ndarray[dtype_t,ndim=1] S1, np.ndarray[dtype_t,ndim=1] S2,
  np.ndarray[dtype_t,ndim=2] K1, np.ndarray[dtype_t,ndim=2] K2, Py_ssize_t nE, Py_ssize_t nS1, Py_ssize_t nS2):
  """
    compute the supeca fluxes
    K1(E,S1), K2(E, S2)
  """
  cdef int k, j1, j2

  Fc1,Fr1=ecaflx(E, S1, K1, nE, nS1)
  Fc2,Fr2=ecaflx(E, S2, K2, nE, nS2)
  GS1E=np.zeros((nE,nS1))
  GS2E=np.zeros((nE,nS2))
  for k in range(nE):
    for j1 in range(nS1):
      GS1E[k,j1]=Fc1[k] + Fr1[j1]
    for j2 in range(nS2):
      GS2E[k,j2]=Fc2[k] + Fr2[j2]
  return Fc1, Fr1, Fc2, Fr2, GS1E, GS2E

@cython.boundscheck(False)
@cython.wraparound(False)
def eca(np.ndarray[dtype_t,ndim=1] E, np.ndarray[dtype_t,ndim=1] S, np.ndarray[dtype_t,ndim=2] K,
  Py_ssize_t nE, Py_ssize_t nS):
  """
  compute enzyme substrate complexes using the eca kinetics
  E: enzyme, S: substrates, K: affinity parameter K(E,S)
  """
  cdef int i1,j1
  if nE !=len(E) or nS != len(S):
    raise ValueError("check input array size")

  cplx=np.zeros((nE,nS))
  Fc,Fr=ecaflx(E, S, K, nE, nS)
  for i1 in range(nE):
    for j1 in range(nS):
      if is_activek(K[i1,j1]):
        cplx[i1,j1] = E[i1]*S[j1]/K[i1,j1]/(1.0+Fc[i1]+Fr[j1])
  return cplx

def ecanorm(np.ndarray[dtype_t,ndim=1] E, np.ndarray[dtype_t,ndim=1] S, np.ndarray[dtype_t,ndim=2] K,
  Py_ssize_t nE, Py_ssize_t nS):
  """
  compute enzyme substrate complexes using the eca kinetics
  E: enzyme, S: substrates, K: affinity parameter K(E,S)
  """
  cdef int i1,j1
  if nE !=len(E) or nS != len(S):
    raise ValueError("check input array size")

  cplx=np.zeros((nE,nS))
  Fc,Fr=ecaflx(E, S, K, nE, nS)
  for i1 in range(nE):
    for j1 in range(nS):
      if is_activek(K[i1,j1]):
        cplx[i1,j1] = S[j1]/K[i1,j1]/(1.0+Fc[i1]+Fr[j1])
  return cplx

def ecaflx(np.ndarray[dtype_t,ndim=1] E, np.ndarray[dtype_t,ndim=1] S, np.ndarray[dtype_t,ndim=2] K,
  Py_ssize_t nE, Py_ssize_t nS):
  """
   compute the eca fluxes
  """

  Fc=np.zeros(nE)
  Fr=np.zeros(nS)
  for i1 in range(nE):
    for j1 in range(nS):
      if is_activek(K[i1,j1]):
        Fc[i1]=Fc[i1]+S[j1]/K[i1,j1]
        Fr[j1]=Fr[j1]+E[i1]/K[i1,j1]
  return Fc,Fr


def get_tscal(double rerr, double dt_scal):
  """
   obtain the time step scalar for adaptive ode
  """
  cdef double rerr_thr=1.e-4

  if rerr<0.5*rerr_thr:
    dt_scal = 2.0
    acc = True
  elif rerr < rerr_thr:
    dt_scal = 1.0
    acc = True
  elif rerr < 2.0*rerr_thr:
    dt_scal=0.5
  else:
    dt_scal=0.5
    acc = False
  return dt_scal, acc


def get_rerr(yc,yf):
  """
    get relative error of vector yc against yf
  """
  cdef double rerr
  cdef double rtmp
  cdef unsigned int dlen
  cdef unsigned int ii
  yc=np.array(yc,dtype=np.double)
  yf=np.array(yf,dtype=np.double)
  dlen=yc.shape[0]
  rerr = 0.
  for ii in range(dlen):
    rtmp = abs(yc[ii]-yf[ii])/(abs(yf[ii])+1.e-20)
    rerr = max(rtmp,rerr)
  return rerr

def _clapp_hornberg_par(double pct_sand, double pct_clay, double fom=0.):
    """Calculate Clapp Honberg parameters
    chb: b constant
    sat: soil porosity
    psisat: mm
    ksat: mm/s
    """
    cdef:
      double chb, sat, psisat, ksat
    chb=2.91+0.195*pct_clay
    sat=0.489-0.00126*pct_sand
    psisat=-10.0**(1.88-0.0131*pct_sand)
    ksat=0.0070556*10.0**(-0.884+0.0153*pct_sand)
    if fom > 0.:
        chb=chb*(1.-fom)+fom*2.7
        psisat=psisat*(1.-fom)+fom*(-10.3)
        sat=sat*(1.-fom)+fom*0.9
    return chb, sat,psisat,ksat

def _moldrup_tau(double sat, double chb, double s_sat):
  """calculate tortuosity using Moldrup's (2003) equation"""
  #equation (5)
  cdef:
    double epsi, theta, taug, tauw
  epsi=sat*(1.0-s_sat)
  theta=sat*s_sat
  taug=epsi*(1.0-s_sat)**(3.0/chb)
  #equation (3)
  tauw=theta*(s_sat+1.e-10)**(chb/3.0-1.0)
  return taug,tauw

cdef inline double _cosby_psi(double psisat, double chb,
  double s_sat):
  """compute the soil water potential
    s_sat: level of saturation
    chb: b constant
    psi: Pa
  """
  return np.fmax(psisat*(s_sat+1.e-20)**(-chb),-1.e8)*1.e1


def wfilm_thick(double psisat, double chb, double s_sat, double tsoi):
  """
  compute water film thickness, m
  """
  cdef:
    double rhow
    double hfus
  tfrz=273.15
  hfus=3.337e5
  rhow=1.e3
  if tsoi < tfrz:
    psi= hfus*(tfrz-tsoi)/tsoi * rhow  #Pa
  else:
    psi=_cosby_psi(psisat, chb, s_sat) #Pa
  delta = np.exp(-13.65-0.857*np.log(-psi*1.e-6))
  delta = np.fmax(delta,1.e-8)
  return delta


cdef double minp(np.ndarray[dtype_t,ndim=1] p, np.ndarray[dtype_t,ndim=1] v):
  """
    find the minimum of the nonzero p entries, with the entry determined
    by nonzero values of v
  """
  cdef int nvar, j
  cdef double ans

  nvar= p.shape[0]
  ans = 1.0
  for j in range(nvar):
    if v[j] != 0.0:
      ans=min(ans,p[j])
  return ans

def calc_state_pscal(double dtime, np.ndarray[dtype_t,ndim=1] ystate,
  np.ndarray[dtype_t,ndim=1] p_dt, np.ndarray[dtype_t,ndim=1] d_dt):
  """
    compute the p scaling vector to avoid negative state variables
  """
  cdef:
    int nprimvars, j
    bint lneg
    double yt, tmp, tiny_val, p_par
  tiny_val=1.e-14
  p_par=1.0-1.e-4

  lneg=False
  nprimvars=ystate.shape[0]
  pscal=np.ones((nprimvars))

  for j in range(nprimvars):
    yt = ystate[j]+(p_dt[j]+d_dt[j])*dtime
    if yt < tiny_val and d_dt[j]<0.:
      tmp = dtime * d_dt[j]
      pscal[j]=-(p_dt[j]*dtime+ystate[j])/tmp*p_par
      lneg=True
      if pscal[j] < 0.0:
        raise IncorrectFlux()

  return pscal, lneg

def calc_reaction_rscal(int nprimvars, int nr, np.ndarray[dtype_t,ndim=1] pscal,
  np.ndarray[dtype_t,ndim=2] csm_d):
  """
    compute the scaling factor for each reaction, using
    index information from csm_d
  """
  rscal=np.ones(nr)
  for j in range(nr):
    rscal[j] = minp(pscal, csm_d[0:nprimvars,j])
  return rscal

def calc_reaction_rscal_sparse(int nr, np.ndarray[dtype_t,ndim=1] pscal, \
  np.ndarray[int] csc_indices, np.ndarray[int] csc_indptr):
  """
    compute the scaling factor for each reaction, using
    index information from csm_d
  """
  rscal=np.ones(nr)
  for j in range(nr):
    rscal[j] = 1.0
    for k in range(csc_indptr[j],csc_indptr[j+1]):
      rscal[j]=min(rscal[j],pscal[csc_indices[k]])
  return rscal


def csr_matmul(np.ndarray[dtype_t] data, np.ndarray[int] indices, \
  np.ndarray[int] indptr, int nc, np.ndarray[dtype_t] v):
  """
  multiply a Compressed Sparse Row matrix with vector v
  """
  cdef int nr
  if nc != np.size(v):
    raise ValueError("check input array size")
  nr = np.size(indptr)-1
  result=np.zeros(nr)

  for j in range(nr):
    for k in range(indptr[j],indptr[j+1]):
      result[j] =result[j]+data[k]*v[indices[k]]
  return result

def _fact(double tsoi, double n, double N_CH, double Delta_H_s, double Rgas):
  cdef:
    double cp
    double Delta_Cp, Delta_G_E
    double Delta_S_s

  Delta_S_s=18.1
  T_Hs=373.6
  T_s=385.2

  Delta_Cp = -46.0+30.*(1.-1.54*n**(-0.268))*N_CH
  Delta_G_E= Delta_H_s-tsoi*Delta_S_s+Delta_Cp*((tsoi-T_Hs)-tsoi*np.log(tsoi/T_s))
  fact=1./(1.+np.exp(-n*Delta_G_E/(Rgas*tsoi)))

def _calKenz(double k2, double Dw, double S_radius, double f0):
  """
  compute the affinity parameter
  k2: maximum hydrolysis rate
  Dw: enzyme diffusivity in the soil
  S_radius: POC radius
  f0: fraction of active enzyme
  """
  cdef:
    double Kenz0
    double kx1w
  NA=6.0221e23
  kx1w=4.0 * M_PI * Dw * S_radius * NA
  Kenz0=k2*f0/kx1w
  return Kenz0, kx1w

def _calcKmic(double k2, double Dw, double cell_radius, double f0):
  """
  compute the affinity parameter
  k2, cell specific substrate uptake rate
  Dw, aqueous diffusivity of the substate molecule
  cell_radius, radius of the microbial cell
  f0, interception probability
  """
  cdef:
    double Kmic0   #mol substate molecule/m3
    double kx1w
    double NA
  NA=6.0221e23
  kx1w=4.0*M_PI*Dw*cell_radius*NA*f0
  Kmic0=k2/kx1w

  return Kmic0, kx1w

def _calvsmGamma(double Db, double Dw0, double rm, double flm, double kx1w, double Ncell):
  """
  NA avogadro number
  """
  cdef:
    double NA
    double cell_dens
    double vm
    double vsmGamma
  vm = 4./3.*M_PI*(rm)**3.
  NA=6.0221e23
  cell_dens=Ncell/(vm*NA)
  vsmGamma=1.0+(vm*flm/(Dw0*rm*(rm+flm))+vm/(Db*(rm+flm)))/(4.*M_PI)*kx1w*cell_dens
  return vsmGamma
