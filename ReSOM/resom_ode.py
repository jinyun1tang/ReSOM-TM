import numpy as np

from ReSOM import resom_mathlib as rmath

def bgc_integrate(nprimvars, nreactions, dtime,
  csm_p,csm_d,csm,rrates,ystates):
  """
    update the temporal derivative
  """
  it = 0
  itmax = 10

  while True:
    p_dt=np.matmul(csm_p[0:nprimvars,:], rrates)
    d_dt=np.matmul(csm_d[0:nprimvars,:], rrates)
    pscal, lneg=rmath.calc_state_pscal(dtime, ystates, p_dt, d_dt)
    if lneg and it <= itmax:
      rscal=rmath.calc_reaction_rscal(nprimvars, nreactions, pscal, csm_d)
      rrates=rscal*rrates
    else:
      dydt=np.matmul(csm,rrates)
      break
  ystate_new=dydt*dtime+ystates

  return ystate_new

def bgc_integrate_sparse(nprimvars, nreactions, dtime,
  csc_csm_p,csc_csm_d,csc_csm,rrates0,ystates):
  """
    update the temporal derivative
  Input:
  nprimvars  : integer scalar, number of primary variables
  nreactions : integer scalar, number of reactions
  dtime      : float scalar, time step size
  csc_csm_p  : sparse matrix, production
  csc_csm_d  : sparse matrix, destruction
  csc_csm    : sparse matrix, overall stoichiometry
  rrate0     : vector, reference reaction rates
  ystates    : vector, state variables
  Output:
  ystates_new: vector, updated state variable
  rrates     : vector, updated reaction rates
  """
  it = 0
  itmax = 10
  rrates=np.zeros(rrates0.shape)
  rrates=rrates0.copy()
  while True:
    p_dt=csc_csm_p.dot(rrates)
    d_dt=csc_csm_d.dot(rrates)
    pscal, lneg=rmath.calc_state_pscal(dtime, ystates, p_dt, d_dt)
    if lneg and it <= itmax:
      rscal=rmath.calc_reaction_rscal_sparse(nreactions, pscal, csc_csm_d.indices, csc_csm_d.indptr)
      rrates=rscal*rrates
    else:
      dydt=csc_csm.dot(rrates)
      break
    it=it+1
  ystate_new=np.zeros(ystates.shape)
  ystate_new=dydt*dtime+ystates
  return ystate_new,rrates
