import numpy as np

import resom_mathlib as rmath

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
  csc_csm_p,csc_csm_d,csc_csm,rrates,ystates):
  """
    update the temporal derivative
  """
  it = 0
  itmax = 10

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
  ystate_new=dydt*dtime+ystates

  return ystate_new
