import numpy as np

import ReSOM.resom_micdyn as resom_mic
import ReSOM.resom_para as resom_para


varid=resom_para.varid()
reid=resom_para.reactionid(varid)

print ('number of bulk variables=%d, total variables=%d\n'%(varid.nbvars,varid.ntvars))
print ('number of reactions=%d\n'%reid.nbreactions)
resompar=resom_para.resomPar(varid)

resompar.Kaff_Enz=resompar.Kaff_Enz+1.0
resompar.vmax_depoly=resompar.vmax_depoly+1.e-5
resompar.micYVM = resompar.micYVM+0.1
resompar.micYXE = resompar.micYXE+0.3
resompar.micYXV = resompar.micYXV+0.3
resompar.micPE_alpha=resompar.micPE_alpha+1.e-2
resompar.micX_h0 = resompar.micX_h0+0.01
fo2=[0.81]
resompar.micX_h = resompar.micX_h0 * fo2

resompar.micV_m = resompar.micV_m+1.e-4
resompar.micPerstV=resompar.micPerstV+1.e-5

ystates=np.zeros(varid.ntvars)
ystates[varid.beg_microbeX]=0.1
ystates[varid.beg_microbeV]=0.1

dtime=3600.

newCell,rCO2_phys,newEnz,phyMortCell,mobileX=resom_mic.cell_physioloy(ystates,dtime, resompar,varid,fo2)


print (newCell,rCO2_phys,newEnz,phyMortCell,mobileX)
csc_matrixp, csc_matrixd, csc_matrixs=resom_mic.set_reaction_matrix(varid, reid,resompar)
print (mobileX-newCell-rCO2_phys-newEnz)
print (csc_matrixp.data)
print (csc_matrixd.data)
print (csc_matrixs.data)
