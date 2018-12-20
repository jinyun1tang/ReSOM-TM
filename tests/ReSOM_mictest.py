import numpy as np

import ReSOM.resom_micdyn as resom_mic



varid=resom_mic.varid()

print 'number of variables=%d\n'%varid.nvars

resompar=resom_mic.resomPar(varid)

resompar.Kaff_Enz=resompar.Kaff_Enz+1.0
resompar.vmax_depoly=resompar.vmax_depoly+1.e-5
resompar.micYVM = resompar.micYVM+0.1
resompar.micYXE = resompar.micYXE+0.3
resompar.micYXV = resompar.micYXV+0.3
resompar.micPE_alpha=resompar.micPE_alpha+1.e-2
resompar.micX_h0 = resompar.micX_h0+0.01
fo2=0.1
resompar.micX_h = resompar.micX_h0 * fo2

resompar.micV_m = resompar.micV_m+1.e-4
resompar.micPerstV=resompar.micPerstV+1.e-5

ystates=np.array([0.,0.,0.,0.0,0.,0.1,0.2,0.0,0.0,0.0])

newcell,rCO2_m,rCO2_g,rCO2_e,newEnz,phyMortCell=resom_mic.cell_physioloy(ystates,resompar,varid)


print (newcell,rCO2_m,rCO2_g,rCO2_e,newEnz,phyMortCell)
