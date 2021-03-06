import numpy as np
import ReSOM.resom_mathlib as remath

from scipy.sparse import csr_matrix, csc_matrix
from ReSOM.constants import dzsoi, Ras, Rgas
from ReSOM.constants import pom_B,mC_amino


def resom_exinput(dtime, substrate_input, varid, ystates0):
	"""
	add substrates
	"""
	ystates=np.copy(ystates0)
	for j in range(varid.beg_orgsubstrates, varid.end_orgsubstrates+1):
		ystates[j]=ystates0[j]+dtime*substrate_input[j-varid.beg_orgsubstrates]
	return ystates

def cell_physioloy(ystates,dtime, resomPar,varid,fo2):
	"""
	 doing microbial cell physiology
	Input:
	ystates     : vector, model state variables
	dtime       : scalar, time step size
	resomPar    : structure, microbial parameters
	varid       : structure, variable labels
	fo2         : vector, fraction of binded oxygen acceptors
	Output:
	newCell     : vector, cell growth rate, shrinking when <0
 	rCO2_phys   : vector, physiological CO2 production rate
	newEnz      : vector, new enzyme synthesis rate
	phyMortCell : vector, microbial death rate
	mobileX     : vector, reserve mobilization rate
	"""
	newCell= np.zeros(varid.nmicrobes)     #cell growth or shrinking
	rCO2_m = np.zeros(varid.nmicrobes)     #co2 respiration associated with maintenance
	rCO2_g = np.zeros(varid.nmicrobes)     #co2 respiration associated with growth
	rCO2_e = np.zeros(varid.nmicrobes)     #co2 respiration associated with enzyme production
	rCO2_phys = np.zeros(varid.nmicrobes)  #co2 respiration associated with enzyme production
	newEnz = np.zeros(varid.nmicrobes)     #new enzyme production
	emortCell=np.zeros(varid.nmicrobes)    #cell mortality
	imortCell=np.zeros(varid.nmicrobes)
	phyMortCell=np.zeros(varid.nmicrobes)  #physiological mortalitiy
	mobileX=np.zeros(varid.nmicrobes)      #reserve mobilization rate
	for j in range(varid.nmicrobes):
		#loop over each microbes
		#determine reserve density
		if ystates[varid.beg_microbeV+j]>0.0:
			ee=ystates[varid.beg_microbeX+j]/ystates[varid.beg_microbeV+j]  #reserve density
			dmg=resomPar.micX_h0[j]*fo2[j]*ee
			dm = dmg-resomPar.micV_m[j]            #total avaiable reserve flux for non-maintenance use
			#print('dm=%e,dmg=%e,resomPar.micV_m[j]=%e'%(dm,dmg,resomPar.micV_m[j]))
			if dm <= 0.0:
				#check whether electron acceptor limitation is on by computing e-acceptor unlimited reserve mobilization
				dm0=resomPar.micX_h0[j]*ee-resomPar.micV_m[j]
				#print('dm0=%e'%dm0)
				if dm0<=0.0:
					#reserve limited
					#cell shrink because reserve limitation
					newCell[j]=dm/(ee+resomPar.micYVM[j])    # < 0.
					rCO2_m[j]=(resomPar.micX_h0[j]*fo2[j]-newCell[j])*ee-newCell[j] #amount co2 for maitanance
				else:
					#electron acceptor limitation is on
					#reserve is not limiting, maintenance deficit is
					#translated into mortality
					emortCell[j]=-dm  #enhanced mortality due to unfulfilled maintenance
					rCO2_m[j]=dmg
			else:
				#print "active growth, iterate using the secant method"
				newCell[j]=dm/(ee+(1.0+resomPar.micPE_alpha[j])/resomPar.micYXV[j])
				newEnz[j] =resomPar.micPE_alpha[j]*newCell[j]*resomPar.micYXE[j]/resomPar.micYXV[j]
				rCO2_m[j] =resomPar.micV_m[j]
				rCO2_e[j]=newEnz[j]*(1.0/resomPar.micYXE[j]-1.0)
				rCO2_g[j]=newCell[j]*(1.0/resomPar.micYXV[j]-1.0)
			imortCell[j] = resomPar.miciMort[j]                  #intrinsinc mortality
			phyMortCell[j]=(emortCell[j]+imortCell[j])* \
				ystates[varid.beg_microbeV+j]/(resomPar.micPerstV[j]+ystates[varid.beg_microbeV+j])
			#dormancy equivalent reduction in mortality
			#phyMortCell[j]=phyMortCell[j]*resomPar.fact_soil
			mobileX[j]=(resomPar.micX_h0[j]*fo2[j]-newCell[j])*ee
			rCO2_m[j]=rCO2_m[j]*ystates[varid.beg_microbeV+j]
			rCO2_g[j]=rCO2_g[j]*ystates[varid.beg_microbeV+j]
			rCO2_e[j]=rCO2_e[j]*ystates[varid.beg_microbeV+j]
			rCO2_phys[j]=(rCO2_m[j]+rCO2_g[j]+rCO2_e[j])
			#print('co2:m=%18.10e,g=%18.10e,e=%18.10e,mobx=%18.10e,newcell=%18.10e'%(rCO2_m[j]*dtime,rCO2_g[j]*dtime,\
			#	rCO2_e[j]*dtime,mobileX[j]*dtime,newCell[j]*dtime))
			yee=ee-dtime*mobileX[j]
			if yee<0.:
				fx=ee/(dtime*mobileX[j])*0.999
				mobileX[j]=mobileX[j]*fx
				newCell[j]=newCell[j]*fx
				rCO2_phys[j]=rCO2_phys[j]*fx
				newEnz[j]=newEnz[j]*fx

	return newCell,rCO2_phys,newEnz,phyMortCell,mobileX

def depolymerization(substrates, consumers, varid, resomPar, vmsoi):
	"""
	compute depolymerization
	Inputs:
		substrates: vector of substrates, polymers and mineral surfaces
		consumers: vector of consumers, enzymes
		varid: structure holding variable ids
		resomPar: structure holding model parameters
	Output:
		depolymer: matrix of depolymerization flux
	"""
	#enzyme hydrolysis, Fp

	#conver polymers into sorbtion surfaces
	substrates[0:varid.npolymers]=substrates[0:varid.npolymers]/pom_B
	#normalize enzyme concentration
	consumers[0:varid.nenzymes]=consumers[0:varid.nenzymes]/(mC_amino*vmsoi)
	consumers[0:varid.nenzymes]=consumers[0:varid.nenzymes]/resomPar.enz_n[0:varid.nenzymes]
	nS = np.size(substrates)
	nE = np.size(consumers)
	#collect matrix of affinity parameters
	Kffs=resomPar.Kaff_Enz
	sc_ij=remath.eca(consumers, substrates, Kffs, nE, nS)
	#enzyme degradation
	de_polymer=sc_ij[0:varid.nenzymes,0:varid.npolymers]*resomPar.vmax_depoly*pom_B
#	print('depolymer')
#	print(de_polymer)
	return de_polymer

def uptake_monomer(ystates, varid, resomPar, vmsoil):
	"""
	monomer uptake
	input:
		ystates  : vector, model state variables
		varid    : structure, variable labels
		resomPar : structure, microbial parameters
	output:
		mic_upmonomer : matrix, monomer uptake rate
		fo2           : vector, fraction of binded oxygen acceptor
	"""
	from ReSOM.constants import cmass_to_cell
	S1=np.concatenate((ystates[varid.beg_monomer:varid.end_monomer+1], \
		ystates[varid.beg_microbeX:varid.end_microbeX+1]))

	S1[varid.nmonomers:]=S1[varid.nmonomers:]/ystates[varid.beg_microbeV:varid.end_microbeV+1]
	#convert into mols of monomers
	S1[0:varid.nmonomers]=S1[0:varid.nmonomers]/(vmsoil*resomPar.catom_monomer[0:varid.nmonomers])

	S2=np.array([ystates[varid.oxygen]])
	E = np.zeros(varid.nmicrobes+varid.nmineralAs+1)

	E[0:varid.nmicrobes+varid.nmineralAs] =np.concatenate((ystates[varid.beg_microbeV:varid.end_microbeV+1], \
		ystates[varid.beg_mineralA:varid.end_mineralA+1]))

	E[varid.nmicrobes+varid.nmineralAs]=np.sum(ystates[varid.beg_polymer:varid.end_polymer+1])

	#conver into mol of cells
	E[0:varid.nmicrobes]=E[0:varid.nmicrobes]*cmass_to_cell/vmsoil
	nS1=np.size(S1)
	nS2=np.size(S2)
	nE =np.size(E)
	#K1(nE,nS1),
	K1=np.zeros((nE,nS1))
	K1=np.concatenate((resomPar.K_mic_monomer,resomPar.K_minerals_monomer))
	#K2(nE, nS2)
	K2=np.zeros((nE,nS2))
	K2[:,0]=resomPar.K_mic_oxygen
	#sc_ijk(nE,nS1,nS2)
	sc_ijk=remath.supeca(E, S1, S2, K1, K2, nE, nS1, nS2)
	mic_upmonomer=sc_ijk[0:varid.nmicrobes,0:varid.nmonomers,0]*resomPar.vmax_umonomer
	#print('monomer')
	#print(mic_upmonomer)
	#gaseous oxygen uptake
	S1=np.array([ystates[varid.oxygen]])
	E =ystates[varid.beg_microbeV:varid.end_microbeV+1]
	nS=np.size(S1)
	nE=np.size(E)
	Kffs=np.zeros((nE,nS))
	Kffs[:,:]=resomPar.K_mic_oxygen
	fo2=remath.ecanorm(E, S1, Kffs, nE, nS)
	return mic_upmonomer,fo2

def resom_dyncore(ystates, dtime, varid, reactionid, resomPar, vmsoi):
	"""
	resom microbial dynamics
	Inputs:
		ystates      : vector, model state variables
		dtime        : scalar, model time step
		reactionid   : structure, reaction labels
		resomPar     : structure, microbial parameters
		vmsoi        : volumetric soil moisture content
	Outputs:
		rrates       : vector reaction rates
    	mic_umonomer : matrix, monomer uptake rate
		rCO2_phys    : vector, physiological CO2 production rate
 		newcell      : vector, cell growth rate, actively growing if >0
		newEnz       : vector, enzyme production rate
		phyMortCell  : vector, mortality rate
 		mobileX      : vector, reserve mobilization rate
	"""
	rrates=np.zeros(reactionid.nbreactions)

	#collect substrates
	substrates = np.concatenate((ystates[varid.beg_polymer:varid.end_polymer+1], \
		ystates[varid.beg_mineralA:varid.end_mineralA+1]))
	#collect consumers
	consumers=np.concatenate((ystates[varid.beg_enzyme:varid.end_enzyme+1],ystates[varid.beg_monomer:varid.end_monomer+1]))

	#do enzymatic depolymerization
	de_polymer=depolymerization(substrates, consumers, varid, resomPar, vmsoi)
	#monomer uptake
	mic_umonomer,fo2=uptake_monomer(ystates, varid, resomPar, vmsoi)
	#cell physiology
	newcell, rCO2_phys, newEnz, phyMortCell, mobileX=cell_physioloy(ystates,dtime, resomPar,varid,fo2)
	#newcell=newcell*0.
	#rCO2_phys=rCO2_phys*0.
	#newEnz=newEnz*0.
	#phyMortCell=phyMortCell*0.
	#mobileX=mobileX*0.
	#trophic dynamics induced cell mortality
	#this will be place to plug in trophic dynamics related mortality
	#assuming all cells are lysed right away.
	#assemble the reactions
	#depolymerization
	#de_polymer=de_polymer*0.
	#mic_umonomer=mic_umonomer*0.
	for j in range(varid.npolymers):
		rrates[reactionid.reac_beg_depolymer+j]=np.sum(de_polymer[:,j])
	#monomer uptake
	for j in range(varid.nmonomers):
		rrates[reactionid.reac_beg_mics_upmonomer+j]=np.sum(mic_umonomer[:,j])
	#carbon consuming respiration
	rrates[reactionid.reac_mics_cresp_oxygen]=np.sum(rCO2_phys)
	return rrates, mic_umonomer, rCO2_phys, newcell, newEnz, phyMortCell, mobileX

def set_polymer_monomer_matrix():
	"""
	set up polymer composition as fractions of monomers
	"""
	#at present, one monomer is assumed to be associated with one polymer
	polymer_matrix=np.zeros((3,3))
	#polymer1->monomer1
	polymer_matrix[0,:]=[1.,0.,0.]
	#polymer2->monomer2
	polymer_matrix[1,:]=[0.,1.,0.]
	#polymer3->monomer3
	polymer_matrix[2,:]=[0.,0.,1.]
	return polymer_matrix

def set_reaction_matrix(varid, reactionid, resompar):
	"""
	define the reaction matrix, between polymers, monomers, and oxygen
	"""

	matrixs=np.zeros((varid.nbvars, reactionid.nbreactions))
	#reaction of depolymerization
	#polymer -> monomer
	poly2monomer_matrix=set_polymer_monomer_matrix()
	for j in range(varid.npolymers):
		polymer_id=varid.beg_polymer+j
		react_id=reactionid.reac_beg_depolymer+j
		matrixs[polymer_id,react_id]=-1.
		for k in range(varid.nmonomers):
			monomerid=varid.beg_monomer+k
			matrixs[monomerid,react_id]=poly2monomer_matrix[j,k]
	#monomer uptake
	#nomomer + (0.25*nosc+1)o2->co2 + energy
	for j in range(reactionid.reac_beg_mics_upmonomer,reactionid.reac_end_mics_upmonomer+1):
		k=j-reactionid.reac_beg_mics_upmonomer
		monomerid=varid.beg_monomer+k
		matrixs[monomerid,j]=-1.
		matrixs[varid.oxygen,j]=(-1.0+0.25*resompar.nosc_monomer[k])*(1.-resompar.YX_monomer[k])
		matrixs[varid.co2,j]=1.-resompar.YX_monomer[k]
		matrixs[varid.mics_cum_cresp_co2,j]=1.-resompar.YX_monomer[k]
		#the following have to be
		matrixs[varid.beg_mics_cummonomer+k,j]=1.
	#assuming growth and maintenance respiration has oxygen quotient of 1.
	matrixs[varid.oxygen,reactionid.reac_mics_cresp_oxygen]=-1.
	matrixs[varid.co2, reactionid.reac_mics_cresp_oxygen] = 1.
	matrixs[varid.mics_cum_cresp_co2, reactionid.reac_mics_cresp_oxygen]=1.
	matrixp=np.copy(matrixs)
	matrixd=np.copy(matrixs)
	matrixd[matrixs>0.0]=0.0
	matrixp[matrixs<0.0]=0.0
	csc_matrixp=csc_matrix(matrixp)
	csc_matrixd=csc_matrix(matrixd)
	csc_matrixs=csc_matrix(matrixs)
	return csc_matrixp, csc_matrixd, csc_matrixs


def updates_microbes(varid, reactionid, resompar, ystates, dtime, rrates0,rrates, \
	mic_umonomer,rCO2_phys,newCell,newEnz,phyMortCell,mobileX):
	"""
	update microbial biomass
	varid       : structure, variable lables
	reactionid  : structure, reaction labels
	resompar    : structure, microbial parameters
	ystates     : vector, model state variable vector
	dtime       : scalar, model time step
	rrates0     : vector, reference reaction rates
	rrates      : vector, updated reaction rates after accounting for substrate limitation
	mic_umonomer: matrix, reference monomer uptake rate
 	rCO2_phys   : vector, reference physiological CO2 production rate
	newCell     : vector, reference cell growth rate, shrinking when < 0
	newEnz      : vector, refnerece enzyme synthesis rate
	phyMortCell : vector, reference microbial mortality
	mobileX     : vector, reference reserve mobilization rate
	"""
	ystatesf=np.copy(ystates)
	#obtain the actual flux limiter
	#oxygen
	if rrates0[reactionid.reac_mics_cresp_oxygen] > 0.:
		#the strength of oxygen limitation
		foxygen=rrates[reactionid.reac_mics_cresp_oxygen]/rrates0[reactionid.reac_mics_cresp_oxygen]
	else:
		foxygen=0.0
	#organic monomers
	fmonomer=np.zeros(varid.nmonomers)
	for j in range(varid.nmonomers):
		if rrates0[reactionid.reac_beg_mics_upmonomer+j] > 0.0:
			fmonomer[j]=rrates[reactionid.reac_beg_mics_upmonomer+j]/rrates0[reactionid.reac_beg_mics_upmonomer+j]

	for j in range(varid.nmonomers):
		if fmonomer[j] > 0.0:
			for k in range(varid.nmicrobes):
				ystatesf[varid.beg_microbeX+k]=ystatesf[varid.beg_microbeX+k]+\
					dtime*mic_umonomer[k,j]*fmonomer[j]*resompar.YX_monomer[j]
	#update microbial physiological variables
	for k in range(varid.nmicrobes):
		#growth
		#decrease of reserve from mobilization
		ystatesf[varid.beg_microbeX+k]=ystatesf[varid.beg_microbeX+k]-\
			(dtime*mobileX[k]*foxygen)*ystatesf[varid.beg_microbeV+k]
		#increase of structure due to active growth
		ystatesf[varid.beg_microbeV+k]=ystatesf[varid.beg_microbeV+k]*\
			(1.+dtime*newCell[k]*foxygen)

		#reserve biomass goes to necromass
		#end of mortality reserve biomass
		ystate=ystatesf[varid.beg_microbeX+k]*np.exp(-dtime*phyMortCell[k])
		ystatesf[varid.polymer_necm]=ystatesf[varid.polymer_necm]-\
			(ystate-ystatesf[varid.beg_microbeX+k])*resompar.YX2necm[k]
		ystatesf[varid.beg_monomer]=ystatesf[varid.beg_monomer]-\
			(ystate-ystatesf[varid.beg_microbeX+k])*(1.-resompar.YX2necm[k])
		ystatesf[varid.beg_microbeX+k]=ystate

		#structural biomass goes to necromass
		#end of mortality structural biomass
		ystate=ystatesf[varid.beg_microbeV+k]*np.exp(-dtime*phyMortCell[k])
		ystatesf[varid.polymer_necm]=ystatesf[varid.polymer_necm]-ystate+ystatesf[varid.beg_microbeV+k]
		ystatesf[varid.beg_microbeV+k]=ystate

		#enzyme production
		ystatesf[varid.beg_enzyme+k]=ystatesf[varid.beg_enzyme+k]+\
			dtime*newEnz[k]*foxygen*ystatesf[varid.beg_microbeV+k]
		#compute decayed enzyme
		ystate=ystatesf[varid.beg_enzyme+k]*np.exp(-dtime*resompar.EnziDek[k])
		ystatesf[varid.polymer_denz]=ystatesf[varid.polymer_denz]-ystate+ystatesf[varid.beg_enzyme+k]
		ystatesf[varid.beg_enzyme+k]=ystate

	return ystatesf


def diffusion_gases(varid,resompar,y,dtime):
	"""
	gas diffusion
	"""
	ystates=np.copy(y)

	ystates[varid.oxygen]=resompar.o2_atm+\
		(y[varid.oxygen]-resompar.o2_atm)*np.exp(-dtime*resompar.conds_o2)

	return ystates
