import numpy as np
import resom_mathlib as remath

from scipy.sparse import csr_matrix, csc_matrix


class varid():
	def __init__(self):
		#substrates
		#polymers
		self.beg_orgsubstrates=0
		self.npolymers=3
		self.polymer_pom=0    #local id, polymeric organic matter
		self.polymer_denz=1   #denatured enzymed
		self.polymer_necm=2   #necromass from microbial lysis
		self.beg_polymer=0
		self.end_polymer=self.beg_polymer+self.npolymers-1
		#monomers
		self.nmonomers=self.npolymers
		self.beg_monomer=self.end_polymer+1
		self.monomer_pom=self.beg_monomer
		self.monomer_denz=self.monomer_pom+1
		self.monomer_necm=self.monomer_denz+1
		self.end_monomer=self.beg_monomer+self.nmonomers-1
		self.end_orgsubstrates=self.end_monomer
		self.norgsubstrates=self.end_orgsubstrates-self.beg_orgsubstrates+1

		self.oxygen=self.end_monomer+1
		self.co2   =self.oxygen+1
		#count total number of substrates

		#cumulative monomer uptake
		self.beg_mics_cummonomer=self.co2+1
		self.end_mics_cummonomer=self.beg_mics_cummonomer+self.nmonomers-1
		self.mics_cum_cresp_co2 =self.end_mics_cummonomer+1
		self.nbvars = self.mics_cum_cresp_co2+1
		#microbes
		#microbial reserve
		self.nmicrobes=1
		self.beg_microbeX=self.mics_cum_cresp_co2+1
		self.end_microbeX=self.beg_microbeX+self.nmicrobes-1
		#microbial biomass
		self.beg_microbeV=self.end_microbeX+1
		self.end_microbeV=self.beg_microbeV+self.nmicrobes-1
		#enzymes
		self.nenzymes=1
		self.beg_enzyme=self.end_microbeV+1
		self.end_enzyme=self.beg_enzyme+self.nenzymes-1
		#mineral surface, inert, only equilibrium sorption is considered
		self.nmineralAs=1
		self.beg_mineralA=self.end_enzyme+1
		self.end_mineralA=self.beg_mineralA+self.nmineralAs-1
		self.ntvars=self.end_mineralA+1

class reactionid():
	def __init__(self,varid):
		self.beg_depolymer=0
		self.end_depolymer=self.beg_depolymer+varid.npolymers-1
		self.beg_mics_upmonomer=self.end_depolymer+1
		self.end_mics_upmonomer=self.beg_mics_upmonomer+varid.nmonomers-1
		self.mics_cresp_oxygen = self.end_mics_upmonomer+1
		self.nbreactions =self.mics_cresp_oxygen+1

class resomPar():
	def __init__(self,varid):
		nE=varid.nenzymes
		nS=varid.npolymers+varid.nmineralAs
		self.Kaff_Enz=np.ones((nE,nS))+0.1                   #enzyme-substrate affinity
		self.vmax_depoly=np.ones((nE,varid.npolymers))*1.e-6 #depolymerization rate
		self.micYVM = np.zeros(varid.nmicrobes)+0.3          #maintenance yield from structural biomass
		self.micYXE = np.zeros(varid.nmicrobes)+0.5          #enzyme production yield from reserve
		self.micYXV = np.zeros(varid.nmicrobes)+0.5          #structural biomass yield from reserve
		self.micPE_alpha=np.zeros(varid.nmicrobes)+0.1       #maximum enzyme production rate
		self.micX_h = np.zeros(varid.nmicrobes)+0.1       #reserve mobilization rate
		self.micX_h0= np.zeros(varid.nmicrobes)+0.1
		self.micV_m = np.zeros(varid.nmicrobes)+1.e-7        #somatic maintenance
		self.EnziDek= np.zeros(varid.nmicrobes)+1.e-6
		self.miciMort=np.zeros(varid.nmicrobes)+1.e-7
		self.micPerstV=np.zeros(varid.nmicrobes)+0.01
		self.K_mic_monomer=np.zeros((varid.nmicrobes,varid.nmonomers))+1.e-2
		self.K_mins_monomer=np.zeros((varid.nmineralAs,varid.nmonomers))+1.0
		self.nosc_monomer=np.zeros(varid.nmonomers)    #nominal oxidation status
		self.YX_monomer=np.zeros(varid.nmonomers)+0.5
		self.vmax_umonomer=np.ones((varid.nmicrobes,varid.nmonomers))*1.e-6
		self.K_mic_oxygen=np.ones(varid.nmicrobes)*1.e-3
		self.YX2necm=np.zeros(varid.nmicrobes)+0.05    #fraction of lysed cell go to monomers
		self.diffus_o2=1.e-3
		self.diffus_co2=1.e-3
		self.o2_atm=8.
		self.co2_atm=4.e-3

def resom_exinput(dtime, substrate_input, varid, ystates0):
	"""
	add substrates
	"""
	ystates=np.copy(ystates0)
	for j in range(varid.beg_orgsubstrates, varid.end_orgsubstrates+1):
		ystates[j]=ystates0[j]+dtime*substrate_input[j-varid.beg_orgsubstrates]
	return ystates

def cell_physioloy(ystates,resomPar,varid,fo2):
	"""
	 doing microbial cell physiology
	"""
	newCell= np.zeros(varid.nmicrobes)  #cell growth or shrinking
	rCO2_m = np.zeros(varid.nmicrobes)  #co2 respiration associated with maintenance
	rCO2_g = np.zeros(varid.nmicrobes)  #co2 respiration associated with growth
	rCO2_e = np.zeros(varid.nmicrobes)  #co2 respiration associated with enzyme production
	rCO2_phys = np.zeros(varid.nmicrobes)  #co2 respiration associated with enzyme production
	newEnz = np.zeros(varid.nmicrobes)  #new enzyme production
	emortCell=np.zeros(varid.nmicrobes)  #cell mortality
	imortCell=np.zeros(varid.nmicrobes)
	phyMortCell=np.zeros(varid.nmicrobes)#physiological mortalitiy
	mobileX=np.zeros(varid.nmicrobes)    #reserve mobilization rate
	for j in range(varid.nmicrobes):
		#loop over each microbes
		#determine reserve density
		ee=ystates[varid.beg_microbeX+j]/ystates[varid.beg_microbeV+j]
		dm = resomPar.micX_h[j]*fo2[j]*ee-resomPar.micV_m[j]
		if dm <= 0.0:
			#check whether electron acceptor limitation is on
			dm0=resomPar.micX_h0[j]*ee-resomPar.micV_m[j]
			if dm0<=0.0:
				#cell shrink because reserve limitation
				newCell[j]=dm/(ee+resomPar.micYVM[j])
				rCO2_m[j]=(resomPar.micX_h[j]-newCell[j])*ee-newCell[j]
				rCO2_g[j]=0.0
				rCO2_e[j]=0.0
				newEnz[j]=0.0
			else:
				#electron acceptor limitation is on
				#reserve is not limiting, maintenance deficit is
				#translated into mortality
				emortCell[j]=dm  #enhanced mortality due to unfulfilled maintenance
		else:
			#print "active growth, iterate using the secant method"
			newCell[j]=dm/(ee+(1.0+resomPar.micPE_alpha[j])/resomPar.micYXV[j])
			newEnz[j] =resomPar.micPE_alpha[j]*newCell[j]*resomPar.micYXE[j]/resomPar.micYXV[j]
			rCO2_m[j] =resomPar.micV_m[j]
			rCO2_e[j]=newEnz[j]*(1.0/resomPar.micYXE[j]-1.0)
			rCO2_g[j]=newCell[j]*(1.0/resomPar.micYXV[j]-1.0)
		imortCell[j] = resomPar.miciMort[j]  #intrinsinc mortality
		phyMortCell[j]=(emortCell[j]+imortCell[j])* \
			ystates[varid.beg_microbeV+j]/(resomPar.micPerstV[j]+ystates[varid.beg_microbeV+j])
		mobileX[j]=(resomPar.micX_h[j]-newCell[j])*ee
		rCO2_phys[j]=rCO2_m[j]+rCO2_g[j]+rCO2_e[j]
	return newCell,rCO2_phys,newEnz,phyMortCell,mobileX

def depolymerization(ystates, varid, resomPar):
	#enzyme hydrolysis, Fp
	#collect substrates
	substrates = np.concatenate((ystates[varid.beg_polymer:varid.end_polymer+1], \
		ystates[varid.beg_mineralA:varid.end_mineralA+1]))
	#collect consumers
	consumers=ystates[varid.beg_enzyme:varid.end_enzyme+1]
	nS = np.size(substrates)
	nE = np.size(consumers)
	#collect matrix of affinity parameters
	Kffs=resomPar.Kaff_Enz
	sc_ij=remath.eca(consumers, substrates, Kffs, nE, nS)
	#enzyme degradation
	de_polymer=sc_ij[0:nE,0:varid.npolymers]*resomPar.vmax_depoly

	return de_polymer

def uptake_monomer(ystates, varid, resomPar):
	"""
	monomer uptake
	"""
	S1=ystates[varid.beg_monomer:varid.end_monomer+1]
	S2=np.array([ystates[varid.oxygen]])
	E =np.concatenate((ystates[varid.beg_microbeV:varid.end_microbeV+1], \
		ystates[varid.beg_mineralA:varid.end_mineralA+1]))
	nS1=np.size(S1)
	nS2=np.size(S2)
	nE =np.size(E)
	#K1(nE,nS1),
	K1=np.zeros((nE,nS1))
	K1=np.concatenate((resomPar.K_mic_monomer,resomPar.K_mins_monomer))
	#K2(nE, nS2)
	K2=np.zeros((nE,nS2))
	K2[:,0]=resomPar.K_mic_oxygen
	#sc_ijk(nE,nS1,nS2)
	sc_ijk=remath.supeca(E, S1, S2, K1, K2, nE, nS1, nS2)
	mic_upmonomer=sc_ijk[0:varid.nmicrobes,0:varid.nmonomers,0]*resomPar.vmax_umonomer

	S1=np.array([ystates[varid.oxygen]])
	E =ystates[varid.beg_microbeV:varid.end_microbeV+1]
	nS=np.size(S1)
	nE=np.size(E)
	Kffs=np.zeros((nE,nS))
	Kffs[:,:]=resomPar.K_mic_oxygen
	fo2=remath.ecanorm(E, S1, Kffs, nE, nS)

	return mic_upmonomer,fo2

def resom_dyncore(ystates, varid, reactionid, resomPar):
	"""
	define the resom microbial dynamics
	rrate
	"""
	rrates=np.zeros(reactionid.nbreactions)
	de_polymer=depolymerization(ystates, varid, resomPar)

	#monomer uptake
	mic_umonomer,fo2=uptake_monomer(ystates, varid, resomPar)
	#cell physiology
	newcell, rCO2_phys, newEnz, phyMortCell, mobileX=cell_physioloy(ystates,resomPar,varid,fo2)
	#trophic dynamics induced cell mortality
	#this will be place to plug in trophic dynamics related mortality

	#assuming all cells are lysed right away.

	#assemble the reactions
	#depolymerization
	for j in range(varid.npolymers):
		rrates[reactionid.beg_depolymer+j]=np.sum(de_polymer[:,j])
	#monomer uptake
	for j in range(varid.nmonomers):
		rrates[reactionid.beg_mics_upmonomer+j]=np.sum(mic_umonomer[:,j])
	#carbon consuming respiration
	rrates[reactionid.mics_cresp_oxygen]=np.sum(rCO2_phys)
	return rrates,mic_umonomer, rCO2_phys, newcell, newEnz, phyMortCell, mobileX

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
	define the reaction matrix
	"""

	matrixs=np.zeros((varid.nbvars, reactionid.nbreactions))
	#reaction of depolymerization
	#polymer -> monomer
	poly2monomer_matrix=set_polymer_monomer_matrix()
	for j in range(varid.npolymers):
		polymer_id=varid.beg_polymer+j
		react_id=reactionid.beg_depolymer+j
		matrixs[polymer_id,react_id]=-1.
		for k in range(varid.nmonomers):
			monomerid=varid.beg_monomer+k
			matrixs[monomerid,react_id]=poly2monomer_matrix[j,k]
	#monomer uptake
	#nomomer + (0.25*nosc+1)o2->co2 + energy
	for j in range(reactionid.beg_mics_upmonomer,reactionid.end_mics_upmonomer+1):
		k=j-reactionid.beg_mics_upmonomer
		monomerid=varid.beg_monomer+k
		matrixs[monomerid,j]=-1.
		matrixs[varid.oxygen,j]=(-1.0+0.25*resompar.nosc_monomer[k])*(1.-resompar.YX_monomer[k])
		matrixs[varid.co2,j]=resompar.YX_monomer[k]
		#the following have to be
		matrixs[varid.beg_mics_cummonomer+k,j]=1.
	#assuming growth and maintenance respiration has oxygen quotient of 1.
	matrixs[varid.oxygen,reactionid.mics_cresp_oxygen]=-1.
	matrixs[varid.co2, reactionid.mics_cresp_oxygen] = 1.
	matrixs[varid.mics_cum_cresp_co2, reactionid.mics_cresp_oxygen]=1.

	matrixp=np.copy(matrixs)
	matrixd=np.copy(matrixs)
	matrixd[matrixs>0.0]=0.0
	matrixp[matrixs<0.0]=0.0
	csc_matrixp=csc_matrix(matrixp)
	csc_matrixd=csc_matrix(matrixd)
	csc_matrixs=csc_matrix(matrixs)
	return csc_matrixp, csc_matrixd, csc_matrixs

def updates_microbes(varid, reactionid, resompar, ystates, dtime, rrates0,rrates, \
	mic_umonomer,rCO2_phys,newcell,newEnz,phyMortCell,mobileX):
	"""
	update microbial biomass
	"""
	ystatesf=np.copy(ystates)
	#obtain the actual flux limiter
	if rrates0[reactionid.mics_cresp_oxygen] > 0.:
		foxygen=rrates[reactionid.mics_cresp_oxygen]/rrates0[reactionid.mics_cresp_oxygen]
	else:
		foxygen=0.0
	fmonomer=np.zeros(varid.nmonomers)
	for j in range(varid.nmonomers):
		if rrates0[reactionid.beg_mics_upmonomer+j] > 0.0:
			fmonomer[j]=rrates[reactionid.beg_mics_upmonomer+j]/rrates0[reactionid.beg_mics_upmonomer+j]


	for j in range(varid.nmonomers):
		if fmonomer[j] > 0.0:
			for k in range(varid.nmicrobes):
				ystatesf[varid.beg_microbeX+k]=ystatesf[varid.beg_microbeX+k]+\
					dtime*mic_umonomer[k,j]*fmonomer[j]*resompar.YX_monomer[j]

	if foxygen > 0.:
		#update microbial physiological variables
		for k in range(varid.nmicrobes):
			#growth
			ystatesf[varid.beg_microbeX+k]=ystatesf[varid.beg_microbeX+k]-\
				(dtime*mobileX[k]*foxygen)*ystatesf[varid.beg_microbeV+k]
			ystatesf[varid.beg_microbeV+k]=ystatesf[varid.beg_microbeV+k]*\
				np.exp(dtime*newCell[k]*foxygen)
			#reserve goes to necromass
			ystate=ystatesf[varid.beg_microbeX+k]*np.exp(-dtime*phyMortCell[k])
			ystatesf[varid.polymer_necm]=ystatesf[varid.polymer_necm]-\
				(ystate-ystatesf[varid.beg_microbeX+k])*resompar.YX2necm[j]
			ystatesf[varid.beg_monomer]=ystatesf[varid.beg_monomer]-\
				(ystate-ystatesf[varid.beg_microbeX+k])*(1.-resompar.YX2necm[j])
			ystatesf[varid.beg_microbeX+k]=ystate
			#structural goes to necromass
			ystate=ystatesf[varid.beg_microbeV+k]*np.exp(-dtime*phyMortCell[j])
			ystatesf[varid.polymer_necm]=ystatesf[varid.polymer_necm]-ystate+ystatesf[varid.beg_microbeV+k]
			ystatesf[varid.beg_microbeV+k]=ystate
			#enzyme production
			ystatesf[varid.beg_enzyme+k]=ystatesf[varid.beg_enzyme+k]+\
				newEnz[j]*foxygen*ystatesf[varid.beg_microbeV+k]*dtime
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
		(y[varid.oxygen]-resompar.o2_atm)*np.exp(-dtime*resompar.diffus_o2)

	ystates[varid.co2]=resompar.co2_atm+\
		(y[varid.co2]-resompar.co2_atm)*np.exp(-dtime*resompar.diffus_co2)

	return ystates
