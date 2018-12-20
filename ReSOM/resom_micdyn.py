import numpy as np
import resom_mathlib as remath


class varid():
	def __init__(self):
		#substrates
		#polymers
		self.beg_substrates=0
		self.npolymers=3
		self.polymer_pom=0    #local id, polymeric organic matter
		self.polymer_denz=1   #denatured enzymed
		self.polymer_necm=2   #necromass from microbial lysis
		self.beg_polymer=0
		self.end_polymer=self.beg_polymer+self.npolymers-1
		#monomers
		self.nmonomers=self.npolymers
		self.beg_monomer=self.end_polymer+1
		self.end_monomer=self.beg_monomer+self.nmonomers-1
		#count total number of substrates
		self.end_substrates=self.end_monomer
		self.nsubstartes=self.end_substrates-self.beg_substrates+1
		#enzymes
		self.nenzymes=1
		self.beg_enzyme=self.end_monomer+1
		self.end_enzyme=self.beg_enzyme+self.nenzymes-1

		#microbes
		#microbial reserve
		self.nmicrobes=1
		self.beg_microbeX=self.end_enzyme+1
		self.end_microbeX=self.beg_microbeX+self.nmicrobes-1
		#microbial biomass
		self.beg_microbeV=self.end_microbeX+1
		self.end_microbeV=self.beg_microbeV+self.nmicrobes-1

		#mineral surface, inert, only equilibrium sorption is considered
		self.nmineralAs=1
		self.beg_mineralA=self.end_microbeV+1
		self.end_mineralA=self.beg_mineralA+self.nmineralAs-1
		self.oxygen=self.end_mineralA+1
		self.co2   =self.oxygen+1
		self.nvars=self.co2+1

class reactionid():
	def __init__(self,varid):
		self.beg_depolymer=0
		self.end_depolymer=self.beg_depolymer+varid.npolymers-1
		self.beg_mic_upmonomer=self.end_depolymer+1
		self.end_mic_upmonomer=self.beg_mic_upmonomer+varid.nmonomers-1
		self.beg_mic_mobileX=self.end_mic_upmonomer+1
		self.end_mic_mobileX=self.beg_mic_mobileX + varid.nmicrobes-1
		self.beg_mic_mort  =self.end_mic_mobileX+1
		self.end_mic_mort  =self.beg_mic_mort+varid.nmicrobes-1
		self.beg_pro_enzyme=self.end_mic_mort+1
		self.end_pro_enzyme=self.beg_pro_enzyme+varid.nenzymes-1
		self.beg_de_enzyme =self.end_pro_enzyme+1
		self.end_de_enzyme =self.beg_de_enzyme+varid.nenzymes-1
		self.beg_mic_upoxygen=self.end_de_enzyme+1
		self.end_mic_upoxygen=self.beg_mic_upoxygen+varid.nmicrobes-1
		self.beg_mic_proco2  = self.end_mic_upoxygen+1
		self.end_mic_proco2  = self.beg_mic_proco2+varid.nmicrobes-1
		self.diffus_oxygen = self.end_mic_proco2+1
		self.diffus_co2    = self.diffus_oxygen+1
		self.nreactions =self.diffus_co2+1


class resomPar():
	def __init__(self,varid):
		nE=varid.nenzymes
		nS=varid.npolymers+varid.nmineralAs
		self.Kaff_Enz=np.ones((nE,nS))                 #enzyme-substrate affinity
		self.vmax_depoly=np.ones((nE,varid.npolymers)) #depolymerization rate
		self.micYVM = np.zeros(varid.nmicrobes)        #maintenance yield from structural biomass
		self.micYXE = np.zeros(varid.nmicrobes)        #enzyme production yield from reserve
		self.micYXV = np.zeros(varid.nmicrobes)        #structural biomass yield from reserve
		self.micPE_alpha=np.zeros(varid.nmicrobes)       #maximum enzyme production rate
		self.micX_h = np.zeros(varid.nmicrobes)        #reserve mobilization rate
		self.micX_h0= np.zeros(varid.nmicrobes)
		self.micV_m = np.zeros(varid.nmicrobes)        #somatic maintenance
		self.miciMort=np.zeros(varid.nmicrobes)
		self.micPerstV=np.zeros(varid.nmicrobes)
		self.K_mic_monomer=np.zeros((varid.nmicrobes,varid.nmonomers))
		self.K_mins_monomer=np.zeros((varid.nmineralAs,varid.nmonomers))
		self.vmax_umonomer=np.ones((varid.nmicrobes,varid.nmonomers))
		self.K_mic_oxygen=np.zeros(varid.nmicrobes)

def resom_exinput(dtime, substrate_input, varid, ystates0):
	"""
	add substrates
	"""
	ystates=np.copy(ystates0)
	for j in range(varid.beg_substrates, varid.end_substrates+1):
		ystates[j]=ystates0[j]+dtime*substrate_input[j-varid.beg_substrates]
	return ystates

def cell_physioloy(ystates,resomPar,varid):
	"""
	 doing microbial cell physiology
	"""
	newCell= np.zeros(varid.nmicrobes)  #cell growth or shrinking
	rCO2_m = np.zeros(varid.nmicrobes)  #co2 respiration associated with maintenance
	rCO2_g = np.zeros(varid.nmicrobes)  #co2 respiration associated with growth
	rCO2_e = np.zeros(varid.nmicrobes)  #co2 respiration associated with enzyme production
	newEnz = np.zeros(varid.nmicrobes)  #new enzyme production
	emortCell=np.zeros(varid.nmicrobes)  #cell mortality
	imortCell=np.zeros(varid.nmicrobes)
	phyMortCell=np.zeros(varid.nmicrobes)#physiological mortalitiy
	mobileX=np.zeros(varid.nmicrobes)    #reserve mobilization rate
	for j in range(varid.nmicrobes):
		#loop over each microbes
		#determine reserve density
		ee=ystates[varid.beg_microbeX+j]/ystates[varid.beg_microbeV+j]
		dm = resomPar.micX_h[j]*ee-resomPar.micV_m[j]
		if dm <= 0.0:
			#check whether electron acceptor limitation is on
			dm0=resomPar.micX_h0[j]*ee-resomPar.micV_m[j]
			if dm0<=0.0:
				#cell shrink because reserve limitation
				newCell[j]=dm/(ee+resomPar.micYVM[j])
				rCO2_m[j]=resomPar.micV_m[j]*ystates[varid.beg_microbeV+j]
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
	return newCell,rCO2_m,rCO2_g,rCO2_e,newEnz,phyMortCell,mobileX

def depolymerization(ystates, varid, resomPar):
	#enzyme hydrolysis, Fp
	#collect substrates
	substrates = np.concatenate((ystates[varid.beg_polymer:varid.end_polymer+1], \
		ystates[varid.beg_mineralA:varid.end_mineralA+1]))
	#collect consumers
	consumers=ysates[varid.beg_enzyme:varid.end_enzyme+1]
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
	S2=ystates[varid.oxygen]
	E =np.concatenate((ystates[varid.beg_microbeV:varid.end_microbeV+1], \
		ystates[varid.beg_mineralA:varid.end_mineralA+1]))
	nS1=np.size(S1)
	nS2=np.size(S2)
	nE =np.size(E)
	#K1(nE,nS1),
	K1=np.zeros(nE,nS1)
	K1=np.concatenate((resomPar.K_mic_monomer,resomPar.K_mins_monomer))
	#K2(nE, nS2)
	K2=np.zeros(nE,nS2)
	K2[:,0]=K_mic_oxygen
	#sc_ijk(nE,nS1,nS2)
	sc_ijk=remath.supeca(E, S1, S2, K1, K2, nE, nS1, nS2)
	mic_upmonomer=sc_ijk[0:varid.nmicrobes,0:varid.nmonomers,0]*resomPar.vmax_umonomer
	return mic_upmonomer

def resom_dyncore(ystates, nreactions, dtime, varid, resomPar):
	"""
	define the resom microbial dynamics
	rrate
	"""
	rrates=np.zeros(nreactions)
	de_polymer=depolymerization(ystates, varid, resomPar)
	#monomer uptake
	mic_umonomer=uptake_monomer(ystates, varid, resomPar)
	#cell physiology
	newcell, rCO2_m, rCO2_g, rCO2_e, newEnz, phyMortCell, mobileX=cell_physioloy(ystates,resomPar,varid)
	#trophic dynamics induced cell mortality
	#this will be place to plug in trophic dynamics related mortality

	#assuming all cells are lysed right away.

	#oxygen diffusion

	return rrates

def set_reaction_matrix():

	return matrixp, matrixd
