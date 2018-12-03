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
		self.nmonomers=1
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

class resomPar():
	def __init__(self,varid):
		nE=varid.nenzymes
		nS=varid.npolymers+varid.nmineralAs
		self.Kaff_Enz=np.ones((nE,nS))                 #enzyme-substrate affinity
		self.vmax_depoly=np.ones((nE,varid.npolymers)) #depolymerization rate
		self.micYVM = np.zeros(varid.nmicrobes)        #maintenance yield from structural biomass
		self.micYXE = np.zeros(varid.nmicrobes)        #enzyme production yield from reserve
		self.micYXV = np.zeros(varid.nmicrobes)        #structural biomass yield from reserve
		self.micPE_max=np.zeros(varid.nmicrobes)       #maximum enzyme production rate
		self.micX_h = np.zeros(varid.nmicrobes)        #reserve mobilization rate
		self.micX_h0= np.zeros(varid.nmicrobes)
		self.micV_m = np.zeros(varid.nmicrobes)        #somatic maintenance

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
	newCell= np.zeros(varid.nmicrobes)
	rCO2_m = np.zeros(varid.nmicrobes)
	rCO2_g = np.zeros(varid.nmicrobes)
	rCO2_e = np.zeros(varid.nmicrobes)
	newEnz = np.zeros(varid.nmicrobes)
	for j in range(varid.nmicrobes):
		#loop over each microbes
		#determine reserve density
		ee=ystates[varid.beg_microbeX+j]/ystates[varid.beg_microbeV+j]
		dm = resomPar.micX_h[j]*ee-resomPar.micV_m[j]
		if dm <= 0.0:
			#check whether electron acceptor limitation is on
			dm0=resomPar.micX_h0[j]*ee-resomPar.micV_m[j]
			if dm0<=0.0:
				#cell shrink
				newCell[j]=dm/(ee+resomPar.micYVM[j])
				rCO2_m[j]=resomPar.micV_m[j]*ystates[varid.beg_microbeV+j]
				rCO2_g[j]=0.0
				rCO2_e[j]=0.0
				newEnz[j]=0.0
			else:
				#electron acceptor limitation is on

		else:
			#active growth, iterate using the secant method
			newCell0=dm/(ee+1./resomPar.micYXV[j])
			newCell[j]=newCell0
			newEnz[j] = 0.0
			residual0=0.0
			residual=0.0
			if resomPar.micPE_max[j] > 0.0:
				oldCell=newCell[j]
				h0=(resomPar.micX_h[j]-newCell[j])*ee-resomPar.micV_m[j]
				newEnz[j]=resomPar.micPE_max[j]*h0/(resomPar.micPE_max[j]+h0)
				residual0=(resomPar.micX_h[j]-newCell[j])*ee-resomPar.micV_m[j]- \
						newCell[j]/resomPar.micYXV[j]-newEnz[j]/resomPar.micYXE[j]
				#find a new value
				newCell[j]=newCell0+newEnz[j]/resomPar.micYXE[j]/(ee+1.0/resomPar.micYXV[j])
				h0=(resomPar.micX_h[j]-newCell[j])*ee-resomPar.micV_m[j]
				newEnz[j]=resomPar.micPE_max[j]*h0/(resomPar.micPE_max[j]+h0)

				while abs(residual0)>h0*1.e-3 or residual0 < 0.0:
					residual=(resomPar.micX_h[j]-newCell[j])*ee-resomPar.micV_m[j]- \
						newCell[j]/resomPar.micYXV[j]-newEnz[j]/resomPar.micYXE[j]
					newCell1=newCell[j]-residual/(residual0-residual)*(oldCell-newCell[j])
					residual0=residual
					oldCell=newCell[j]
					newCell[j]=newCell1
					h0=(resomPar.micX_h[j]-newCell[j])*ee-resomPar.micV_m[j]
					newEnz[j]=resomPar.micPE_max[j]*h0/(resomPar.micPE_max[j]+h0)

			rCO2_e[j]=newEnz[j]*(1.0/resomPar.micYXE[j]-1.0)
			rCO2_g[j]=newCell[j]*(1.0/resomPar.micYXV[j]-1.0)

	return newCell,rCO2_m,rCO2_g,rCO2_e,newEnz,residual

def depolymerization(ystates, varid, resomPar):
	#enzyme hydrolysis, Fp
	#collect substrates
	substrates = np.concatenate(ystates[varid.beg_polymer:varid.end_polymer],
		ystates[varid.beg_mineralA:varid.end_mineralA])
	#collect consumers
	consumers=ysates[varid.beg_enzyme:varid.end_enzyme].copy()
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
	S1=np.concatenate(ystates[varid.beg_polymer:varid.end_polymer],
		ystates[varid.beg_mineralA:varid.end_mineralA])
	S2=ystates[varid.oxygen]
	sc_ij=remath.supeca(E, S1, S2, K1, K2, nE, nS1, nS2)
	return de_monomer
def resom_dyncore(ystates, nreactions, dtime, varid, resomPar):
	"""
	define the resom microbial dynamics
	rrate
	"""
	rrates=np.zeros(nreactions)
	de_polymer=depolymerization(ystates, varid, resomPar)
	#monomer uptake
	de_monomer=uptake_monomer(ystates, varid, resomPar)
	#cell physiology
	newcell,rCO2_m,rCO2_g,rCO2_e,newEnz,residual=cell_physioloy(ystates,resomPar,varid)
	#cell mortality, assuming all cells are lysed right away.

	#oxygen diffusion

	return rrates
