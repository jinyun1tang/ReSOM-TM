import numpy as np

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
		self.enz_n = np.ones(varid.nmicrobes)
		self.N_CH  = np.ones(varid.nmicrobes)
		self.Delta_H_s=np.ones(varid.nmicrobes)
		self.Kaff_Enz=np.ones((nE,nS))+0.1                   #enzyme-substrate affinity
		self.vmax_depoly_0=np.ones((nE,varid.npolymers))*1.e-6
		self.vmax_depoly=np.ones((nE,varid.npolymers))*1.e-6 #depolymerization rate
		self.Delta_E =np.ones((nE,varid.npolymers))
		self.Delta_E_Eminerals=np.ones((nE,varid.nmineralAs))
		self.Delta_G_X=np.ones(varid.nmicrobes)
		self.micYVM = np.zeros(varid.nmicrobes)+0.3          #maintenance yield from structural biomass
		self.micYXE = np.zeros(varid.nmicrobes)+0.5          #enzyme production yield from reserve
		self.micYXV = np.zeros(varid.nmicrobes)+0.5          #structural biomass yield from reserve
		self.micPE_alpha=np.zeros(varid.nmicrobes)+0.1       #maximum enzyme production rate
		self.micX_hr= np.zeros(varid.nmicrobes)+0.1
		self.micX_h = np.zeros(varid.nmicrobes)+0.1       #reserve mobilization rate
		self.micX_h0= np.zeros(varid.nmicrobes)+0.1
		self.micV_mr= np.zeros(varid.nmicrobes)+1.e-7
		self.micV_m = self.micV_mr        #somatic maintenance
		self.EnziDek= np.zeros(varid.nmicrobes)+1.e-6
		self.miciMort=np.zeros(varid.nmicrobes)+1.e-7
		self.micPerstV=np.zeros(varid.nmicrobes)+0.01
		self.K_mic_monomer=np.zeros((varid.nmicrobes,varid.nmonomers))+1.e-2
		self.K_minerals_monomer=np.zeros((varid.nmineralAs,varid.nmonomers))+1.0
		self.nosc_monomer=np.zeros(varid.nmonomers)    #nominal oxidation status
		self.YX_monomer=np.zeros(varid.nmonomers)+0.5
		self.vmax_umonomer_0=np.ones((varid.nmicrobes,varid.nmonomers))*1.e-6
		self.vmax_umonomer=self.vmax_umonomer_0
		self.K_mic_oxygen=np.ones(varid.nmicrobes)*1.e-3
		self.YX2necm=np.zeros(varid.nmicrobes)+0.05    #fraction of lysed cell go to monomers
		self.conds_o2=1.e-3
		self.o2_atm=8.
		self.co2_atm=4.e-3

class envPar():
	def __init__(self):
		self.sat=0.
		self.chb=0.
		self.psisat=0.
		self.chb=0.


def update_kinetics_par(varid, resompar, tsoil, envpar, vmsoi, veffpore):
	"""
	update the microbial kinetic parameters
	vmsoi: soi moisture
	veffpor: effective soil porosity
	envpar:
	"""
	import ReSOM.resom_mathlib as remath
	from ReSOM.constants import dzsoi, Ras, Rgas, poc_radius
	#update diffusion parameters
	phig=np.maximum(veffpore-vmsoi,0.)
	s_sat=np.minimum(vmsoi/np.maximum(veffpore,envpar.sat),1.0)
	taug, tauw=remath._moldrup_tau(veffpore, envpar.chb, s_sat)
	resompar.conds_o2=remath._conds_o2(dzsoi, taug, tauw, Ras, tsoil, vmsoi, phig)
	flm=remath.wfilm_thick(envpar.psisat, envpar.chb, s_sat, tsoi)
	Rgastsoi=Rgas*tsoil
	for j in range(varid.nmicrobes):
		#there is a strong assumption here that the thermal characteristics of a microbe and its enzyme is identical
		#fraction of active enzymes
		fact=remath._fact(tsoi, resompar.enz_n[j], resompar.N_CH[j], resompar.Delta_H_s[j], Rgas)
		#for an enzyme producing microbe
		if resompar.micPE_alpha[j] > 0.:
			#updaet enzyme kinetic parameters, affinty and maximum processing rate
			for k in range(varid.npolymers):
				resompar.vmax_depoly[j,k]=resompar.vmax_depoly_0[j,k]*np.exp(-resompar.Delta_E[j,k]/(Rgastsoi))
				resompar.Kaff_Enz[j,k],kx1w=remath._calKenz(resompar.vmax_depoly[j,k],resompar.Dw[j], poc_radius,fact)
				Dw0=1.e-9
				Db=Dw0*tauw
				resompar.Kaff_Enz[j,k]=resompar.Kaff_Enz[j,k]*remath._calvsmGamma(Db, Dw0, rm, flm,  kx1w, Ncell)
			#define enzyme affinity to soil minerals.
			for k in range(varid.nmineralAs):
				resompar.Kaff_Enz[j,k+varid.npolymers]=resompar.KaffE_minerals[j,k]*np.exp(-resompar.Delta_E_Eminerals[j,k]/Rgastsoi)

		#obtain substrate affinity

		for k in range(varid.nmonomers):
			resompar.vmax_umonomer[j,k]=resompar.vmax_umonomer_0[j,k]*np.exp(-resompar.monomer[k]/(Rgastsoi))
			Dw=1.e-9
			resompar.K_mic_monomer[j,k],kx1w=remath._calcKmic(resompar.vmax_umonomer[j,k], Dw, resompar.cell_radius[j], resompar.f0[j,k])
			resompar.K_mic_monomer[j,k]=resompar.K_mic_monomer[j,k]*remath._calvsmGamma(Db, Dw0, rm, flm,  kx1w, Ncell)
		DO2w=1.e-9
		resompar.K_mic_oxygen[j]=remath._calcKmic(resompar.vmax_umonomer[j,k], DO2w, resompar.cell_radius[j], resompar.f0[j,k])
		resompar.K_mic_oxygen[j]=resompar.K_mic_oxygen[j]*remath._calvsmGamma(Db, Dw0, rm, flm,  kx1w, Ncell)
		#update the specific reserve turnover rate
		resompar.micX_h0[j]=resompar.micX_hr[j]*fact*exp(-resompar.Delta_G_X[j]/Rgastsoi)
		resompar.micV_m[j] =resompar.micV_mr[j]*exp(-resompar.Delta_G_V[j]/Rgastsoi)
