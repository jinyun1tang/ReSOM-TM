import numpy as np

class varid():
	"""
	structure holding the location of different variables within the vector ystates
	"""
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
	"""
	structure holding the ids of the reactions
	"""
	def __init__(self,varid):
		#depolymerization
		self.reac_beg_depolymer=0
		self.reac_end_depolymer=self.reac_beg_depolymer+varid.npolymers-1
		#monomer uptake
		self.reac_beg_mics_upmonomer=self.reac_end_depolymer+1
		self.reac_end_mics_upmonomer=self.reac_beg_mics_upmonomer+varid.nmonomers-1
		#respiratory oxygen consumption
		self.reac_mics_cresp_oxygen = self.reac_end_mics_upmonomer+1
		self.nbreactions =self.reac_mics_cresp_oxygen+1

class resomPar():
	"""
	parameters of resom
	"""
	def __init__(self,varid):
		nE=varid.nenzymes
		nS=varid.npolymers+varid.nmineralAs
		#activation energies
		self.Delta_E_depoly =np.ones((nE,varid.npolymers))*30.e3     #activation energy of depolymerization, J/mol
		self.Delta_E_Eminerals=np.ones((nE,varid.nmineralAs))*45.e3  #activation energy of enzyme-mineral binding, J/mol
		self.Delta_G_X=np.ones(varid.nmicrobes)*60.e3                #activiation energy of reserve mobilization, J/mol
		self.Delta_G_V=np.ones(varid.nmicrobes)*30.e3                #Gibbs energy for somatic maintenance, J/mol
		self.Delta_monomer=np.zeros(varid.nmonomers)+20.e3           #activiation energy of monomer oxidation
		#thermal stability parameter
		self.enz_n = np.ones(varid.nmicrobes)*270.                   #average number of amino acids in the enzyme
		self.N_CH  = np.ones(varid.nmicrobes)*6.                     #average number of non-polar hydrogen atmos per amino acid
		self.Delta_H_s=np.ones(varid.nmicrobes)*5330.                #J/mol amino acids
		self.Tref  = np.ones(varid.nmicrobes)*298.15                 #reference temperature where kinetic rates are computed

		self.Kaff_Enz=np.ones((nE+varid.nmonomers,nS))*0.            #enzyme-substrate affinity, mol/m3, estimated
		self.k2_enz=np.ones((nE,nS))*5.                              #specific enzyme hydrolysis rate
		self.Enz_radius= np.ones(nE)*5.e-9                            #5 nm, hydrated enzyme radius
		self.A_enz     = np.ones(nE)
		self.vmax_depoly_0=np.ones((nE,varid.npolymers))*3.2e3       #reference maximum depolymerization rate, s-1, estimated
		self.vmax_depoly=np.ones((nE,varid.npolymers))*3.2e-3        #instantaneous depolymerization rate, s-1
		self.DwEnz=np.zeros(nE)+1.e-10                               #enzyme diffusivity, Brune and Kim (1993), can be estimated by Einstein equation
		self.KaffE_minerals=np.ones((nE,varid.nmineralAs))*10.e0      #affinity parameter of enzyme-mineral binding
		#enzyme parameters
		self.micPE_alpha=np.zeros(varid.nmicrobes)+0.1               #maximum enzyme production rate, s-1
		self.EnziDek= np.zeros(varid.nmicrobes)+1.e-6                #enzyme decay rate, s-1
		#yield rates
		self.micYVM = np.zeros(varid.nmicrobes)+0.3                  #maintenance yield from structural biomass
		self.micYXE = np.zeros(varid.nmicrobes)+0.5                  #enzyme production yield from reserve
		self.micYXV = np.zeros(varid.nmicrobes)+0.5                  #structural biomass yield from reserve
		self.micX_hr= np.ones(varid.nmicrobes)*5.e-5                 #reference reserve mobilization rate, s-1
		self.micX_h0= np.zeros(varid.nmicrobes)+0.1                  #reserve mobilization rate, s-1
		self.micV_mr= np.zeros(varid.nmicrobes)+1.e-7                #reference specific maintenance, s-1
		self.micV_m = self.micV_mr                                   #somatic maintenance
		#mortality parameter
		self.miciMort=np.zeros(varid.nmicrobes)+0.5e-6                #intrinsic specific microbial mortality, s-1
		self.micPerstV=np.zeros(varid.nmicrobes)+0.01                #half saturation constant for microbial mortality, mol/m3
		self.cell_radius=np.zeros(varid.nmicrobes)+1.e-6             #cell radius, m

		self.K_mic_monomer=np.zeros((varid.nmicrobes,varid.nmonomers+varid.nmicrobes))+1.e-2  #momonmer affinity to microbes, mol/m3
		self.K_minerals_monomer=np.zeros((varid.nmineralAs+1,varid.nmonomers+varid.nmicrobes))+1.0  #monomer affinity to mineral surfaces+polymers (as the last row), mol/m3
		self.nosc_monomer=np.zeros(varid.nmonomers)                  #nominal oxidation status
		self.YX_monomer=np.zeros(varid.nmonomers)+0.5                #reserve yield rate from monomer oxidation
		self.k2_umonomer=np.ones((varid.nmicrobes,varid.nmonomers+1))*100.       #specific monomer processing rate per transporter , s-1
		self.k2_tp_monomer=np.ones((varid.nmicrobes,varid.nmonomers+1))*3000.    #number of monomer transporters
		self.vmax_umonomer=np.ones((varid.nmicrobes,varid.nmonomers))*1.e-6    #maximum monomer uptake rate, s-1
		self.vmax_umonomer_0=np.ones((varid.nmicrobes,varid.nmonomers))*1.e-6    #reference maximum monomer uptake rate, s-1
		self.catom_monomer=np.ones(varid.nmonomers)*10.				 #number of C atoms per monomer
		self.Dw_monomer = np.ones(varid.nmonomers+1)*1.e-9			 #monomer diffusivity
		self.K_mic_oxygen=np.ones((varid.nmicrobes))*1.e-3           #oxygen affinity parameter, mol/m3
		#transporter parameters
		self.k2_uo2=np.ones(varid.nmicrobes)*100.                    #maximum oxygen uptake rate per transporter, s-1
		self.mic_tp_o2=np.ones(varid.nmicrobes)*3000.                #number of oxygen transporters
		self.f0=np.ones((varid.nmicrobes,varid.nmonomers+1))*0.5       #monomer interception rate
		self.f0_o2=np.ones(varid.nmicrobes)*0.5                      #oxygen interception rate
		self.YX2necm=np.zeros(varid.nmicrobes)+0.05                  #fraction of lysed cell go to monomers
		self.conds_o2=1.e-3                                          #conductance for oxygen, m/s
		self.o2_atm=8.57                                             #atmospheric oxygen concentration
		self.co2_atm=4.e-3
		self.fact_soil=1.
class envPar():
	def __init__(self):
		self.sat=0.
		self.chb=0.
		self.psisat=0.
		self.chb=0.
		self.df_frac=1./2.52

def update_microbial_par(varid, resompar):
	"""
	update constant microbial parameters
	"""
	from ReSOM.constants import pom_radius, mC_amino, pom_B, cmass_to_cell, M_Acef
	for j in range(varid.nmicrobes):
		resompar.A_enz[j]=4.*(pom_radius/resompar.Enz_radius[j])**2  # maximum number of enzymes binding site per pom
		resompar.k2_uo2[j]=resompar.k2_uo2[j]*resompar.mic_tp_o2[j]
		for k in range(varid.nmonomers):
			resompar.k2_umonomer[j,k]=resompar.k2_umonomer[j,k]*resompar.k2_tp_monomer[j,k]
			resompar.vmax_umonomer_0[j,k]=resompar.k2_umonomer[j,k]*resompar.catom_monomer[k]
		k=varid.nmonomers
		resompar.k2_umonomer[j,k]=resompar.k2_umonomer[j,k]*resompar.k2_tp_monomer[j,k]
			#print('vmax_monomer=%e s-1'%(resompar.vmax_umonomer_0[j,k]*cmass_to_cell))
		#following are some very crude approximation rules
		resompar.micX_hr[j]=np.max(resompar.vmax_umonomer_0[j,:])*cmass_to_cell*0.3
		resompar.micV_mr[j]=resompar.micX_hr[j]*0.05
		for k in range(varid.npolymers):
			resompar.vmax_depoly_0[j,k]=resompar.k2_enz[j,k]*M_Acef/resompar.Enz_radius[j]**3   #s-1, assuming Enz in mol m-3, and pom in mol C m-3
#			print('vmax_e=%e mol monomer s-1'%(resompar.vmax_depoly_0[j,k]))

def update_kinetics_par(varid, resompar, tsoil, envpar, vmsoi, veffpore):
	"""
	update the microbial kinetic parameters
	vmsoi: soi moisture
	veffpor: effective soil porosity
	envpar:
	"""
	import ReSOM.resom_mathlib as remath
	from ReSOM.constants import dzsoi, Ras, Rgas, pom_radius, Ncell, rm, pom_B
	#update diffusion parameters
	phig=np.maximum(veffpore-vmsoi,0.)
	s_sat=np.minimum(vmsoi/np.maximum(veffpore,envpar.sat),1.0)
	resompar.fact_soil=s_sat**envpar.df_frac
	taug, tauw=remath._moldrup_tau(veffpore, envpar.chb, s_sat)
	resompar.conds_o2=remath._conds_o2(dzsoi, taug, tauw, Ras, tsoil, vmsoi, phig)
	flm=remath.wfilm_thick(envpar.psisat, envpar.chb, s_sat, tsoil)
	bo2=remath._bunsen_o2(tsoil)
	diffo2_b,diffo2_w=remath._cal_o2_diffbw(tsoil, phig, vmsoi, taug, tauw, bo2)

	#the matrix is of size (nE + nmonomers, npolymers+nminerals)
	for jj in range(varid.nmonomers):
		j=jj+varid.nmicrobes
		for k in range(varid.npolymers):
			Dw0= resompar.Dw_monomer[jj]
			resompar.Kaff_Enz[j,k],kx1w=remath._calKenz(50.,Dw0, pom_radius)
			resompar.Kaff_Enz[j,k]= resompar.Kaff_Enz[j,k]*resompar.A_enz[0]
			resompar.Kaff_Enz[j,k]=resompar.Kaff_Enz[j,k]/(tauw*vmsoi)


	for j in range(varid.nmicrobes):
		iRgastsoi=1./Rgas*(1./tsoil-1./resompar.Tref[j])
		#there is a strong assumption here that the thermal characteristics of a microbe and its enzyme is identical
		#fraction of active enzymes
		factT=remath._fact(tsoil, resompar.enz_n[j], resompar.N_CH[j], resompar.Delta_H_s[j], Rgas)
		fact=factT*resompar.fact_soil
		#for an enzyme producing microbe
		if resompar.micPE_alpha[j] > 0.:
			#updaet enzyme kinetic parameters, affinty and maximum processing rate
			for k in range(varid.npolymers):
				Dw0=resompar.DwEnz[j]
				resompar.Kaff_Enz[j,k],kx1w=remath._calKenz(resompar.k2_enz[j,k],Dw0, pom_radius)
				#obtain the affinity in the unit of mol enzyme per m3
				resompar.Kaff_Enz[j,k]= resompar.Kaff_Enz[j,k]*resompar.A_enz[j]
				#resompar.Kaff_Enz[j,k]= resompar.Kaff_Enz[j,k]/(mC_amino*resompar.enz_n[j])
				#apply the diffusition limitation
				resompar.Kaff_Enz[j,k]=resompar.Kaff_Enz[j,k]/(tauw*vmsoi)
				#print("Kaff_Enz=%f mol enz m-3, tau=%f"%(resompar.Kaff_Enz[j,k],tauw))
				resompar.vmax_depoly[j,k]=resompar.vmax_depoly_0[j,k]*np.exp(-resompar.Delta_E_depoly[j,k]*iRgastsoi)
			#define enzyme affinity to soil minerals.
			for k in range(varid.nmineralAs):
				resompar.Kaff_Enz[j,k+varid.npolymers]=resompar.KaffE_minerals[j,k]*np.exp(-resompar.Delta_E_Eminerals[j,k]*iRgastsoi)

		#obtain substrate affinity

		for k in range(varid.nmonomers):
			fT=np.exp(-resompar.Delta_monomer[k]*iRgastsoi)
			resompar.vmax_umonomer[j,k]=resompar.vmax_umonomer_0[j,k]*fT*fact
			k2=resompar.k2_umonomer[j,k]*fT
			Dw0=resompar.Dw_monomer[k]
			resompar.K_mic_monomer[j,k],kx1w=remath._calcKmic(k2, Dw0, resompar.cell_radius[j], resompar.f0[j,k])
			Db=resompar.Dw_monomer[k]*tauw
			resompar.K_mic_monomer[j,k]=resompar.K_mic_monomer[j,k]*remath._calvsmGamma(Db, Dw0, rm, flm,  kx1w, Ncell)
			#print('Kaff doc mic=%f mol m-3'%resompar.K_mic_monomer[j,k])
		#add the reserve binding rate by the transporter
		k =varid.nmonomers+j
		k2=resompar.k2_umonomer[j,varid.nmonomers]*fT
		Dw0=resompar.Dw_monomer[varid.nmonomers]
		resompar.K_mic_monomer[j,k],kx1w=remath._calcKmic(k2, Dw0, resompar.cell_radius[j], resompar.f0[j,varid.nmonomers])
		DO2w=diffo2_w
		resompar.K_mic_oxygen[j], kx1w=remath._calcKmic(resompar.k2_uo2[j], DO2w, resompar.cell_radius[j], resompar.f0_o2[j])
		#print('Km_oxygen0=%f mol aqueous O2 m-3'%resompar.K_mic_oxygen[j])
		Db=diffo2_b
		resompar.K_mic_oxygen[j]=resompar.K_mic_oxygen[j]*remath._calvsmGamma(Db, Dw0, rm, flm,  kx1w, Ncell)
		#convert into gaseous affinity parameter
		resompar.K_mic_oxygen[j]=resompar.K_mic_oxygen[j]/bo2
		#print('Km_oxygen1=%f mol aqueous O2 m-3'%resompar.K_mic_oxygen[j])
		#update the specific reserve turnover rate
		resompar.micX_h0[j]=resompar.micX_hr[j]*fact*np.exp(-resompar.Delta_G_X[j]*iRgastsoi)
		#update demand of somatic maintenance
		resompar.micV_m[j] =resompar.micV_mr[j]*np.exp(-resompar.Delta_G_V[j]*iRgastsoi)
