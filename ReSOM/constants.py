
import math
__all__=['dzsoi','Ras','Rgas', 'pom_radius','pom_B','mC_amino','cmass_to_cell', 'M_Acef']


dzsoi=0.1          # 10 cm
Ras=50.            # s m-1
Rgas=8.3144598     # universal gas constant,J⋅mol−1⋅K−1
pom_radius=75.e-6  # 75 mu m, poc radius
#Enz_radius=5.e-9   # 5 nm, hydrated enzyme radius
#k2_enz=5.          # specific enzyme hydrolysis rate
#A_enz=4.*(pom_radius/Enz_radius)  # maximum number of enzymes binding site per pom
pom_fc=0.5          #carbon content of pom, by weight
pom_dens=1.5e6      #pom density, g/m3
catomw=12.
Na=6.02e23         #Avogadro's number
pom_B=4./3.*math.pi*(pom_radius)**3*pom_dens*pom_fc/catomw*Na  #mol C per pom particle, convert from mol C into mol particle
Ncell=20.          #number of cells per microsite
rc=1.e-6           # m
alphaV=80.         # volume scaling factor, unitless
rm=rc*(alphaV*Ncell)**(1/3.)   #microsite radius
mC_amino=64./12.   #mean mol C per mole amino acids
mwt_amino=136.75   #mean molecular weight of amino acids
cmass_to_cell=2.68e-11  #mol cells (mol C)-1
dfrac=2.52         #fractal dimension
M_Acef=catomw/(pom_fc*Na*pom_dens*math.pi)  #mol of enzymes per mol C pom can bind multiplied enz_radius^3
