

__all__=['dzsoi','Ras','Rgas']


dzsoi=0.1         # 10 cm
Ras=50.           # s m-1
Rgas=8.3144598    # universal gas constant,J⋅mol−1⋅K−1
poc_radius=1.e-3  # 1mm, poc radius
Ncell=20.
rc=1.e-6          # m
alphaV=80.        # volume scaling factor, unitless
rm=rc*(alphaV*Ncell)**(1/3.)
