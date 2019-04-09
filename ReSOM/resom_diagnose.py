import numpy as np


def total_cmass_sum(ystates,varid):
    """
    sum up the total carbon mass
    """
    total_polymer=np.sum(ystates[varid.beg_polymer:varid.end_polymer+1])
    total_monomer=np.sum(ystates[varid.beg_monomer:varid.end_monomer+1])
    total_micbmassx=np.sum(ystates[varid.beg_microbeX:varid.end_microbeX+1])
    total_micbmassv=np.sum(ystates[varid.beg_microbeV:varid.end_microbeV+1])
    total_enzymes=np.sum(ystates[varid.beg_enzyme:varid.end_enzyme+1])
    total_mass=total_polymer+total_monomer+total_micbmassx+total_micbmassv+total_enzymes+ystates[varid.co2]
    print("polymer=%18.10e,monomer=%18.10e,micbiomassx=%18.10e,micbiomassv=%18.10e,enzyme=%18.10e,co2=%18.10e,total=%18.10e"%\
        (total_polymer,total_monomer, total_micbmassx,total_micbmassv, total_enzymes, ystates[varid.co2], total_mass))
    return total_mass
