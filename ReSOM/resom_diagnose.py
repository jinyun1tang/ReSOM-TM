import numpy as np


def total_cmass_sum(nstep,ystates,varid):
    """
    sum up the total carbon mass
    """
    total_polymer=np.sum(ystates[varid.beg_polymer:varid.end_polymer+1])
    total_monomer=np.sum(ystates[varid.beg_monomer:varid.end_monomer+1])
    total_micbmassx=np.sum(ystates[varid.beg_microbeX:varid.end_microbeX+1])
    total_micbmassv=np.sum(ystates[varid.beg_microbeV:varid.end_microbeV+1])
    total_enzymes=np.sum(ystates[varid.beg_enzyme:varid.end_enzyme+1])
    total_mass=total_polymer+total_monomer+total_micbmassx+total_micbmassv+total_enzymes+ystates[varid.co2]
    print("nstep=%d,polymer=%18.12e,monomer=%18.12e,micbiomassx=%18.12e,micbiomassv=%18.12e,enzyme=%18.12e,co2=%18.12e,total=%18.12e"%\
        (nstep,total_polymer,total_monomer, total_micbmassx,total_micbmassv, total_enzymes, ystates[varid.co2], total_mass))
    return total_mass

def get_daily_ts(hourly_ts):
    """
    convert hourly data into daily average
    """
    daz=np.reshape(hourly_ts,(24,int(len(hourly_ts)/24)))
    day_ts=np.mean(daz,axis=0)
    return day_ts
