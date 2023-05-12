
import numpy as np
from two_pop_params_current import *

def calc_death_bars(deaths,total_pops):
    mort = deaths/total_pops
    b_m = mort[5:10]
    w_m = mort[0:5]

    w_p = total_pops[0:5]
    b_p = total_pops[5:10]

    mort_comp=abs(b_m/w_m - 1)
    c = 0

    for i in mort_comp:
        if np.isnan(i):
            mort_comp[c]=0
        if np.isinf(i):
            mort_comp[c]= abs(b_m[c]/(1/w_p[c])-1)
        c =c+1   
    return sum(mort_comp)

def calc_mort_diff(deaths,total_pops):
    mort = deaths/total_pops
    b_m = mort[5:10]
    w_m = mort[0:5]

    mort_comp=abs(b_m - w_m)

    return sum(mort_comp)

def calc_mort_idis(deaths,total_pops,total_pops_age):
    mort = deaths/total_pops
    mort_total=(deaths[0:5]+deaths[5:])/total_pops_age

    mort_idis = 100*(1/2)*(abs(mort[5:]-mort_total)+abs(mort[0:5]-mort_total))/mort_total

    return sum(mort_idis)

def calc_ylls(deaths,two_pop_daly,pops):
    dalys = deaths*two_pop_daly
    dalys_per_100000 = (dalys/pops)*100000
    return dalys_per_100000

def calc_yll_diff(deaths,two_pop_daly,pops):
    dalys = deaths*two_pop_daly
    dalys_per_100000 = (dalys/pops)*100000
    return sum(abs(dalys_per_100000[0:5]-dalys_per_100000[5:10]))

def calc_index_idis(deaths,two_pop_daly,pops):
    dalys = deaths*two_pop_daly
    dalys_per_100000 = (dalys/pops)*100000

    dalys_total=(deaths[0:5]+deaths[5:])*two_pop_daly_total
    dalys_total_per_100000 = (dalys_total/total_pops_age)*100000

    idis = 100*(1/2)*(abs(dalys_per_100000[5:]-dalys_total_per_100000)+abs(dalys_per_100000[0:5]-dalys_total_per_100000))/dalys_total_per_100000
    return sum(idis)