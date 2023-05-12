
import numpy as np
import scipy
from coronavirusEqs import *
from fitting_params import *

def asymp_contact_multipliers_objfn_new(sd_params,params,known_mr_comp,known_deaths_o,y0,pops):
    d=0
    penalty = 20000
    frac_sym_new = np.ones(10)
    frac_sym_new[0:5]=sd_params[0:5]
    frac_sym_new[5:]=sd_params[0:5]
    for i in range(len(frac_sym_new)-1):
        if frac_sym_new[i+1]<frac_sym_new[i]:
            d=d+penalty
    
    #new parameters
    params_test = params.copy()
    params_test[2]=frac_sym_new
    beta_new = findBetaModel_coronavirusEqs(C, frac_sym_new, gammaA, gammaE,  gammaI, gammaP, hosp_rate, redA,  redP, red_sus, sigma, R0, totalPop)
    params_test[0]=beta_new
    contacts_total_new =sd_params[5]*C
    params_test[1] = contacts_total_new
    params_test[11]=np.ones(10) - frac_sym_new

    #run model
    t = np.linspace(0,365,300)
    sol= scipy.integrate.solve_ivp(lambda t,y:coronavirusEqs_with_vaccination_only(t,y, params_test), [0,365],y0,t_eval=t)
    sol_plot=np.reshape(sol.y,(21,10,300))
    deaths =sol_plot[18,:,-1]
    mr = deaths/pops
    #compare to known deaths
    dw = mr[1:5]
    db = mr[6:10]
    dr = np.array([dw,db])/dw
    dr = dr.reshape(8)
    #try different weights to get different fits, **w
    d = d+sum(abs(deaths-known_deaths_o))+abs(sum(deaths)-sum(known_deaths_o))**2 + sum(abs(dr - known_mr_comp))**2
    
    return d

def asymp_contact_multipliers_objfn_social_distancing_later(sd_params,params,known_mr_comp,known_deaths_o,y0,pops):
    d=0
    penalty = 20000
    frac_sym_new = np.ones(10)
    frac_sym_new[0:5]=sd_params[0:5]
    frac_sym_new[5:]=sd_params[0:5]
    for i in range(len(frac_sym_new)-1):
        if frac_sym_new[i+1]<frac_sym_new[i]:
            d=d+penalty

    
    params_test = params.copy()
    params_test[2]=frac_sym_new
    params_test[11]=np.ones(10) - frac_sym_new
    beta_new = findBetaModel_coronavirusEqs(C, frac_sym_new, gammaA, gammaE,  gammaI, gammaP, hosp_rate, redA,  redP, red_sus, sigma, R0, totalPop)
    params_test[0]=beta_new
    
    
    #One month without social distancing
    t = np.linspace(0,31,300)
    sol= scipy.integrate.solve_ivp(lambda t,y:coronavirusEqs_with_vaccination_only(t,y, params_test), [0,31],y0,t_eval=t)
    sol_plot=np.reshape(sol.y,(21,10,300))
    #new initial conditions
    y0_2020=sol_plot[:,:,-1]
    y0_2020=y0_2020.reshape(210)

    #social distancing in contacts
    contacts_total_new =sd_params[5]*C
    params_test[1] = contacts_total_new

    #Rest of year with social distancing
    t = np.linspace(0,365-31,300)
    sol= scipy.integrate.solve_ivp(lambda t,y:coronavirusEqs_with_vaccination_only(t,y, params_test), [0,365-31],y0_2020)
    sol_plot=np.reshape(sol.y,(21,10,len(sol.t)))
    
    #number of deaths from 2020
    deaths =sol_plot[18,:,-1]

    mr = deaths/pops

    #compare to known deaths
    dw = mr[1:5]
    db = mr[6:10]
    dr = np.array([dw,db])/dw
    dr = dr.reshape(8)
    #try different weights to get different fits, **w
    d = d+sum(abs(deaths-known_deaths_o))+abs(sum(deaths)-sum(known_deaths_o))**2 + sum(abs(dr - known_mr_comp))**2
    
    return d


def asymp_contact_multipliers_and_4_mat(sd_params,params,known_mr_comp, known_deaths_o,y0,pops):
    d=0
    penalty = 2000
    frac_sym_new = np.ones(10)
    frac_sym_new[0:5]=sd_params[0:5]
    frac_sym_new[5:]=sd_params[0:5]
    for i in range(len(frac_sym_new)-1):
        if frac_sym_new[i+1]<frac_sym_new[i]:
            d=d+penalty
            
    oneMinusSympRate_new = np.ones(10) - frac_sym

    #create new contact matrix
    contacts_total_new = sd_params[5]*work_total+sd_params[6]*other_total+sd_params[7]*home_total +sd_params[8]*school_total 
    
    params_test = params.copy()
    params_test[2]=frac_sym_new
    beta_new = findBetaModel_coronavirusEqs(C, frac_sym_new, gammaA, gammaE,  gammaI, gammaP, hosp_rate, redA,  redP, red_sus, sigma, R0, totalPop)
    params_test[0]=beta_new
    params_test[1] = contacts_total_new
    params_test[11]=np.ones(10) - frac_sym_new

    #run model
    t = np.linspace(0,365,300)
    sol= scipy.integrate.solve_ivp(lambda t,y:coronavirusEqs_with_vaccination_only(t,y, params_test), [0,365],y0,t_eval=t)
    sol_plot=np.reshape(sol.y,(21,10,300))
    deaths =sol_plot[18,:,-1]
    mr = deaths/pops
    #compare to known deaths
    dw = mr[1:5]
    db = mr[6:10]
    dr = np.array([dw,db])/dw
    dr = dr.reshape(8)
    #try different weights to get different fits, **w
    d = d+sum(abs(deaths-known_deaths_o))**2+abs(sum(deaths)-sum(known_deaths_o)) + sum(abs(dr - known_mr_comp))**4
    
    return d

def single_contacts_multiplier_objfn(sd_params,params,known_mr_comp,known_deaths_o,y0,pops):
    #create new contact matrix
    params_test = params.copy()
    contacts_total_new = sd_params[0]*C
    params_test[1] = contacts_total_new
    #run model
    t = np.linspace(0,365,300)
    sol= scipy.integrate.solve_ivp(lambda t,y:coronavirusEqs_with_vaccination_only(t,y, params_test),[0,365],y0,t_eval=t)
    sol_plot=np.reshape(sol.y,(21,10,300))
    deaths =sol_plot[18,:,-1]
    mr = deaths/pops
    #compare to known deaths
    dw = mr[1:5]
    db = mr[6:10]
    dr = np.array([dw,db])/dw
    dr = dr.reshape(8)
    #try different weights to get different fits, **w
    d = sum(abs(dr - known_mr_comp))**4 + sum(abs(deaths-known_deaths_o))+abs(sum(deaths)-sum(known_deaths_o))
    return d

def contact_multipliers_4mat_2race(sd_params,params,known_mr_comp,known_deaths_o,y0,pops):
    params_test = params.copy()
    #create new contact matrix
    work_total_new = np.vstack((sd_params[0]*w_work,sd_params[1]*other_work))
    other_total_new = np.vstack((sd_params[2]*w_other,sd_params[3]*other_other))
    home_total_new = np.vstack((sd_params[4]*w_home,sd_params[5]*other_home))
    school_total_new = np.vstack((sd_params[6]*w_school,sd_params[7]*other_school))
    
    contacts_total_new = work_total_new+other_total_new+home_total_new+school_total_new
    params_test[1] = contacts_total_new

    #run model
    t = np.linspace(0,365,300)
    sol= scipy.integrate.solve_ivp(lambda t,y:coronavirusEqs_with_vaccination_only(t,y, params_test),[0,365],y0,t_eval=t)
    sol_plot=np.reshape(sol.y,(21,10,300))
    deaths =sol_plot[18,:,-1]
    
    mr = deaths/pops
    #compare to known deaths
    dw = mr[1:5]
    db = mr[6:10]
    dr = np.array([dw,db])/dw
    dr = dr.reshape(8)
    #try different weights to get different fits, **w
    d=sum(abs(dr - known_mr_comp))**3 +sum(abs(known_deaths_o[1:3] - deaths[1:3]+known_deaths_o[6:8] - deaths[6:8]))+abs(sum(deaths)-1704)
    
    return d

def contact_multipliers_all_school_same(sd_params,params,known_mr_comp, known_deaths_o,y0,pops):
    params_test = params.copy()
    #create new contact matrix
    work_total_new = np.vstack((sd_params[0]*w_work,sd_params[1]*other_work))
    other_total_new = np.vstack((sd_params[2]*w_other,sd_params[3]*other_other))
    home_total_new = np.vstack((sd_params[4]*w_home,sd_params[5]*other_home))
    school_total_new = np.vstack((sd_params[6]*w_school,sd_params[6]*other_school))
    
    contacts_total_new = work_total_new+other_total_new+home_total_new+school_total_new
    params_test[1] = contacts_total_new

    #run model
    t = np.linspace(0,365,300)
    sol= scipy.integrate.solve_ivp(lambda t,y:coronavirusEqs_with_vaccination_only(t,y, params_test),[0,365],y0,t_eval=t)
    sol_plot=np.reshape(sol.y,(21,10,300))
    deaths =sol_plot[18,:,-1]
    
    mr = deaths/pops
    #compare to known deaths
    dw = mr[1:5]
    db = mr[6:10]
    dr = np.array([dw,db])/dw
    dr = dr.reshape(8)
    #try different weights to get different fits, **w
    d=sum(abs(dr - known_mr_comp))**3 + abs(sum(deaths)-1704)+sum(abs(known_deaths_o[1:3] - deaths[1:3]+known_deaths_o[6:8] - deaths[6:8]))
    return d

def contact_multipliers_4mat(sd_params,params,known_mr_comp, known_deaths_o,y0,pops):
    params_test = params.copy()
    #create new contact matrix
    contacts_total_new = sd_params[0]*work_total+sd_params[1]*other_total+sd_params[2]*home_total +sd_params[3]*school_total 
    params_test[1] = contacts_total_new

    #run model
    t = np.linspace(0,365,300)
    sol= scipy.integrate.solve_ivp(lambda t,y:coronavirusEqs_with_vaccination_only(t,y, params_test),[0,365],y0,t_eval=t)
    sol_plot=np.reshape(sol.y,(21,10,300))
    deaths =sol_plot[18,:,-1]
    
    mr = deaths/pops
    #compare to known deaths
    dw = mr[1:5]
    db = mr[6:10]
    dr = np.array([dw,db])/dw
    dr = dr.reshape(8)
    #try different weights to get different fits, **w
    d=sum(abs(known_deaths_o[[3,8]] - deaths[[3,8]]))+sum(abs(dr - known_mr_comp))**4+abs(sum(known_deaths_o) - sum(deaths))
    return d

def contact_multipliers_all_school_and_work_same(sd_params,params,known_mr_comp, known_deaths_o,y0,pops):
    params_test = params.copy()
    #create new contact matrix
    contacts_total_new = sd_params[0]*work_total+np.vstack((sd_params[1]*w_other,sd_params[2]*other_other))+np.vstack((sd_params[3]*w_home,sd_params[4]*other_home)) +sd_params[5]*school_total 
    params_test[1] = contacts_total_new

    #run model
    t = np.linspace(0,365,300)
    sol= scipy.integrate.solve_ivp(lambda t,y:coronavirusEqs_with_vaccination_only(t,y, params_test),[0,365],y0,t_eval=t)
    sol_plot=np.reshape(sol.y,(21,10,300))
    deaths =sol_plot[18,:,-1]
    
    mr = deaths/pops
    #compare to known deaths
    dw = mr[1:5]
    db = mr[6:10]
    dr = np.array([dw,db])/dw
    dr = dr.reshape(8)
    #try different weights to get different fits, **w
    d = sum(abs(dr - known_mr_comp))**4 + abs(sum(deaths)-1704)+sum(abs(known_deaths_o[1:4] - deaths[1:4]+known_deaths_o[6:9] - deaths[6:9]))
    return d

def contact_multipliers_objfn_fit_age(sd_params,params,known_mr_comp, known_deaths_o,y0,pops):
    params_test = params.copy()
    #create new contact matrix
    mult = np.concatenate((sd_params,sd_params))
    contacts_total_new = mult*C
    params_test[1] = contacts_total_new

    #run model
    t = np.linspace(0,365,300)
    sol= scipy.integrate.solve_ivp(lambda t,y:coronavirusEqs_with_vaccination_only(t,y, params_test),[0,365],y0,t_eval=t)
    sol_plot=np.reshape(sol.y,(21,10,300))
    deaths =sol_plot[18,:,-1]
    
    mr = deaths/pops
    #compare to known deaths
    dw = mr[1:5]
    db = mr[6:10]
    dr = np.array([dw,db])/dw
    dr = dr.reshape(8)
    #try different weights to get different fits, **w
    d = sum(abs(dr - known_mr_comp))**2 + abs(sum(deaths)-1704)+sum(abs(known_deaths_o - deaths))**3
    return d

def contact_multipliers_objfn_fit_age_race(sd_params,params,known_mr_comp, known_deaths_o,y0,pops):
    params_test = params.copy()
    #create new contact matrix
    mult1 = np.concatenate((sd_params[0:5],sd_params[0:5]))
    mult2 = np.concatenate((sd_params[5:],sd_params[5:]))
    contacts_total_new = np.vstack((mult1*C[0:5,:],mult2*C[5:,:]))
    params_test[1] = contacts_total_new

    #run model
    t = np.linspace(0,365,300)
    sol= scipy.integrate.solve_ivp(lambda t,y:coronavirusEqs_with_vaccination_only(t,y, params_test),[0,365],y0,t_eval=t)
    sol_plot=np.reshape(sol.y,(21,10,300))
    deaths =sol_plot[18,:,-1]
    
    mr = deaths/pops
    #compare to known deaths
    dw = mr[1:5]
    db = mr[6:10]
    dr = np.array([dw,db])/dw
    dr = dr.reshape(8)
    #try different weights to get different fits, **w
    d = sum(abs(dr - known_mr_comp))**3 + sum(abs(known_deaths_o[1:3] - deaths[1:3]))**2+ sum(abs(known_deaths_o[6:9] - deaths[6:9]))**2+abs(sum(deaths)-1704)
    return d
