import numpy as np
import pandas as pd

def findBetaModel_coronavirusEqs(C, frac_sym, gammaA, gammaE,  gammaI, gammaP, hosp_rate, redA,  redP, red_sus, sigma, R0, totalPop):
    #compute the value of beta for model
    #here, frac_sym is a vector not a scalar and there is a different beta for each group representing a different
    #susceptibility
    # compute the eignevalues of F*V^(-1) assuming the infected states are 5, namely: 
    #E, A, P, I, H
    [n1, n2] = np.shape(C)
    # create F
    N = np.sum(totalPop)
    Z = np.zeros((n1, n1))
    C1 = np.zeros((n1, n1))
    for ivals in range(n1):
        for jvals in range(n1):
            C1[ivals, jvals] = red_sus[ivals] * C[ivals, jvals] * totalPop[ivals]/totalPop[jvals]

    
    #create F by concatenating different matrices:
    F1 = np.concatenate((Z, redA*C1, redP*C1, C1), 1)
    
    F2 = np.zeros((3*n1, 4*n1))
    

    F = np.concatenate((F1, F2), 0)
    #print(np.shape(F))
    #print(np.shape(F1))
    #print(np.shape(F2))
    #create V
    VgammaE = np.diag(gammaE * np.ones(n1))
    VgammaA = np.diag(gammaA * np.ones(n1))
    VgammaP = np.diag(gammaP * np.ones(n1))



    Vsub1 = np.diag(-(np.ones(n1)-frac_sym) * gammaE)
    Vsub2 = np.diag(-(frac_sym) * gammaE)

    Vsub3 = np.diag((np.ones(n1) - hosp_rate) * gammaI + np.multiply(sigma, hosp_rate))


    V1 = np.concatenate((VgammaE, Z, Z, Z), 1)
    V2 = np.concatenate((Vsub1, VgammaA, Z, Z), 1)
    V3 = np.concatenate((Vsub2, Z, VgammaP, Z), 1)
    V4 = np.concatenate((Z, Z, -VgammaP, Vsub3), 1)


    V = np.concatenate((V1, V2, V3, V4), 0)
    #print(np.shape(V))

    myProd = np.dot(F, np.linalg.inv(V))
    # print(myProd)
    myEig = np.linalg.eig(myProd)
    # print(myEig)
    largestEig = np.max(myEig[0])
    if largestEig.imag == 0.0:

        beta = R0 / largestEig.real
        #print('beta', beta)
        return beta
    else:
        #print(largestEig)
        raise Exception('largest eigenvalue is not real')

p_total = 4060795
N = 4060795

w_pop = np.array([ 612303., 1210263.,  416905.,  464663.,  461350.])
b_pop = np.array([22813., 38647.,  9274.,  7169.,  4707.])
n_pop = np.array([11649., 20019.,  6124.,  5373.,  3992.])
a_pop = np.array([ 40687., 101711.,  23844.,  17782.,  14673.])
h_pop = np.array([214440., 261993.,  47421.,  26634.,  16359.])

other_pop=b_pop+n_pop+a_pop+h_pop

w_fracs = w_pop/sum(w_pop)
b_fracs = b_pop/sum(b_pop)
n_fracs = n_pop/sum(n_pop)
a_fracs = a_pop/sum(a_pop)
h_fracs = h_pop/sum(h_pop)

other_fracs = other_pop/sum(other_pop)

w_prop = sum(w_pop)/p_total
b_prop = sum(b_pop)/p_total
n_prop = sum(n_pop)/p_total
a_prop = sum(a_pop)/p_total
h_prop = sum(h_pop)/p_total

other_prop = sum(other_pop)/p_total

pop_fracs = np.array([w_pop,other_pop]).reshape(10)
pop_fracs =pop_fracs/p_total

age_fracs_16 = np.array([0.04587244, 0.04915714, 0.05018789, 0.05031751, 0.05490297,
       0.06667233, 0.06832345, 0.06911869, 0.06324047, 0.06224758,
       0.06163061, 0.06957834, 0.07359301, 0.07167126, 0.05788379,
       0.08560251])

total_pops=np.array([w_pop,other_pop]).reshape(10)

p_total = sum(total_pops)

other_eth_fracs = np.array([sum(b_pop)/sum(other_pop),sum(n_pop)/sum(other_pop),sum(a_pop)/sum(other_pop),sum(h_pop)/sum(other_pop)])

prop_o=sum(other_pop)/p_total
prop_w=1-sum(other_pop)/p_total

def convert_16_to_5(contacts16, agefracs16):

    contacts5=np.zeros((5,5))
    temp=contacts16*age_fracs_16[:, None]

    w = np.ones((16,16))*age_fracs_16[0:16, None]
    ind = [0,4,10,12,14,15]
    for i in range(5):
        i1 = ind[i]
        i2 = ind[i+1]
        contacts5[i,i]=np.average(contacts16[i1:i2,i1:i2], weights = w[i1:i2,i1:i2])
        for j in range(i+1,5):
            j1 = ind[j]
            j2 = ind[j+1]
            contacts5[i,j]=np.average(contacts16[i1:i2,j1:j2], weights = w[i1:i2,j1:j2]) 
            contacts5[j,i]=np.average(contacts16[j1:j2,i1:i2], weights = w[j1:j2,i1:i2])
           
    return contacts5

#Contact Matrix
import pandas as pd

df_home = pd.read_csv('contact_matrices_177_countries/contacts_home_USA.csv')
contacts16_home = df_home.to_numpy().reshape(16,16)

df_other = pd.read_csv('contact_matrices_177_countries/contacts_other_USA.csv')
contacts16_other = df_other.to_numpy().reshape(16,16)

df_school = pd.read_csv('contact_matrices_177_countries/contacts_school_USA.csv')
contacts16_school = df_school.to_numpy().reshape(16,16)

df_work = pd.read_csv('contact_matrices_177_countries/contacts_work_USA.csv')
contacts16_work = df_work.to_numpy().reshape(16,16)

#adapted from Prem
contacts_home=convert_16_to_5(contacts16_home, age_fracs_16)
contacts_other=convert_16_to_5(contacts16_other, age_fracs_16)
contacts_work=convert_16_to_5(contacts16_work, age_fracs_16)
contacts_school=convert_16_to_5(contacts16_school, age_fracs_16)
contacts_full = contacts_home+contacts_work+contacts_other+contacts_school

w_work = (.09+0.5*.11+ 0.25*.79)*contacts_work
b_work = (.21 +0.5* .15 + 0.25*.64)*contacts_work
n_work = (.1675+0.5*.1675 + 0.25*.665)*contacts_work # .335 essential, unknown split using 1/2
a_work = (.13+0.5*.12 + 0.25*.75)*contacts_work
h_work = (.24 +0.5*0.09 + 0.25*.65)*contacts_work

other_work = other_eth_fracs[0]*b_work+other_eth_fracs[1]*n_work+other_eth_fracs[2]*a_work+other_eth_fracs[3]*h_work


w_work=np.hstack((prop_w*w_work,prop_o*w_work))
other_work=np.hstack((prop_w*other_work,prop_o*other_work))

work_total = np.vstack((w_work,other_work))

w_school=np.hstack((prop_w*contacts_school,prop_o*contacts_school))
other_school=np.hstack((prop_w*contacts_school,prop_o*contacts_school))
school_total = np.vstack((w_school,other_school))

w_other=np.hstack((prop_w*contacts_other,prop_o*contacts_other))
other_other=np.hstack((prop_w*contacts_other,prop_o*contacts_other))
other_total = np.vstack((w_other,other_other))

w_home = contacts_home
b_home = ((1.6/1.95 + 1.1/1.3 + 1.7/2.15 + 0.51/0.6)/4)*contacts_home#0.49*contacts_home
n_home = ((2.1/1.95 + 1.4/1.3 + 2.15/2.15 + 0.9/0.6)/4)*contacts_home#1.18*contacts_home
a_home = ((2.1/1.95 + 1.4/1.3 + 2.15/2.15 + 0.9/0.6)/4)*contacts_home#1.18*contacts_home
h_home = ((2.3/1.95 + 1.75/1.3 + 2.6/2.15 + 0.9/0.6)/4)*contacts_home#1.27*contacts_home

other_home = other_eth_fracs[0]*b_home+other_eth_fracs[1]*n_home+other_eth_fracs[2]*a_home+other_eth_fracs[3]*h_home

same = 0.898
interacial = .102

w_home = np.hstack((same*w_home,interacial*w_home))
other_home = np.hstack((interacial*other_home,same*other_home))
home_total = np.vstack((w_home,other_home))

contacts_total = work_total+other_total+home_total+school_total
C = contacts_total

#0-19,20-49,50-59,60-69,70+
#w:0,1,2,3,4
#o:5,6,7,8,9

frac_sym=(1-.321)*np.ones(10)
frac_sym[[0,5]]=(1-0.467)
frac_sym[[3,4,8,9]]=(1-0.197)

gammaA = 1/6
gammaE = 1/3
gammaI = 1/4
gammaP = 1/2

 #hospitalization duration 
gammaH = np.ones(10)
gammaH[[0,1,5,6]] = 1 / 3 #0-49
gammaH[[2,7]] = 1 / 4 #50-59

#60-69
gammaH[3]=(1/4)*w_fracs[3]/(w_fracs[3]+w_fracs[4]) + (1/6)*w_fracs[4]/(w_fracs[3]+w_fracs[4])
gammaH_b=(1/4)*b_fracs[3]/(b_fracs[3]+b_fracs[4]) + (1/6)*b_fracs[4]/(b_fracs[3]+b_fracs[4])
gammaH_n=(1/4)*n_fracs[3]/(n_fracs[3]+n_fracs[4]) + (1/6)*n_fracs[4]/(n_fracs[3]+n_fracs[4])
gammaH_a=(1/4)*a_fracs[3]/(a_fracs[3]+a_fracs[4]) + (1/6)*a_fracs[4]/(a_fracs[3]+a_fracs[4])
gammaH_h=(1/4)*h_fracs[3]/(h_fracs[3]+h_fracs[4]) + (1/6)*h_fracs[4]/(h_fracs[3]+h_fracs[4])
gammaH[8]=other_eth_fracs[0]*gammaH_b+other_eth_fracs[1]*gammaH_n+other_eth_fracs[2]*gammaH_a+other_eth_fracs[3]*gammaH_h
#70p
gammaH[[4,9]] = 1 / 6


#hospitalization rate
hr = np.array([0.1/100*sum(age_fracs_16[0:2])/sum(age_fracs_16[0:4])+0.3/100*sum(age_fracs_16[2:4])/sum(age_fracs_16[0:4]),1.2/100*sum(age_fracs_16[4:6])/sum(age_fracs_16[4:10])+3.2/100*sum(age_fracs_16[6:8])/sum(age_fracs_16[4:10])+4.9/100*sum(age_fracs_16[8:10])/sum(age_fracs_16[4:10]),
  10.2/100,16.6/100,24.3/100*0.67 + 27.3/100*.33])
w_hr = hr
#from hr/pop
#b_hr = np.array([2.8, 3.4, 4. , 4.6, 2.6])*hr
#n_hr = np.array([9.4, 6.3, 6.2, 3.5, 2. ])*hr
#a_hr = np.array([1.1, 1.2, 1.1, 1.8, 1.5])*hr
#h_hr = np.array([4. , 4.8, 6.1, 5.8, 3.3])*hr
#other_hr = other_eth_fracs[0]*b_hr+other_eth_fracs[1]*n_hr+other_eth_fracs[2]*a_hr+other_eth_fracs[3]*h_hr
other_hr = np.array([3.62095775, 3.83162408, 4.51816678, 4.73219442, 2.70416873])*hr
hosp_rate = np.array([w_hr,other_hr]).reshape(10)


#death rate after hospitalization
dr = np.array([4/100*sum(age_fracs_16[0:2])/sum(age_fracs_16[0:4])+12.365/100*sum(age_fracs_16[2:4])/sum(age_fracs_16[0:4]),3.122/100,
  10.745/100,10.745/100*sum(age_fracs_16[11:13])/sum(age_fracs_16[11:15]) + 23.158/100*sum(age_fracs_16[13:15])/sum(age_fracs_16[11:15]),23.158/100])
w_dr = np.array([1,1,1,1,1])*dr
b_dr = np.array([1,0.62222222, 1.6013289 , 0.51920246, 0.7553917 ])*dr
n_dr = np.array([1,2.56      , 1.86821705, 1.20390071, 1.2453755])*dr
a_dr = np.array([1,1.35757576, 0.37019969, 0.52918712, 0.85984538])*dr
h_dr = np.array([1,1.48258993, 1.30177683, 0.87799763, 0.72045836])*dr
other_dr = other_eth_fracs[0]*b_dr+other_eth_fracs[1]*n_dr+other_eth_fracs[2]*a_dr+other_eth_fracs[3]*h_dr
#other_dr =2.3213072934753667*dr
death_rate = np.array([dr,other_dr]).reshape(10)

#one_minus
oneMinusHospRate = np.ones(10) - hosp_rate
oneMinusSympRate = np.ones(10) - frac_sym
oneMinusDeathRate = np.ones(10) - death_rate

#reduced infectiousness
redA=0.75
redP=1
redH = 0
sigma_base = 1 / 3.8
sigma = sigma_base * np.ones(10)

#reduced susceptibility
red_sus = np.ones(10)
#red_sus[3]= 1*(0.505654635725246)+2.7*(0.49434536427475395)
#red_sus_b= 1*(0.5688380527270192)+2.7*(0.4311619472729809)
#red_sus_n= 1*(0.5404801786711334)+2.7*(0.45951982132886654)
#red_sus_a= 1*(0.5312675739511866)+2.7*(0.46873242604881343)
#red_sus_h= 1*(0.593489524667718)+2.7*(0.4065104753322821)
#red_sus[8]=other_eth_fracs[0]*red_sus_b+other_eth_fracs[1]*red_sus_n+other_eth_fracs[2]*red_sus_a+other_eth_fracs[3]*red_sus_h
#red_sus[[4,9]] = 2.7
#R0
R0 = 3

#pop
numGroups=10
totalPop = total_pops

beta = findBetaModel_coronavirusEqs(C, frac_sym, gammaA, gammaE,  gammaI, gammaP, hosp_rate, redA,  redP, red_sus, sigma, R0, totalPop)

    
#create new contact matrix
#fit_params = np.array([0.01000842, 0.37792662, 0.7       , 0.        ])
#np.array([0.99999997, 0.22149849, 0.5 ,       0.2       ])

VE_SYMPv=0.66
VE_v=0.7
VE_Iv=0.7
VE_Hv=0.9
VE_Dv=0.9

omega_R = 1/90
omega_Rv = 1/100
omega_Sv = 1/90
fitting_params = [beta, C, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaP, hosp_rate,  numGroups,
     oneMinusHospRate, oneMinusSympRate, redA, redH, redP, red_sus, sigma, totalPop,VE_Iv,
     VE_SYMPv,VE_v,VE_Hv,VE_Dv,omega_R,omega_Rv,omega_Sv,death_rate,oneMinusDeathRate]



