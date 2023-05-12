import numpy as np
import scipy
import scipy.integrate
import warnings
warnings.filterwarnings('ignore')

from two_pop_params_current import params,N,total_pops
from init_cond_current import y0_use

from vaccination_campaign import *

from scipy.optimize import LinearConstraint

from scipy.optimize import Bounds
bounds = Bounds([0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,])

deaths_from_curr_inf=np.array([4.58757571e-02, 4.69799864e-01, 2.31994490e+00, 1.24985607e+01,
 5.20870763e+01, 8.20421397e-02, 1.01792926e+00,2.86382141e+00,
 6.33601042e+00, 1.07779985e+01])


total_pops_age = total_pops[0:5]+total_pops[5:]

two_pop_daly=np.array([69.10735437, 44.0869492 , 24.14447896, 14.32827318,  3.81621329,69.10284857, 45.23438822, 24.52634191, 14.67306961,  3.97826531])

daly_mult_avg = np.array([total_pops[0]/total_pops_age[0],total_pops[1]/total_pops_age[1],total_pops[2]/total_pops_age[2],total_pops[3]/total_pops_age[3],total_pops[4]/total_pops_age[4],
	total_pops[5]/total_pops_age[0],total_pops[6]/total_pops_age[1],total_pops[7]/total_pops_age[2],total_pops[8]/total_pops_age[3],total_pops[9]/total_pops_age[4]])

two_pop_daly_total=daly_mult_avg[0:5]*two_pop_daly[0:5]+daly_mult_avg[5:]*two_pop_daly[5:]


def vaccinate_sus_y0(p,y0):
	y_new = y0.copy()
	v1= y0[1:5]*p[0:4]
	v2 = y0[6:10]*p[4:]
	y_new[1:5]=y0[1:5] - v1
	y_new[6:10]=y0[6:10] - v2
	y_new[10*8+1:10*8+5] = y0[10*8+1:10*8+5]+v1
	y_new[10*8+6:10*9] = y0[10*8+6:10*9]+v2
	return y_new

def adjust_p(p,c,pops):
	p=abs(p)
	if (p[0:4]@pops[1:5]+p[4:]@pops[6:10])>c*N:
		p[0:4]=(p[0:4]*pops[1:5]/N)*(c*N)/pops[1:5]
		p[4:]=(p[4:]*pops[6:10]/N)*(c*N)/pops[6:10] 
	return p


def obj_fn_ylls(y0,params,p,pops,deaths_from_curr_inf,c):
	p = adjust_p(p,c,y0[0:10])
	y0_new = y0.reshape(21,10)
	y0_new[18] = np.zeros(10)
	y0_new=vaccinate_sus_y0(p,y0_new.reshape(210))

	t = np.linspace(0,4*4*7,300)
	sol=scipy.integrate.solve_ivp(lambda t,y:coronavirusEqs_with_vaccination_only(t,y, params),[0,4*4*7], y0_new, t_eval=t)
	sol_plot=np.reshape(sol.y,(21,10,300))
	deaths = sol_plot[18,:,-1]-deaths_from_curr_inf 
	dalys = deaths*two_pop_daly
	dalys_per_100000 = (dalys/pops)*100000

	return sum(dalys_per_100000)

def obj_fn_ineq_ylls(y0,params,p,pops,deaths_from_curr_inf,c,weight):
	p = adjust_p(p,c,y0[0:10])
	y0_new = y0.reshape(21,10)
	y0_new[18] = np.zeros(10)
	y0_new=vaccinate_sus_y0(p,y0_new.reshape(210))

	t = np.linspace(0,4*4*7,300)
	sol=scipy.integrate.solve_ivp(lambda t,y:coronavirusEqs_with_vaccination_only(t,y, params),[0,4*4*7], y0_new, t_eval=t)
	sol_plot=np.reshape(sol.y,(21,10,300))
	deaths = sol_plot[18,:,-1]-deaths_from_curr_inf
	dalys = deaths*two_pop_daly
	dalys_per_100000 = (dalys/pops)*100000
    
	return sum(abs(dalys_per_100000[0:5]-dalys_per_100000[5:10]))**weight


def obj_fn_ineq_ylls_IDIS(y0,params,p,pops,deaths_from_curr_inf,c,weight):
	#https://nationalequityatlas.org/about-the-atlas/methodology/indexmethod
	p = adjust_p(p,c,y0[0:10])
	y0_new = y0.reshape(21,10)
	y0_new[18] = np.zeros(10)
	y0_new=vaccinate_sus_y0(p,y0_new.reshape(210))

	t = np.linspace(0,4*4*7,300)
	sol=scipy.integrate.solve_ivp(lambda t,y:coronavirusEqs_with_vaccination_only(t,y, params),[0,4*4*7], y0_new, t_eval=t)
	sol_plot=np.reshape(sol.y,(21,10,300))
	deaths = sol_plot[18,:,-1]-deaths_from_curr_inf 
	dalys = deaths*two_pop_daly
	dalys_per_100000 = (dalys/pops)*100000

	dalys_total=(deaths[0:5]+deaths[5:])*two_pop_daly_total
	dalys_total_per_100000 = (dalys_total/total_pops_age)*100000

	mort_idis = (100*sum((1/2)*(abs(dalys_per_100000[5:]-dalys_total_per_100000)+abs(dalys_per_100000[0:5]-dalys_total_per_100000))/dalys_total_per_100000))**weight

	return mort_idis


def obj_fn_both(y0,params,p,pops,deaths_from_curr_inf,c,ineq_func,weight):
	return  ineq_func(y0,params,p,pops,deaths_from_curr_inf,c,weight) + obj_fn_ylls(y0,params,p,pops,deaths_from_curr_inf,c)



def worker_ylls_10(x):
	linear_constraint= LinearConstraint([y0_use[1],y0_use[2],y0_use[3],y0_use[4],y0_use[6],y0_use[7],y0_use[8],y0_use[9]], [0], 0.1*N)
	ineq_cons = {'type': 'ineq',
	'fun' : lambda x: np.array([N*0.1 - x[0:4]@y0_use[1:5]-x[4:]@y0_use[6:10]])}

	min_f = lambda p: obj_fn_ylls(y0_use,params,p,total_pops,deaths_from_curr_inf,0.1)
	res_temp1 = scipy.optimize.minimize(min_f, x,method='trust-constr', bounds = bounds,constraints=linear_constraint)
	nm_res1= res_temp1.x
	obj_fn_evals1= res_temp1.fun
	
	res_temp2 = scipy.optimize.minimize(min_f, x,method='SLSQP', constraints = ineq_cons,bounds=[(0, 1) for i in range(len(x))])
	nm_res2= res_temp2.x
	obj_fn_evals2= res_temp2.fun

	res_temp3 = scipy.optimize.minimize(min_f, x,method='nelder-mead', bounds = bounds)
	nm_res3= res_temp3.x
	obj_fn_evals3= res_temp3.fun

	min_val = min(obj_fn_evals1,obj_fn_evals2,obj_fn_evals3)
	if min_val==obj_fn_evals1:
		return [nm_res1,obj_fn_evals1]
	elif min_val==obj_fn_evals2:
		return [nm_res2,obj_fn_evals2]
	else:
		return [nm_res3,obj_fn_evals3]

def worker_ylls_20(x):
	linear_constraint= LinearConstraint([y0_use[1],y0_use[2],y0_use[3],y0_use[4],y0_use[6],y0_use[7],y0_use[8],y0_use[9]], [0], 0.2*N)
	ineq_cons = {'type': 'ineq',
	'fun' : lambda x: np.array([N*0.2 - x[0:4]@y0_use[1:5]-x[4:]@y0_use[6:10]])}

	min_f = lambda p: obj_fn_ylls(y0_use,params,p,total_pops,deaths_from_curr_inf,0.2)
	res_temp1 = scipy.optimize.minimize(min_f, x,method='trust-constr', bounds = bounds,constraints=linear_constraint)
	nm_res1= res_temp1.x
	obj_fn_evals1= res_temp1.fun
	
	res_temp2 = scipy.optimize.minimize(min_f, x,method='SLSQP', constraints = ineq_cons,bounds=[(0, 1) for i in range(len(x))])
	nm_res2= res_temp2.x
	obj_fn_evals2= res_temp2.fun

	res_temp3 = scipy.optimize.minimize(min_f, x,method='nelder-mead', bounds = bounds)
	nm_res3= res_temp3.x
	obj_fn_evals3= res_temp3.fun

	min_val = min(obj_fn_evals1,obj_fn_evals2,obj_fn_evals3)
	if min_val==obj_fn_evals1:
		return [nm_res1,obj_fn_evals1]
	elif min_val==obj_fn_evals2:
		return [nm_res2,obj_fn_evals2]
	else:
		return [nm_res3,obj_fn_evals3]

def worker_ylls_25(x):
	linear_constraint= LinearConstraint([y0_use[1],y0_use[2],y0_use[3],y0_use[4],y0_use[6],y0_use[7],y0_use[8],y0_use[9]], [0], 0.25*N)
	ineq_cons = {'type': 'ineq',
	'fun' : lambda x: np.array([N*0.2 - x[0:4]@y0_use[1:5]-x[4:]@y0_use[6:10]])}

	min_f = lambda p: obj_fn_ylls(y0_use,params,p,total_pops,deaths_from_curr_inf,0.25)
	res_temp1 = scipy.optimize.minimize(min_f, x,method='trust-constr', bounds = bounds,constraints=linear_constraint)
	nm_res1= res_temp1.x
	obj_fn_evals1= res_temp1.fun
	
	res_temp2 = scipy.optimize.minimize(min_f, x,method='SLSQP', constraints = ineq_cons,bounds=[(0, 1) for i in range(len(x))])
	nm_res2= res_temp2.x
	obj_fn_evals2= res_temp2.fun

	res_temp3 = scipy.optimize.minimize(min_f, x,method='nelder-mead', bounds = bounds)
	nm_res3= res_temp3.x
	obj_fn_evals3= res_temp3.fun

	min_val = min(obj_fn_evals1,obj_fn_evals2,obj_fn_evals3)
	if min_val==obj_fn_evals1:
		return [nm_res1,obj_fn_evals1]
	elif min_val==obj_fn_evals2:
		return [nm_res2,obj_fn_evals2]
	else:
		return [nm_res3,obj_fn_evals3]

def worker_ylls_30(x):
	linear_constraint= LinearConstraint([y0_use[1],y0_use[2],y0_use[3],y0_use[4],y0_use[6],y0_use[7],y0_use[8],y0_use[9]], [0], 0.3*N)
	ineq_cons = {'type': 'ineq',
	'fun' : lambda x: np.array([N*0.3 - x[0:4]@y0_use[1:5]-x[4:]@y0_use[6:10]])}

	min_f = lambda p: obj_fn_ylls(y0_use,params,p,total_pops,deaths_from_curr_inf,0.3)
	res_temp1 = scipy.optimize.minimize(min_f, x,method='trust-constr', bounds = bounds,constraints=linear_constraint)
	nm_res1= res_temp1.x
	obj_fn_evals1= res_temp1.fun
	
	res_temp2 = scipy.optimize.minimize(min_f, x,method='SLSQP', constraints = ineq_cons,bounds=[(0, 1) for i in range(len(x))])
	nm_res2= res_temp2.x
	obj_fn_evals2= res_temp2.fun

	res_temp3 = scipy.optimize.minimize(min_f, x,method='nelder-mead', bounds = bounds)
	nm_res3= res_temp3.x
	obj_fn_evals3= res_temp3.fun

	min_val = min(obj_fn_evals1,obj_fn_evals2,obj_fn_evals3)
	if min_val==obj_fn_evals1:
		return [nm_res1,obj_fn_evals1]
	elif min_val==obj_fn_evals2:
		return [nm_res2,obj_fn_evals2]
	else:
		return [nm_res3,obj_fn_evals3]

def worker_ineq_ylls_10(x):
	linear_constraint= LinearConstraint([y0_use[1],y0_use[2],y0_use[3],y0_use[4],y0_use[6],y0_use[7],y0_use[8],y0_use[9]], [0], 0.1*N)
	ineq_cons = {'type': 'ineq',
	'fun' : lambda x: np.array([N*0.1 - x[0:4]@y0_use[1:5]-x[4:]@y0_use[6:10]])}

	min_f = lambda p: obj_fn_ineq_ylls(y0_use,params,p,total_pops,deaths_from_curr_inf,0.1,1)
	res_temp1 = scipy.optimize.minimize(min_f, x,method='trust-constr', bounds = bounds,constraints=linear_constraint)
	nm_res1= res_temp1.x
	obj_fn_evals1= res_temp1.fun

	res_temp2 = scipy.optimize.minimize(min_f, x,method='SLSQP', constraints = ineq_cons,bounds=[(0, 1) for i in range(len(x))])
	nm_res2= res_temp2.x
	obj_fn_evals2= res_temp2.fun

	res_temp3 = scipy.optimize.minimize(min_f, x,method='nelder-mead', bounds = bounds)
	nm_res3= res_temp3.x
	obj_fn_evals3= res_temp3.fun

	min_val = min(obj_fn_evals1,obj_fn_evals2,obj_fn_evals3)
	if min_val==obj_fn_evals1:
		return [nm_res1,obj_fn_evals1]
	elif min_val==obj_fn_evals2:
		return [nm_res2,obj_fn_evals2]
	else:
		return [nm_res3,obj_fn_evals3]

def worker_ineq_ylls_20(x):
	linear_constraint= LinearConstraint([y0_use[1],y0_use[2],y0_use[3],y0_use[4],y0_use[6],y0_use[7],y0_use[8],y0_use[9]], [0], 0.2*N)
	ineq_cons = {'type': 'ineq',
	'fun' : lambda x: np.array([N*0.2 - x[0:4]@y0_use[1:5]-x[4:]@y0_use[6:10]])}

	min_f = lambda p: obj_fn_ineq_ylls(y0_use,params,p,total_pops,deaths_from_curr_inf,0.2,1)
	res_temp1 = scipy.optimize.minimize(min_f, x,method='trust-constr', bounds = bounds,constraints=linear_constraint)
	nm_res1= res_temp1.x
	obj_fn_evals1= res_temp1.fun

	res_temp2 = scipy.optimize.minimize(min_f, x,method='SLSQP', constraints = ineq_cons,bounds=[(0, 1) for i in range(len(x))])
	nm_res2= res_temp2.x
	obj_fn_evals2= res_temp2.fun

	res_temp3 = scipy.optimize.minimize(min_f, x,method='nelder-mead', bounds = bounds)
	nm_res3= res_temp3.x
	obj_fn_evals3= res_temp3.fun

	min_val = min(obj_fn_evals1,obj_fn_evals2,obj_fn_evals3)
	if min_val==obj_fn_evals1:
		return [nm_res1,obj_fn_evals1]
	elif min_val==obj_fn_evals2:
		return [nm_res2,obj_fn_evals2]
	else:
		return [nm_res3,obj_fn_evals3]

def worker_ineq_ylls_25(x):
	linear_constraint= LinearConstraint([y0_use[1],y0_use[2],y0_use[3],y0_use[4],y0_use[6],y0_use[7],y0_use[8],y0_use[9]], [0], 0.25*N)
	ineq_cons = {'type': 'ineq',
	'fun' : lambda x: np.array([N*0.2 - x[0:4]@y0_use[1:5]-x[4:]@y0_use[6:10]])}

	min_f = lambda p: obj_fn_ineq_ylls(y0_use,params,p,total_pops,deaths_from_curr_inf,0.25,1)
	res_temp1 = scipy.optimize.minimize(min_f, x,method='trust-constr', bounds = bounds,constraints=linear_constraint)
	nm_res1= res_temp1.x
	obj_fn_evals1= res_temp1.fun

	res_temp2 = scipy.optimize.minimize(min_f, x,method='SLSQP', constraints = ineq_cons,bounds=[(0, 1) for i in range(len(x))])
	nm_res2= res_temp2.x
	obj_fn_evals2= res_temp2.fun

	res_temp3 = scipy.optimize.minimize(min_f, x,method='nelder-mead', bounds = bounds)
	nm_res3= res_temp3.x
	obj_fn_evals3= res_temp3.fun

	min_val = min(obj_fn_evals1,obj_fn_evals2,obj_fn_evals3)
	if min_val==obj_fn_evals1:
		return [nm_res1,obj_fn_evals1]
	elif min_val==obj_fn_evals2:
		return [nm_res2,obj_fn_evals2]
	else:
		return [nm_res3,obj_fn_evals3]

def worker_ineq_ylls_30(x):
	linear_constraint= LinearConstraint([y0_use[1],y0_use[2],y0_use[3],y0_use[4],y0_use[6],y0_use[7],y0_use[8],y0_use[9]], [0], 0.3*N)
	ineq_cons = {'type': 'ineq',
	'fun' : lambda x: np.array([N*0.3 - x[0:4]@y0_use[1:5]-x[4:]@y0_use[6:10]])}

	min_f = lambda p: obj_fn_ineq_ylls(y0_use,params,p,total_pops,deaths_from_curr_inf,0.3,1)
	res_temp1 = scipy.optimize.minimize(min_f, x,method='trust-constr', bounds = bounds,constraints=linear_constraint)
	nm_res1= res_temp1.x
	obj_fn_evals1= res_temp1.fun

	res_temp2 = scipy.optimize.minimize(min_f, x,method='SLSQP', constraints = ineq_cons,bounds=[(0, 1) for i in range(len(x))])
	nm_res2= res_temp2.x
	obj_fn_evals2= res_temp2.fun

	res_temp3 = scipy.optimize.minimize(min_f, x,method='nelder-mead', bounds = bounds)
	nm_res3= res_temp3.x
	obj_fn_evals3= res_temp3.fun

	min_val = min(obj_fn_evals1,obj_fn_evals2,obj_fn_evals3)
	if min_val==obj_fn_evals1:
		return [nm_res1,obj_fn_evals1]
	elif min_val==obj_fn_evals2:
		return [nm_res2,obj_fn_evals2]
	else:
		return [nm_res3,obj_fn_evals3]

def worker_ineq_ylls_idis_10(x):
	linear_constraint= LinearConstraint([y0_use[1],y0_use[2],y0_use[3],y0_use[4],y0_use[6],y0_use[7],y0_use[8],y0_use[9]], [0], 0.1*N)
	ineq_cons = {'type': 'ineq',
	'fun' : lambda x: np.array([N*0.1 - x[0:4]@y0_use[1:5]-x[4:]@y0_use[6:10]])}

	min_f = lambda p: obj_fn_ineq_ylls_IDIS(y0_use,params,p,total_pops,deaths_from_curr_inf,0.1,0)
	res_temp1 = scipy.optimize.minimize(min_f, x,method='trust-constr', bounds = bounds,constraints=linear_constraint)
	nm_res1= res_temp1.x
	obj_fn_evals1= res_temp1.fun

	res_temp2 = scipy.optimize.minimize(min_f, x,method='SLSQP', constraints = ineq_cons,bounds=[(0, 1) for i in range(len(x))])
	nm_res2= res_temp2.x
	obj_fn_evals2= res_temp2.fun

	res_temp3 = scipy.optimize.minimize(min_f, x,method='nelder-mead', bounds = bounds)
	nm_res3= res_temp3.x
	obj_fn_evals3= res_temp3.fun

	min_val = min(obj_fn_evals1,obj_fn_evals2,obj_fn_evals3)
	if min_val==obj_fn_evals1:
		return [nm_res1,obj_fn_evals1]
	elif min_val==obj_fn_evals2:
		return [nm_res2,obj_fn_evals2]
	else:
		return [nm_res3,obj_fn_evals3]

def worker_ineq_ylls_idis_20(x):
	linear_constraint= LinearConstraint([y0_use[1],y0_use[2],y0_use[3],y0_use[4],y0_use[6],y0_use[7],y0_use[8],y0_use[9]], [0], 0.2*N)
	ineq_cons = {'type': 'ineq',
	'fun' : lambda x: np.array([N*0.2 - x[0:4]@y0_use[1:5]-x[4:]@y0_use[6:10]])}

	min_f = lambda p: obj_fn_ineq_ylls_IDIS(y0_use,params,p,total_pops,deaths_from_curr_inf,0.2,0)
	res_temp1 = scipy.optimize.minimize(min_f, x,method='trust-constr', bounds = bounds,constraints=linear_constraint)
	nm_res1= res_temp1.x
	obj_fn_evals1= res_temp1.fun

	res_temp2 = scipy.optimize.minimize(min_f, x,method='SLSQP', constraints = ineq_cons,bounds=[(0, 1) for i in range(len(x))])
	nm_res2= res_temp2.x
	obj_fn_evals2= res_temp2.fun

	res_temp3 = scipy.optimize.minimize(min_f, x,method='nelder-mead', bounds = bounds)
	nm_res3= res_temp3.x
	obj_fn_evals3= res_temp3.fun

	min_val = min(obj_fn_evals1,obj_fn_evals2,obj_fn_evals3)
	if min_val==obj_fn_evals1:
		return [nm_res1,obj_fn_evals1]
	elif min_val==obj_fn_evals2:
		return [nm_res2,obj_fn_evals2]
	else:
		return [nm_res3,obj_fn_evals3]

def worker_ineq_ylls_idis_25(x):
	linear_constraint= LinearConstraint([y0_use[1],y0_use[2],y0_use[3],y0_use[4],y0_use[6],y0_use[7],y0_use[8],y0_use[9]], [0], 0.25*N)
	ineq_cons = {'type': 'ineq',
	'fun' : lambda x: np.array([N*0.2 - x[0:4]@y0_use[1:5]-x[4:]@y0_use[6:10]])}

	min_f = lambda p: obj_fn_ineq_ylls_IDIS(y0_use,params,p,total_pops,deaths_from_curr_inf,0.25,0)
	res_temp1 = scipy.optimize.minimize(min_f, x,method='trust-constr', bounds = bounds,constraints=linear_constraint)
	nm_res1= res_temp1.x
	obj_fn_evals1= res_temp1.fun

	res_temp2 = scipy.optimize.minimize(min_f, x,method='SLSQP', constraints = ineq_cons,bounds=[(0, 1) for i in range(len(x))])
	nm_res2= res_temp2.x
	obj_fn_evals2= res_temp2.fun

	res_temp3 = scipy.optimize.minimize(min_f, x,method='nelder-mead', bounds = bounds)
	nm_res3= res_temp3.x
	obj_fn_evals3= res_temp3.fun

	min_val = min(obj_fn_evals1,obj_fn_evals2,obj_fn_evals3)
	if min_val==obj_fn_evals1:
		return [nm_res1,obj_fn_evals1]
	elif min_val==obj_fn_evals2:
		return [nm_res2,obj_fn_evals2]
	else:
		return [nm_res3,obj_fn_evals3]

def worker_ineq_ylls_idis_30(x):
	linear_constraint= LinearConstraint([y0_use[1],y0_use[2],y0_use[3],y0_use[4],y0_use[6],y0_use[7],y0_use[8],y0_use[9]], [0], 0.3*N)
	ineq_cons = {'type': 'ineq',
	'fun' : lambda x: np.array([N*0.3 - x[0:4]@y0_use[1:5]-x[4:]@y0_use[6:10]])}

	min_f = lambda p: obj_fn_ineq_ylls_IDIS(y0_use,params,p,total_pops,deaths_from_curr_inf,0.3,0)
	res_temp1 = scipy.optimize.minimize(min_f, x,method='trust-constr', bounds = bounds,constraints=linear_constraint)
	nm_res1= res_temp1.x
	obj_fn_evals1= res_temp1.fun

	res_temp2 = scipy.optimize.minimize(min_f, x,method='SLSQP', constraints = ineq_cons,bounds=[(0, 1) for i in range(len(x))])
	nm_res2= res_temp2.x
	obj_fn_evals2= res_temp2.fun

	res_temp3 = scipy.optimize.minimize(min_f, x,method='nelder-mead', bounds = bounds)
	nm_res3= res_temp3.x
	obj_fn_evals3= res_temp3.fun

	min_val = min(obj_fn_evals1,obj_fn_evals2,obj_fn_evals3)
	if min_val==obj_fn_evals1:
		return [nm_res1,obj_fn_evals1]
	elif min_val==obj_fn_evals2:
		return [nm_res2,obj_fn_evals2]
	else:
		return [nm_res3,obj_fn_evals3]



def worker_both_10(x):
	linear_constraint= LinearConstraint([y0_use[1],y0_use[2],y0_use[3],y0_use[4],y0_use[6],y0_use[7],y0_use[8],y0_use[9]], [0], 0.1*N)
	ineq_cons = {'type': 'ineq',
	'fun' : lambda x: np.array([N*0.1 - x[0:4]@y0_use[1:5]-x[4:]@y0_use[6:10]])}

	min_f = lambda p: obj_fn_both(y0_use,params,p,total_pops,deaths_from_curr_inf,0.1,obj_fn_ineq_ylls,1)
	res_temp1 = scipy.optimize.minimize(min_f, x,method='trust-constr', bounds = bounds,constraints=linear_constraint)
	nm_res1= res_temp1.x
	obj_fn_evals1= res_temp1.fun

	res_temp2 = scipy.optimize.minimize(min_f, x,method='SLSQP', constraints = ineq_cons,bounds=[(0, 1) for i in range(len(x))])
	nm_res2= res_temp2.x
	obj_fn_evals2= res_temp2.fun

	res_temp3 = scipy.optimize.minimize(min_f, x,method='nelder-mead', bounds = bounds)
	nm_res3= res_temp3.x
	obj_fn_evals3= res_temp3.fun

	min_val = min(obj_fn_evals1,obj_fn_evals2,obj_fn_evals3)
	if min_val==obj_fn_evals1:
		return [nm_res1,obj_fn_evals1]
	elif min_val==obj_fn_evals2:
		return [nm_res2,obj_fn_evals2]
	else:
		return [nm_res3,obj_fn_evals3]

def worker_both_20(x):
	linear_constraint= LinearConstraint([y0_use[1],y0_use[2],y0_use[3],y0_use[4],y0_use[6],y0_use[7],y0_use[8],y0_use[9]], [0], 0.2*N)
	ineq_cons = {'type': 'ineq',
	'fun' : lambda x: np.array([N*0.2 - x[0:4]@y0_use[1:5]-x[4:]@y0_use[6:10]])}

	min_f = lambda p: obj_fn_both(y0_use,params,p,total_pops,deaths_from_curr_inf,0.2, obj_fn_ineq_ylls,2)
	res_temp1 = scipy.optimize.minimize(min_f, x,method='trust-constr', bounds = bounds,constraints=linear_constraint)
	nm_res1= res_temp1.x
	obj_fn_evals1= res_temp1.fun

	res_temp2 = scipy.optimize.minimize(min_f, x,method='SLSQP', constraints = ineq_cons,bounds=[(0, 1) for i in range(len(x))])
	nm_res2= res_temp2.x
	obj_fn_evals2= res_temp2.fun

	res_temp3 = scipy.optimize.minimize(min_f, x,method='nelder-mead', bounds = bounds)
	nm_res3= res_temp3.x
	obj_fn_evals3= res_temp3.fun

	min_val = min(obj_fn_evals1,obj_fn_evals2,obj_fn_evals3)
	if min_val==obj_fn_evals1:
		return [nm_res1,obj_fn_evals1]
	elif min_val==obj_fn_evals2:
		return [nm_res2,obj_fn_evals2]
	else:
		return [nm_res3,obj_fn_evals3]

def worker_both_25(x):
	linear_constraint= LinearConstraint([y0_use[1],y0_use[2],y0_use[3],y0_use[4],y0_use[6],y0_use[7],y0_use[8],y0_use[9]], [0], 0.25*N)
	ineq_cons = {'type': 'ineq',
	'fun' : lambda x: np.array([N*0.2 - x[0:4]@y0_use[1:5]-x[4:]@y0_use[6:10]])}

	min_f = lambda p: obj_fn_both(y0_use,params,p,total_pops,deaths_from_curr_inf,0.25, obj_fn_ineq_ylls,2)
	res_temp1 = scipy.optimize.minimize(min_f, x,method='trust-constr', bounds = bounds,constraints=linear_constraint)
	nm_res1= res_temp1.x
	obj_fn_evals1= res_temp1.fun

	res_temp2 = scipy.optimize.minimize(min_f, x,method='SLSQP', constraints = ineq_cons,bounds=[(0, 1) for i in range(len(x))])
	nm_res2= res_temp2.x
	obj_fn_evals2= res_temp2.fun

	res_temp3 = scipy.optimize.minimize(min_f, x,method='nelder-mead', bounds = bounds)
	nm_res3= res_temp3.x
	obj_fn_evals3= res_temp3.fun

	min_val = min(obj_fn_evals1,obj_fn_evals2,obj_fn_evals3)
	if min_val==obj_fn_evals1:
		return [nm_res1,obj_fn_evals1]
	elif min_val==obj_fn_evals2:
		return [nm_res2,obj_fn_evals2]
	else:
		return [nm_res3,obj_fn_evals3]


def worker_both_30(x):
	linear_constraint= LinearConstraint([y0_use[1],y0_use[2],y0_use[3],y0_use[4],y0_use[6],y0_use[7],y0_use[8],y0_use[9]], [0], 0.3*N)
	ineq_cons = {'type': 'ineq',
	'fun' : lambda x: np.array([N*0.3 - x[0:4]@y0_use[1:5]-x[4:]@y0_use[6:10]])}

	min_f = lambda p: obj_fn_both(y0_use,params,p,total_pops,deaths_from_curr_inf,0.3, obj_fn_ineq_ylls,2)
	res_temp1 = scipy.optimize.minimize(min_f, x,method='trust-constr', bounds = bounds,constraints=linear_constraint)
	nm_res1= res_temp1.x
	obj_fn_evals1= res_temp1.fun

	res_temp2 = scipy.optimize.minimize(min_f, x,method='SLSQP', constraints = ineq_cons,bounds=[(0, 1) for i in range(len(x))])
	nm_res2= res_temp2.x
	obj_fn_evals2= res_temp2.fun

	res_temp3 = scipy.optimize.minimize(min_f, x,method='nelder-mead', bounds = bounds)
	nm_res3= res_temp3.x
	obj_fn_evals3= res_temp3.fun

	min_val = min(obj_fn_evals1,obj_fn_evals2,obj_fn_evals3)
	if min_val==obj_fn_evals1:
		return [nm_res1,obj_fn_evals1]
	elif min_val==obj_fn_evals2:
		return [nm_res2,obj_fn_evals2]
	else:
		return [nm_res3,obj_fn_evals3]

def worker_both_idis_10(x):
	linear_constraint= LinearConstraint([y0_use[1],y0_use[2],y0_use[3],y0_use[4],y0_use[6],y0_use[7],y0_use[8],y0_use[9]], [0], 0.1*N)
	ineq_cons = {'type': 'ineq',
	'fun' : lambda x: np.array([N*0.1 - x[0:4]@y0_use[1:5]-x[4:]@y0_use[6:10]])}

	min_f = lambda p: obj_fn_both(y0_use,params,p,total_pops,deaths_from_curr_inf,0.1,obj_fn_ineq_ylls_IDIS,2)
	res_temp1 = scipy.optimize.minimize(min_f, x,method='trust-constr', bounds = bounds,constraints=linear_constraint)
	nm_res1= res_temp1.x
	obj_fn_evals1= res_temp1.fun

	res_temp2 = scipy.optimize.minimize(min_f, x,method='SLSQP', constraints = ineq_cons,bounds=[(0, 1) for i in range(len(x))])
	nm_res2= res_temp2.x
	obj_fn_evals2= res_temp2.fun

	res_temp3 = scipy.optimize.minimize(min_f, x,method='nelder-mead', bounds = bounds)
	nm_res3= res_temp3.x
	obj_fn_evals3= res_temp3.fun

	min_val = min(obj_fn_evals1,obj_fn_evals2,obj_fn_evals3)
	if min_val==obj_fn_evals1:
		return [nm_res1,obj_fn_evals1]
	elif min_val==obj_fn_evals2:
		return [nm_res2,obj_fn_evals2]
	else:
		return [nm_res3,obj_fn_evals3]

def worker_both_idis_20(x):
	linear_constraint= LinearConstraint([y0_use[1],y0_use[2],y0_use[3],y0_use[4],y0_use[6],y0_use[7],y0_use[8],y0_use[9]], [0], 0.2*N)
	ineq_cons = {'type': 'ineq',
	'fun' : lambda x: np.array([N*0.2 - x[0:4]@y0_use[1:5]-x[4:]@y0_use[6:10]])}

	min_f = lambda p: obj_fn_both(y0_use,params,p,total_pops,deaths_from_curr_inf,0.2, obj_fn_ineq_ylls_IDIS,2)
	res_temp1 = scipy.optimize.minimize(min_f, x,method='trust-constr', bounds = bounds,constraints=linear_constraint)
	nm_res1= res_temp1.x
	obj_fn_evals1= res_temp1.fun

	res_temp2 = scipy.optimize.minimize(min_f, x,method='SLSQP', constraints = ineq_cons,bounds=[(0, 1) for i in range(len(x))])
	nm_res2= res_temp2.x
	obj_fn_evals2= res_temp2.fun

	res_temp3 = scipy.optimize.minimize(min_f, x,method='nelder-mead', bounds = bounds)
	nm_res3= res_temp3.x
	obj_fn_evals3= res_temp3.fun

	min_val = min(obj_fn_evals1,obj_fn_evals2,obj_fn_evals3)
	if min_val==obj_fn_evals1:
		return [nm_res1,obj_fn_evals1]
	elif min_val==obj_fn_evals2:
		return [nm_res2,obj_fn_evals2]
	else:
		return [nm_res3,obj_fn_evals3]

def worker_both_idis_25(x):
	linear_constraint= LinearConstraint([y0_use[1],y0_use[2],y0_use[3],y0_use[4],y0_use[6],y0_use[7],y0_use[8],y0_use[9]], [0], 0.25*N)
	ineq_cons = {'type': 'ineq',
	'fun' : lambda x: np.array([N*0.2 - x[0:4]@y0_use[1:5]-x[4:]@y0_use[6:10]])}

	min_f = lambda p: obj_fn_both(y0_use,params,p,total_pops,deaths_from_curr_inf,0.25, obj_fn_ineq_ylls_IDIS,2)
	res_temp1 = scipy.optimize.minimize(min_f, x,method='trust-constr', bounds = bounds,constraints=linear_constraint)
	nm_res1= res_temp1.x
	obj_fn_evals1= res_temp1.fun

	res_temp2 = scipy.optimize.minimize(min_f, x,method='SLSQP', constraints = ineq_cons,bounds=[(0, 1) for i in range(len(x))])
	nm_res2= res_temp2.x
	obj_fn_evals2= res_temp2.fun

	res_temp3 = scipy.optimize.minimize(min_f, x,method='nelder-mead', bounds = bounds)
	nm_res3= res_temp3.x
	obj_fn_evals3= res_temp3.fun

	min_val = min(obj_fn_evals1,obj_fn_evals2,obj_fn_evals3)
	if min_val==obj_fn_evals1:
		return [nm_res1,obj_fn_evals1]
	elif min_val==obj_fn_evals2:
		return [nm_res2,obj_fn_evals2]
	else:
		return [nm_res3,obj_fn_evals3]

def worker_both_idis_30(x):
	linear_constraint= LinearConstraint([y0_use[1],y0_use[2],y0_use[3],y0_use[4],y0_use[6],y0_use[7],y0_use[8],y0_use[9]], [0], 0.3*N)
	ineq_cons = {'type': 'ineq',
	'fun' : lambda x: np.array([N*0.3 - x[0:4]@y0_use[1:5]-x[4:]@y0_use[6:10]])}

	min_f = lambda p: obj_fn_both(y0_use,params,p,total_pops,deaths_from_curr_inf,0.3, obj_fn_ineq_ylls_IDIS,2)
	res_temp1 = scipy.optimize.minimize(min_f, x,method='trust-constr', bounds = bounds,constraints=linear_constraint)
	nm_res1= res_temp1.x
	obj_fn_evals1= res_temp1.fun

	res_temp2 = scipy.optimize.minimize(min_f, x,method='SLSQP', constraints = ineq_cons,bounds=[(0, 1) for i in range(len(x))])
	nm_res2= res_temp2.x
	obj_fn_evals2= res_temp2.fun

	res_temp3 = scipy.optimize.minimize(min_f, x,method='nelder-mead', bounds = bounds)
	nm_res3= res_temp3.x
	obj_fn_evals3= res_temp3.fun

	min_val = min(obj_fn_evals1,obj_fn_evals2,obj_fn_evals3)
	if min_val==obj_fn_evals1:
		return [nm_res1,obj_fn_evals1]
	elif min_val==obj_fn_evals2:
		return [nm_res2,obj_fn_evals2]
	else:
		return [nm_res3,obj_fn_evals3]



def sp_ylls_10(x):
    return obj_fn_ylls(y0_use,params,x,total_pops,deaths_from_curr_inf,0.1)

def sp_ylls_20(x):
    return obj_fn_ylls(y0_use,params,x,total_pops,deaths_from_curr_inf,0.2)

def sp_ylls_25(x):
    return obj_fn_ylls(y0_use,params,x,total_pops,deaths_from_curr_inf,0.25)

def sp_ylls_30(x):
    return obj_fn_ylls(y0_use,params,x,total_pops,deaths_from_curr_inf,0.3)    

def sp_ineq_ylls_10(x):
    return obj_fn_ineq_ylls(y0_use,params,x,total_pops,deaths_from_curr_inf,0.1,1)

def sp_ineq_ylls_20(x):
    return obj_fn_ineq_ylls(y0_use,params,x,total_pops,deaths_from_curr_inf,0.2,1)

def sp_ineq_ylls_25(x):
    return obj_fn_ineq_ylls(y0_use,params,x,total_pops,deaths_from_curr_inf,0.25,1)

def sp_ineq_ylls_30(x):
    return obj_fn_ineq_ylls(y0_use,params,x,total_pops,deaths_from_curr_inf,0.3,1)

def sp_ineq_idis_10(x):
    return obj_fn_ineq_ylls_IDIS(y0_use,params,x,total_pops,deaths_from_curr_inf,0.1,0)

def sp_ineq_idis_20(x):
    return obj_fn_ineq_ylls_IDIS(y0_use,params,x,total_pops,deaths_from_curr_inf,0.2,0)

def sp_ineq_idis_25(x):
    return obj_fn_ineq_ylls_IDIS(y0_use,params,x,total_pops,deaths_from_curr_inf,0.25,0)

def sp_ineq_idis_30(x):
    return obj_fn_ineq_ylls_IDIS(y0_use,params,x,total_pops,deaths_from_curr_inf,0.3,0)

def sp_both_ylls_10(x):
	return obj_fn_both(y0_use,params,x,total_pops,deaths_from_curr_inf,0.1, obj_fn_ineq_ylls,2)

def sp_both_ylls_20(x):
	return obj_fn_both(y0_use,params,x,total_pops,deaths_from_curr_inf,0.2, obj_fn_ineq_ylls,2)

def sp_both_ylls_25(x):
	return obj_fn_both(y0_use,params,x,total_pops,deaths_from_curr_inf,0.25, obj_fn_ineq_ylls,2)

def sp_both_ylls_30(x):
	return obj_fn_both(y0_use,params,x,total_pops,deaths_from_curr_inf,0.3, obj_fn_ineq_ylls,2)

def sp_both_idis_10(x):
	return obj_fn_both(y0_use,params,x,total_pops,deaths_from_curr_inf,0.1, obj_fn_ineq_ylls_IDIS,2)

def sp_both_idis_20(x):
	return obj_fn_both(y0_use,params,x,total_pops,deaths_from_curr_inf,0.2, obj_fn_ineq_ylls_IDIS,2)

def sp_both_idis_25(x):
	return obj_fn_both(y0_use,params,x,total_pops,deaths_from_curr_inf,0.25, obj_fn_ineq_ylls_IDIS,2)

def sp_both_idis_30(x):
	return obj_fn_both(y0_use,params,x,total_pops,deaths_from_curr_inf,0.3, obj_fn_ineq_ylls_IDIS,2)
