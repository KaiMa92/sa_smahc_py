# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 17:21:57 2022

@author: kaiser
"""

from pysma.models.Python_models.lumped_sma_model_X import sma_model as sm
import pysma
import json
import numpy as np
import pandas as pd
from pysma.models.heat_transfer_coefficient import *
import time
from scipy.integrate import solve_ivp
from scipy.optimize import newton
from scipy.optimize import root

with open("INPUT.txt") as input_data_file:
    ipt_dct = json.load(input_data_file)
    
    

t0 = time.time()
F_z = - ipt_dct['Load']
alpha = ipt_dct['Alpha elastomer'] 
Tu = ipt_dct['Ambient temperature cluster']
sequences = ipt_dct['Sequences']
dt = ipt_dct['Time increment']
dx = ipt_dct['Spatial increment']
mf0 = ipt_dct['mf0']
stress0 = ipt_dct['stress0']

A = pysma.material.SMAHC(ipt_dct['Actuator type'])
S = A.w
L0 = A.w.length
lam_coeff = -A.stiffness


E_cond = 0
E_conv = 0
E_loss = 0 
E_cond_sum = 0
E_conv_sum = 0 
E_loss_sum = 0
Ein_sum = 0
Usum = 0 
state = 0
T = Tu
T0 = Tu
a = 0
dT = 0
Ein = 0
U = 0
E_loss = 0
stress = stress0
E = S.EM
mf = mf0
mf_1 = mf0
r = S.rM
resistance = r * L0 / (np.pi*((S.diameter/2)**2))
real_As = S.As + stress0/S.CAs
real_Af = S.Af + stress0/S.CAf
real_Mf = S.Mf + stress0/S.CMf
real_Ms = S.Ms + stress0/S.CMs
eps_tr = S.max_strain_zero_load + (S.max_trans_strain - S.max_strain_zero_load)*(1-np.exp(-S.k*(stress0)/S.EA))
strain0 = eps_tr
strain = strain0
R = np.inf
deflection = 0
xmax = A.length


hc = horizontal_cylinder(S.diameter, Tu)
#alpha =hc.alpha(T)


t_tot = sum(sequence[1] for sequence in sequences)
t = 0


column_names = ['t','dt', 'stress0', 'mf0', 'strain0', 'L0', 'E_loss', 'current', 'stress', 'T', 'E', 'mf', 'strain', 'r', 'resistance', 'real_As', 'real_Af', 'real_Mf', 'real_Ms', 'a', 'U', 'dT', 'Ein', 'E_cond', 'E_conv', 'E_loss' , 'E_cond_sum', 'E_conv_sum', 'E_loss_sum', 'Ein_sum', 'Usum', 'deflection', 'xmax', 'state', 'eps_tr', 'alpha']
data_array = np.ones((int((t_tot-t)/dt),len(column_names)))
idx = 0


ht = pysma.models.Python_models.heat_transfer_2D_elastomer_only.Heat_transfer(A, dx, Tu, alpha, dt)


u0_d = ht.u0_d
u0_s = ht.u0_s
u_d = ht.u_d
u_s = ht.u_s
n = 0

u0_d_dct = {}
u0_s_dct = {}


n = 2000 # Knotenanzahl wählen
h = S.length / n #Länge der Intervalle
t_H = A.dist_sub_sma
A_sma = A.g.n * A.w.crosssection_area

def f(s, k): #Gibt Ableitung der DGL der Biegelinie als System\ 
    #von DGL erster Ordnung zurück; s ist Variable, k ist System \
    #erster Ordnung
    k0, k1 = k 
    #Ableitung der Elemente in k
    dk0_ds = k1
    dk1_ds = 1 / (A.s.E * A.s.I) * ( -F_z * np.cos(k0)) + 1 / (A.s.E * A.s.I) 
    return dk0_ds, dk1_ds #entspricht phi' und phi''
    
    
    
    #zweite AB für das AWP mit Hilfe von Newton Verfahren ermitteln

def findphidot0(phidot0, mf): # Anfangswert bestimmen mittels \
    #Newton-Raphson in Abh. von mf
    s = np.linspace(0, S.length, n) #Aktor diskretisieren mit \
        #n Stützstellen
    #nachfolgend wird das System der DGLn (f) numerisch im \
        #Integrationsintervall (0, S.length) mit den \
            #Anfangswerten (phi0, phidot0) integriert, an den\
                #stellen in s ausgewertet und die Lösung für \
                    #phi und phidot in sol.y geschrieben
    sol = solve_ivp(f, (0, S.length), (phi0, phidot0), \
                    t_eval = s) 
    #Werte für phi und phi' an den \ausgewerteten Stützstellen
    phi, phidot = sol.y 
    #E-Modul des Drahtes in Abhängigkeit des Martensitanteils
    E_sma = (A.w.EA * (1 - mf) + A.w.EM * mf) 
    M_sma = (sum((-phidot * t_H) / n) \
             - A.w.max_strain_zero_load * (mf - mf0)) \
        * E_sma * A_sma * t_H 
        #Biegemoment aufgrund der Drahtspannung berechnen
    #Rückgabe der Funktion, die Null werden soll
    return phidot[-1] - 1 / (A.s.E * A.s.I) * M_sma 

discretization = np.linspace(0, S.length, n)

for sequence in sequences:
    sequence_end = t + sequence[1]
    current = sequence[0]
    while t < sequence_end: 
        t += dt
        print(t)
        phi0 = 0
        phidot = np.zeros(n)  
        phidot0 = newton(findphidot0, 0, maxiter = 200, tol = 1e-7, args = (mf,))      
        sol = solve_ivp(f, (0, S.length), (phi0, phidot0), t_eval=discretization)
        phi, phidot = sol.y
        s_x = np.zeros(n)
        s_z = np.zeros(n)
        
        #Auslenkung in x- und z-Richtung berechnen
        for i in range(1, n):
            s_x[i] = h * np.cos(phi[i]) + s_x[i-1]
            s_z[i] = h * np.sin(phi[i]) + s_z[i-1]
        stress0 = -F_z * (s_x[-1]/2)/(t_H*A_sma)
        E,mf, strain, r, resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT, Ein, stress, eps_tr,state = sm(S, dt, stress0, mf0, strain0, L0, E_loss, current,stress, T, E, mf, strain, r, resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT, Ein,state, lam_coeff, mf_1, eps_tr)
        T += dT
        u0_d, u_d, u0_s, u_s, E_cond = ht.do_timestep(u0_d, u_d, u0_s, u_s, T0, T)
        if t >= n:
            u0_d_dct[str(t)] = u0_d
            u0_s_dct[str(t)] = u0_s
            n+=1
        T0 = T
        alpha = hc.alpha(T)
        E_conv = L0 * np.pi * S.diameter * (T-Tu) * dt * hc.alpha(T)
        E_loss = E_cond + E_conv
        
        Ein_sum += Ein
        Usum += U
        E_cond_sum += E_cond
        E_conv_sum += E_conv
        E_loss_sum += E_loss
       
        deflection = s_z[-1]
        xmax = s_x[-1]
        
        
        #collect data in list for plot
        try:
            data_array[idx] = [t, dt, stress0, mf0, strain0, L0, E_loss, current, stress, T, E, mf, strain, r, resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT, Ein, E_cond, E_conv, E_loss , E_cond_sum, E_conv_sum, E_loss_sum, Ein_sum, Usum, deflection, xmax, state, eps_tr, alpha]
        except:
            print('except')
        idx += 1
u0_d_dct[str(t)] = u0_d
u0_s_dct[str(t)] = u0_s


t1 = time.time()
data = pd.DataFrame(data_array, columns = column_names)
data = pysma.models.Python_models.diagnose_functions.compress_df(data)
params = ipt_dct
params['Conduction model'] = ht.name
params['Sequences'] = str(ipt_dct['Sequences'])
params['Computation time'] = t1-t0


pysma.evaluate.Test_file_creator(pysma.access.temperature_fields(),params, pd.DataFrame(u0_d))
pysma.evaluate.Test_file_creator(pysma.access.model_results(),params, data)


pysma.models.Python_models.diagnose_functions.diagnose_load(ipt_dct)

#Model(ipt_dct)