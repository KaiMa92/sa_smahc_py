# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 17:21:57 2022

@author: kaiser
"""


import json
import numpy as np
import pandas as pd

import os

from utils.file_creator import file_creator
from utils.material import SMAHC
from SMAHC_model.SMA_model.lumped_sma_model import sma_model as sm
from SMAHC_model.Heat_transfer_model.twoD_elastomer import Heat_transfer
from SMAHC_model.Heat_transfer_model.heat_transfer_coefficient import horizontal_cylinder
from SMAHC_model.Mechanic_domain_model.twoD_mechanic_model import Mechanic_domain
from project_root import get_project_root
root = get_project_root()

with open(root+os.sep+"INPUT.txt") as input_data_file:
    ipt_dct = json.load(input_data_file)
    
    

F_z = - ipt_dct['Load']
alpha_elastomer = ipt_dct['Alpha elastomer'] 
Tu = ipt_dct['Ambient temperature']
sequences = ipt_dct['Sequences']
dt = ipt_dct['Time increment']
dx = ipt_dct['Spatial increment']
mf0 = ipt_dct['mf0']
stress0 = ipt_dct['stress0']
data_resolution = ipt_dct['data_resolution']

A = SMAHC(ipt_dct['Actuator type'])
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

curr_data_point = 0


hc = horizontal_cylinder(S.diameter, Tu)


t_tot = sum(sequence[1] for sequence in sequences)
t = 0


column_names = ['t','dt', 'stress0', 'mf0', 'strain0', 'L0', 'E_loss', 'current', 'stress', 'T', 'E', 'mf', 'strain', 'r', 'resistance', 'real_As', 'real_Af', 'real_Mf', 'real_Ms', 'a', 'U', 'dT', 'Ein', 'E_cond', 'E_conv', 'E_loss' , 'E_cond_sum', 'E_conv_sum', 'E_loss_sum', 'Ein_sum', 'Usum', 'deflection', 'xmax', 'state', 'eps_tr', 'alpha']
data_array = np.ones((int((t_tot-t)/dt/data_resolution),len(column_names)))
idx = 0


ht = Heat_transfer(A, dx, Tu, alpha_elastomer, dt)


u0_d = ht.u0_d
u0_s = ht.u0_s
u_d = ht.u_d
u_s = ht.u_s

u0_d_dct = {}
u0_s_dct = {}
n = 0

md = Mechanic_domain(A, 2000, F_z, mf0)




for sequence in sequences:
    sequence_end = t + sequence[1]
    current = sequence[0]
    while t < sequence_end: 
        t += dt
        print(t)
        stress0, deflection, xmax = md.solve(mf)
        
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
        
        
        #collect data in list for plot
        try:
            if curr_data_point < data_resolution: 
                curr_data_point += 1
            else:
                curr_data_point = 0
                data_array[idx] = [t, dt, stress0, mf0, strain0, L0, E_loss, current, stress, T, E, mf, strain, r, resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT, Ein, E_cond, E_conv, E_loss , E_cond_sum, E_conv_sum, E_loss_sum, Ein_sum, Usum, deflection, xmax, state, eps_tr, alpha]
        except:
            print('except')
        idx += 1
u0_d_dct[str(t)] = u0_d
u0_s_dct[str(t)] = u0_s



data = pd.DataFrame(data_array, columns = column_names)
params = ipt_dct
params['Conduction model'] = ht.name
params['Sequences'] = str(ipt_dct['Sequences'])



file_creator(root + os.sep + "OUTPUT" + os.sep + "Data_output" , params, pd.DataFrame(u0_d))
file_creator(root + os.sep + "OUTPUT" + os.sep + "Temperature_field",params, data)

