# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 17:21:57 2022

@author: kaiser
"""


# =============================================================================
# Load modules
# =============================================================================
import json
import numpy as np
import pandas as pd
import os

#Own modules
from utils.file_creator import file_creator
from utils.material import SMAHC
from SMAHC_model.SMA_model.lumped_sma_model import sma_model as sm
from SMAHC_model.Heat_transfer_model.twoD_elastomer import Heat_transfer
from SMAHC_model.Heat_transfer_model.heat_transfer_coefficient import horizontal_cylinder
from SMAHC_model.Mechanic_domain_model.twoD_mechanic_model import Mechanic_domain
from project_root import get_project_root


# =============================================================================
# Initialize
# =============================================================================

#Path of the local repository
root = get_project_root()

#Read input data from INPUT.txt and load parameters
with open(root+os.sep+"INPUT.txt") as input_data_file:
    ipt_dct = json.load(input_data_file)
load = ipt_dct['Load']
alpha_elastomer = ipt_dct['Alpha elastomer'] 
Tu = ipt_dct['Ambient temperature']
sequences = ipt_dct['Sequences']
dt = ipt_dct['Time increment']
dx = ipt_dct['Spatial increment']
mf0 = ipt_dct['mf0']
stress0 = ipt_dct['stress0']
data_resolution = ipt_dct['data_resolution']
A = SMAHC(ipt_dct['Actuator type'])

#Initialize state variables
L0 = A.w.length
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
E = A.w.EM
mf = mf0
mf_1 = mf0
r = A.w.rM
resistance = r * L0 / (np.pi*((A.w.diameter/2)**2))
real_As = A.w.As + stress0/A.w.CAs
real_Af = A.w.Af + stress0/A.w.CAf
real_Mf = A.w.Mf + stress0/A.w.CMf
real_Ms = A.w.Ms + stress0/A.w.CMs
eps_tr = A.w.max_strain_zero_load + (A.w.max_trans_strain - A.w.max_strain_zero_load)*(1-np.exp(-A.w.k*(stress0)/A.w.EA))
strain0 = eps_tr
strain = strain0
R = np.inf
deflection = 0
xmax = A.length
t = 0

#Initialize mechanic domain model
md = Mechanic_domain(A, 2000, load, mf0)

#Initialize conductive heat transfer model
ht = Heat_transfer(A, dx, Tu, alpha_elastomer, dt)
u0_d = ht.u0_d
u0_s = ht.u0_s
u_d = ht.u_d
u_s = ht.u_s
u0_d_dct = {}
u0_s_dct = {}

#Initialize convective heat transfer coefficient
hc = horizontal_cylinder(A.w.diameter, Tu, 'dry_air')

#Create array for result data storage
column_names = ['t','dt', 'stress0', 'mf0', 'strain0', 'L0', 'E_loss', 'current', 'stress', 'T', 'E', 'mf', 'strain', 'r', 'resistance', 'real_As', 'real_Af', 'real_Mf', 'real_Ms', 'a', 'U', 'dT', 'Ein', 'E_cond', 'E_conv', 'E_loss' , 'E_cond_sum', 'E_conv_sum', 'E_loss_sum', 'Ein_sum', 'Usum', 'deflection', 'xmax', 'state', 'eps_tr', 'alpha']
t_tot = sum(sequence[1] for sequence in sequences)
data_array = np.ones((int((t_tot-t)/dt/data_resolution),len(column_names)))

#Initialize counters
data_resolution_counter = 0
data_array_idx = 0


# =============================================================================
# Simulate
# =============================================================================

#Work through all sequences
for sequence in sequences:
    sequence_end = t + sequence[1]
    current = sequence[0]
    #Simulate the sequence for the respective time period t
    while t < sequence_end: 
        #Increment the time
        t += dt
        #Print current total simulated time
        print(t)
        
        #Solve mechanic domain model
        stress0, deflection, xmax = md.solve(mf)
        
        #Solve SMA model
        E,mf, strain, r, resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT, Ein, stress, eps_tr,state = sm(A.w, dt, stress0, mf0, strain0, L0, E_loss, current,stress, T, E, mf, strain, r, resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT, Ein,state, -A.stiffness, mf_1, eps_tr)
        
        #Increment temperature
        T += dT
        
        #Solve conductive heat transfer model
        u0_d, u_d, u0_s, u_s, E_cond = ht.do_timestep(u0_d, u_d, u0_s, u_s, T0, T)
        T0 = T
        
        #Solve convective heat transfer
        alpha = hc.alpha(T)
        E_conv = L0 * np.pi * A.w.diameter * (T-Tu) * dt * hc.alpha(T)
        
        #Sum up the heat losses
        E_loss = E_cond + E_conv
        
        #Cumulating the energy components
        Ein_sum += Ein
        Usum += U
        E_cond_sum += E_cond
        E_conv_sum += E_conv
        E_loss_sum += E_loss
        
        #Write data to array
        if data_resolution_counter < data_resolution: 
            data_resolution_counter += 1
        else:
            data_resolution_counter= 0
            data_array[data_array_idx] = [t, dt, stress0, mf0, strain0, L0, E_loss, current, stress, T, E, mf, strain, r, resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT, Ein, E_cond, E_conv, E_loss , E_cond_sum, E_conv_sum, E_loss_sum, Ein_sum, Usum, deflection, xmax, state, eps_tr, alpha]
            data_array_idx += 1


# =============================================================================
# Safe data
# =============================================================================

#Convert array to pandas dataframe
data = pd.DataFrame(data_array, columns = column_names)[:data_array_idx]

#write dataframe and ipt_dct to file in Data_output directory
file_creator(root + os.sep + "OUTPUT" + os.sep + "Temperature_field", ipt_dct, pd.DataFrame(u0_d))
#write last temperature field and ipt_dct to file in Temperature_field directory
file_creator(root + os.sep + "OUTPUT" + os.sep + "Data_output",ipt_dct, data)

