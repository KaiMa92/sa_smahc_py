# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 16:40:32 2021

@author: kaiser
"""

from scipy.optimize import fsolve
import numpy as np

def heat_MtoA(vars, *args):
    dt, stress0, mf0, strain0, L0, Eloss, i, T, s, lam_coeff, mf_1,mfm1 = args
    
    E, mf, strain, r, resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT, Ein, stress,eps_tr = vars
    
    eq2 = lam_coeff * (eps_tr - strain) + (stress - stress0)
    eq1 = mf_1*0.5 * (np.cos(np.pi  * (T - s.As-stress/s.CAs) / (s.Af-s.As+stress*(1/s.CAf-1/s.CAs))) + 1) - mf
    #Constitutive Model
    eq14 = s.EM * mf + s.EA * (1 - mf)-E
    eq11 = E * (strain - eps_tr) - E * eps_tr * (mf - mf0) - (stress)
    
    #Resistance Model
    eq3 = s.rM * mf + s.rA * (1 - mf)-r
    eq4 = r * L0 / (np.pi*((s.diameter/2)**2))-resistance
    
    #Phase fraction Model
    eq5 = s.As - real_As + (stress) / s.CAs
    eq6 = s.Af - real_Af + (stress) / s.CAf
    eq7 = s.Mf - real_Mf + (stress) / s.CMf
    eq8 = s.Ms - real_Ms + (stress) / s.CMs
    
    eq9 = s.dH_M_to_A * (np.pi / (2 * (s.Af-s.As+stress*(1/s.CAf-1/s.CAs)))) * np.sin(np.pi * (T - s.As-stress/s.CAs) / (s.Af-s.As+stress*(1/s.CAf-1/s.CAs)))-a
    #eq9 = s.dH_M_to_A * (np.pi / (2 * (real_Af - real_As))) * np.sin(np.pi * (T - real_As) / (real_Af - real_As)) - a
    eq10 = L0 * np.pi * (s.diameter/2)**2 * s.rho * (((s.cA+s.cM)/2) + a) * dT - U
    
    #eq11 = mf_1*0.5 * (np.cos(np.pi  * (T - real_As) / (real_Af - real_As)) + 1) - mf
    #Joule heat Model
    eq12 = resistance * i * i * dt - Ein
    eq13 = U + Eloss - Ein
    
    eq15 = s.max_strain_zero_load + (s.max_trans_strain - s.max_strain_zero_load)*(1-np.exp(-s.k*(stress)/s.EA)) - eps_tr
    return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14,eq15]


def cool_AtoM(vars, *args):
    dt, stress0, mf0, strain0, L0, Eloss, i, T, s, lam_coeff, mf_1,mfm1 = args
    
    E,mf, strain, r, resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT, Ein, stress,eps_tr = vars
    #Constitutive Model
    eq1 = s.EM * mf + s.EA * (1 - mf)-E
    eq2 = E * (strain - eps_tr) - E * eps_tr * (mf - mf0) - (stress)
    
    #Resistance Model
    eq3 = s.rM * mf + s.rA * (1 - mf)-r
    eq4 = r * L0 / (np.pi*((s.diameter/2)**2))-resistance
    
    #Phase fraction Model
    eq5 = s.As - real_As + (stress) / s.CAs
    eq6 = s.Af - real_Af + (stress) / s.CAf
    eq7 = s.Mf - real_Mf + (stress) / s.CMf
    eq8 = s.Ms - real_Ms + (stress) / s.CMs
    
    eq9 = s.dH_R_to_M * (np.pi / (2 * (real_Ms - real_Mf))) * np.sin(np.pi * (T - real_Mf) / (real_Ms - real_Mf)) - a
    eq10 = L0 * np.pi * (s.diameter/2)**2 * s.rho * (((s.cA+s.cM)/2) + a) * dT - U
    eq11 = 0.5 * (np.cos(np.pi / (real_Ms - real_Mf) * (T - real_Mf)) + 1) - mf
    
    #Joule heat Model
    eq12 = resistance * i * i * dt - Ein
    eq13 = U + Eloss - Ein
    eq14 = lam_coeff * (eps_tr - strain) + (stress - stress0)
    eq15 = s.max_strain_zero_load + (s.max_trans_strain - s.max_strain_zero_load)*(1-np.exp(-s.k*(stress)/s.EA)) - eps_tr
    return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14,eq15]


def const_mf(vars, *args):
    dt, stress0, mf0, strain0, L0, Eloss, i, T, s, lam_coeff, mf_1,mfm1 = args
    
    E,mf, strain, r, resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT, Ein, stress,eps_tr = vars
    #Constitutive Model
    eq1 = s.EM * mfm1 + s.EA * (1 - mfm1)-E
    eq2 = E * (strain - eps_tr) - E * eps_tr * (mfm1 - mf0) - (stress)
    
    #Resistance Model
    eq3 = s.rM * mfm1 + s.rA * (1 - mfm1)-r
    eq4 = r * L0 / (np.pi*((s.diameter/2)**2))-resistance
    
    #Phase fraction Model
    eq5 = s.As - real_As + (stress) / s.CAs
    eq6 = s.Af - real_Af + (stress) / s.CAf
    eq7 = s.Mf - real_Mf + (stress) / s.CMf
    eq8 = s.Ms - real_Ms + (stress) / s.CMs
    
    eq9 = 0 - a
    eq10 = L0 * np.pi * (s.diameter/2)**2 * s.rho * (((s.cA+s.cM)/2)) * dT - U
    eq11 = mf - mfm1
    
    #Joule heat Model
    eq12 = resistance * i * i * dt - Ein
    eq13 = U + Eloss - Ein
    eq14 = lam_coeff * (eps_tr - strain) + (stress - stress0)
    eq15 = s.max_strain_zero_load + (s.max_trans_strain - s.max_strain_zero_load)*(1-np.exp(-s.k*(stress)/s.EA)) - eps_tr
    return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14,eq15]


def sma_model(s, dt, stress0, mf0, strain0, L0, Eloss, i,stress, T, E, mf, strain, r, resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT, Ein,state, lam_coeff, mf_1, eps_tr):
    mfm1 = mf
    all_args = (dt, stress0, mf0, strain0, L0, Eloss, i, T, s, lam_coeff, mf_1, mfm1) 
    ipt = (E, mf, strain, r, resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT, Ein, stress, eps_tr)
    opt = (E, mf, strain, r, resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT, Ein, stress, eps_tr)
    
    if T > real_Af:
        mfm1 = 0
        all_args = (dt, stress0, mf0, strain0, L0, Eloss, i, T, s, lam_coeff, mf_1, mfm1)
        opt = fsolve(const_mf, ipt, args = all_args, xtol = 1e-10)
        state = 0.4
    
    elif T < real_Mf:
        mfm1 = 1
        all_args = (dt, stress0, mf0, strain0, L0, Eloss, i, T, s, lam_coeff, mf_1, mfm1)
        opt = fsolve(const_mf, ipt, args = all_args, xtol = 1e-10)
        state = 0
        
    else:
    
        if U > 0:
            if mf <= 0:
                mfm1 = 0
                all_args = (dt, stress0, mf0, strain0, L0, Eloss, i, T, s, lam_coeff, mf_1, mfm1)
                opt = fsolve(const_mf, ipt, args = all_args, xtol = 1e-10)
                state = 0.5
            else:
                if T <= real_As:
                    opt = fsolve(const_mf, ipt, args = all_args, xtol = 1e-10)
                    state = 0.1
                else:
                    opt = fsolve(heat_MtoA, (E, mf*0.99, strain, r, resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT, Ein, stress, eps_tr), args = all_args, xtol = 1e-10)
                    state = 1
        else:
            if mf >= 1:
                mfm1 = 1
                all_args = (dt, stress0, mf0, strain0, L0, Eloss, i, T, s, lam_coeff, mf_1, mfm1)
                opt = fsolve(const_mf, ipt, args = all_args, xtol = 1e-10)
                state = 0.2
            else:
                if T >= real_Ms:
                    opt = fsolve(const_mf, ipt, args = all_args, xtol = 1e-10)
                    state = 0.3
                else:
                    opt = fsolve(cool_AtoM, (E, mf*1.01, strain, r, resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT, Ein, stress, eps_tr), args = all_args, xtol = 1e-10)
                    state = 2

    opt = np.append(opt,state)
    #print('OPT:\n',opt)  
    return opt