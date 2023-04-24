# -*- coding: utf-8 -*-
"""
@author: Max kaiser
@mail: max.kaiser@ivw.uni-kl.de
@facility: Leibniz Institut für Verbundwerkstoffe GmbH
@license: MIT License
@copyright: Copyright (c) 2023 Leibniz Institut für Verbundwerkstoffe GmbH
@Version: 0.0.
"""
# ========================================================================================
# Load modules
# ========================================================================================
from scipy.optimize import fsolve
import numpy as np
# ========================================================================================
# System of Equations to be solved for transformation from martensite to austenite
# ========================================================================================
def heat_MtoA(vars, *args):
    dt, stress0, mf0, strain0, L0, Eloss, current, T, w, lam_coeff, mf_1,mfm1 = args
    E, mf, strain, r, resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT, Ein, \
        stress,eps_tr = vars
    #Constitutive Model
    eq14 = w.EM * mf + w.EA * (1 - mf)-E
    eq11 = E * (strain - eps_tr) - E * eps_tr * (mf - mf0) - (stress)
    #Resistance Model
    eq3 = w.rM * mf + w.rA * (1 - mf)-r
    eq4 = r * L0 / (np.pi*((w.diameter/2)**2))-resistance
    #Phase fraction Model
    eq5 = w.As - real_As + (stress) / w.CAs
    eq6 = w.Af - real_Af + (stress) / w.CAf
    eq7 = w.Mf - real_Mf + (stress) / w.CMf
    eq8 = w.Ms - real_Ms + (stress) / w.CMs
    #Heat capacitance model
    eq9 = w.dH_M_to_A * (np.pi / (2 * (w.Af-w.As+stress*(1/w.CAf-1/w.CAs)))) * \
        np.sin(np.pi * (T - w.As-stress/w.CAs) / (w.Af-w.As+stress*(1/w.CAf-1/w.CAs)))-a
    eq10 = L0 * np.pi * (w.diameter/2)**2 * w.rho * (((w.cA+w.cM)/2) + a) * dT - U
    eq1 = mf_1*0.5 * (np.cos(np.pi  * (T - w.As-stress/w.CAs) / \
                             (w.Af-w.As+stress*(1/w.CAf-1/w.CAs))) + 1) - mf
    #Joule heat Model
    eq12 = resistance *current *current * dt - Ein
    eq13 = U + Eloss - Ein
    eq2 = lam_coeff * (eps_tr - strain) + (stress - stress0)
    eq15 = w.max_strain_zero_load + (w.max_trans_strain - w.max_strain_zero_load)*\
        (1-np.exp(-w.k*(stress)/w.EA)) - eps_tr
    return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14,eq15]
# ========================================================================================
# System of Equations to be solved for transformation from austenite to martensite
# ========================================================================================
def cool_AtoM(vars, *args):
    dt, stress0, mf0, strain0, L0, Eloss, current, T, w, lam_coeff, mf_1,mfm1 = args
    E,mf, strain, r, resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT, Ein, \
        stress,eps_tr = vars
    #Constitutive Model
    eq1 = w.EM * mf + w.EA * (1 - mf)-E
    eq2 = E * (strain - eps_tr) - E * eps_tr * (mf - mf0) - (stress)
    #Resistance Model
    eq3 = w.rM * mf + w.rA * (1 - mf)-r
    eq4 = r * L0 / (np.pi*((w.diameter/2)**2))-resistance
    #Phase fraction Model
    eq5 = w.As - real_As + (stress) / w.CAs
    eq6 = w.Af - real_Af + (stress) / w.CAf
    eq7 = w.Mf - real_Mf + (stress) / w.CMf
    eq8 = w.Ms - real_Ms + (stress) / w.CMs
    #Heat capacitance model
    eq9 = w.dH_R_to_M * (np.pi / (2 * (real_Ms - real_Mf))) *\
        np.sin(np.pi * (T - real_Mf) / (real_Ms - real_Mf)) - a
    eq10 = L0 * np.pi * (w.diameter/2)**2 * w.rho * (((w.cA+w.cM)/2) + a) * dT - U
    eq11 = 0.5 * (np.cos(np.pi / (real_Ms - real_Mf) * (T - real_Mf)) + 1) - mf
    #Joule heat Model
    eq12 = resistance *current *current * dt - Ein
    eq13 = U + Eloss - Ein
    eq14 = lam_coeff * (eps_tr - strain) + (stress - stress0)
    eq15 = w.max_strain_zero_load + (w.max_trans_strain - w.max_strain_zero_load)*\
        (1-np.exp(-w.k*(stress)/w.EA)) - eps_tr
    return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14,eq15]
# ========================================================================================
# System of Equations to be solved when martensite fraction is constant
# or no transformation occurs
# ========================================================================================
def const_mf(vars, *args):
    dt, stress0, mf0, strain0, L0, Eloss, current, T, w, lam_coeff, mf_1,mfm1 = args
    E,mf, strain, r, resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT, Ein, \
        stress,eps_tr = vars
    #Constitutive Model
    eq1 = w.EM * mfm1 + w.EA * (1 - mfm1)-E
    eq2 = E * (strain - eps_tr) - E * eps_tr * (mfm1 - mf0) - (stress)
    #Resistance Model
    eq3 = w.rM * mfm1 + w.rA * (1 - mfm1)-r
    eq4 = r * L0 / (np.pi*((w.diameter/2)**2))-resistance
    #Phase fraction Model
    eq5 = w.As - real_As + (stress) / w.CAs
    eq6 = w.Af - real_Af + (stress) / w.CAf
    eq7 = w.Mf - real_Mf + (stress) / w.CMf
    eq8 = w.Ms - real_Ms + (stress) / w.CMs
    #Heat capacitance model
    eq9 = 0 - a
    eq10 = L0 * np.pi * (w.diameter/2)**2 * w.rho * (((w.cA+w.cM)/2)) * dT - U
    eq11 = mf - mfm1
    #Joule heat Model
    eq12 = resistance *current *current * dt - Ein
    eq13 = U + Eloss - Ein
    eq14 = lam_coeff * (eps_tr - strain) + (stress - stress0)
    eq15 = w.max_strain_zero_load + (w.max_trans_strain - w.max_strain_zero_load)*\
        (1-np.exp(-w.k*(stress)/w.EA)) - eps_tr
    return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14,eq15]
# ========================================================================================
# Decission tree for distinguishing of transformation state
# ========================================================================================
def sma_model(w, dt, stress0, mf0, strain0, L0, Eloss, current,stress, T, E, mf, strain, r,\
              resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT, Ein,\
                  lam_coeff, mf_1, eps_tr):
    mfm1 = mf
    all_args = (dt, stress0, mf0, strain0, L0, Eloss, current, T, w, lam_coeff, mf_1, mfm1) 
    ipt = (E, mf, strain, r, resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT,\
           Ein, stress, eps_tr)
    opt = (E, mf, strain, r, resistance, real_As, real_Af, real_Mf, real_Ms, a, U, dT,\
           Ein, stress, eps_tr)
    if T > real_Af:
        mfm1 = 0
        all_args = (dt, stress0, mf0, strain0, L0, Eloss, current, T, w, lam_coeff, mf_1,\
                    mfm1)
        opt = fsolve(const_mf, ipt, args = all_args, xtol = 1e-10)
    elif T < real_Mf:
        mfm1 = 1
        all_args = (dt, stress0, mf0, strain0, L0, Eloss, current, T, w, lam_coeff, mf_1,\
                    mfm1)
        opt = fsolve(const_mf, ipt, args = all_args, xtol = 1e-10)
    else:
        if U > 0:
            if mf <= 0:
                mfm1 = 0
                all_args = (dt, stress0, mf0, strain0, L0, Eloss, current, T, w, lam_coeff,\
                            mf_1, mfm1)
                opt = fsolve(const_mf, ipt, args = all_args, xtol = 1e-10)
            else:
                if T <= real_As:
                    opt = fsolve(const_mf, ipt, args = all_args, xtol = 1e-10)
                else:
                    opt = fsolve(heat_MtoA, (E, mf*0.99, strain, r, resistance, real_As,\
                                             real_Af, real_Mf, real_Ms, a, U, dT, Ein, \
                                                 stress, eps_tr), \
                                 args = all_args, xtol = 1e-10)
        else:
            if mf >= 1:
                mfm1 = 1
                all_args = (dt, stress0, mf0, strain0, L0, Eloss, current, T, w, lam_coeff,\
                            mf_1, mfm1)
                opt = fsolve(const_mf, ipt, args = all_args, xtol = 1e-10)
            else:
                if T >= real_Ms:
                    opt = fsolve(const_mf, ipt, args = all_args, xtol = 1e-10)
                else:
                    opt = fsolve(cool_AtoM, (E, mf*1.01, strain, r, resistance, real_As,\
                                             real_Af, real_Mf, real_Ms, a, U, dT, Ein,\
                                                 stress, eps_tr),\
                                 args = all_args, xtol = 1e-10)
    return opt