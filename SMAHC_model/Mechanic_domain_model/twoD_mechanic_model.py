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
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import newton
# ========================================================================================
# Mechanical domain class
# ========================================================================================
class Mechanical_domain:
    #Initialize 
    def __init__(self, SMAHC, nodes, load, mf0):
        self.S = SMAHC
        self.n = nodes
        self.dx = self.S.length / self.n
        self.load = load
        self.A_sma = self.S.g.n * self.S.w.crosssection_area
        self.elements = np.linspace(0, self.S.length, self.n)
        self.phidot = np.zeros(self.n) 
        self.phi0 = 0
        self.mf0 = mf0
    #Method to return derivatives phi' and phi of bendline as system of first order DEQ
    def f(self,s, k): 
        k0, k1 = k 
        phi_d = k1
        phi_dd = 1 / (self.S.s.E * self.S.s.I) * ( -self.load * np.cos(k0)) +\
            1 / (self.S.s.E * self.S.s.I) 
        return phi_d, phi_dd
    def findphi_d_0(self,phi_d_0, mf): 
        #The system of DEQ (f) is numerically integrated using the initial values phi(L=0)
        #and phi'(L=0) and evaluated for all elements with help of the solve_ivp 
        #(solve initial value problem) function.
        sol = solve_ivp(self.f, (0, self.S.length), (self.phi0, phi_d_0), \
                        t_eval = self.elements) 
        #sol.y returns the values for phi and phi' for all elements
        phi, phidot = sol.y 
        #Find E-modulus for sma wire E_sma for given martensite fraction mf
        E_sma = (self.S.w.EA * (1 - mf) + self.S.w.EM * mf) 
        #Bending moment as function of the SMAHC's curvature
        M_sma = (sum((-phidot * self.S.dist_sub_sma) / self.n) \
                 - self.S.w.max_strain_zero_load * (mf - self.mf0)) \
            * E_sma * self.A_sma * self.S.dist_sub_sma 
        #Returns the function from which the root is to be found. 
        return phidot[-1] - 1 / (self.S.s.E * self.S.s.I) * M_sma 
    def find_phi_phidot(self,mf):
        # Find value for phi'(L=0) depending on the martensite fraction using the
        # Newton-Raphson-method
        phi_d_0 = newton(self.findphi_d_0, 0, maxiter = 200, tol = 1e-7, args = (mf,)) 
        # Solve the initial value problem again with known phi'(L=0)
        sol = solve_ivp(self.f, (0, self.S.length), (self.phi0, phi_d_0),\
                        t_eval=self.elements)
        phi, phidot = sol.y
        #Returns phi and phi'
        return phi,phidot
    #Method for evaluating the stress, deflection and xmax
    def solve(self, mf):
        #Find the curvature for all elements
        phi, phidot = self.find_phi_phidot(mf)
        #Create empty lists for deflection (s_z) and regarding x coordinate (s_x)
        s_x = np.zeros(self.n)
        s_z = np.zeros(self.n)
        #Calculate position of element in x and z direction
        for i in range(1, self.n):
            s_x[i] = self.dx * np.cos(phi[i]) + s_x[i-1]
            s_z[i] = self.dx * np.sin(phi[i]) + s_z[i-1]
        #Calculate average stress in SMA 
        stress0 = -self.load * (s_x[-1]/2)/(self.S.dist_sub_sma*self.A_sma)
        #Find deflection at the end of active length
        deflection = s_z[-1]
        #Find xmax
        xmax = s_x[-1]
        return stress0, deflection, xmax