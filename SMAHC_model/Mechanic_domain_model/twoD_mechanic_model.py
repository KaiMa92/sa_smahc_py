# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:45:56 2023

@author: kaiser
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import newton


class Mechanic_domain:
    def __init__(self, SMAHC, nodes, load, mf0):
        self.S = SMAHC
        self.n = nodes
        self.dx = self.S.length / self.n
        self.load = load
        self.A_sma = self.S.g.n * self.S.w.crosssection_area
        self.discretization = np.linspace(0, self.S.length, self.n)
        self.phidot = np.zeros(self.n) 
        self.phi0 = 0
        self.mf0 = mf0
        

    def f(self,s, k): #Gibt Ableitung der DGL der Biegelinie als System\ 
        #von DGL erster Ordnung zurück; s ist Variable, k ist System \
        #erster Ordnung
        k0, k1 = k 
        #Ableitung der Elemente in k
        dk0_ds = k1
        dk1_ds = 1 / (self.S.s.E * self.S.s.I) * ( self.load * np.cos(k0)) + 1 / (self.S.s.E * self.S.s.I) 
        return dk0_ds, dk1_ds #entspricht phi' und phi''
    
    
    
    #zweite AB für das AWP mit Hilfe von Newton Verfahren ermitteln

    def findphidot0(self,phidot0, mf): # Anfangswert bestimmen mittels \
        #Newton-Raphson in Abh. von mf
        s = np.linspace(0, self.S.length, self.n) #Aktor diskretisieren mit \
            #n Stützstellen
        #nachfolgend wird das System der DGLn (f) numerisch im \
            #Integrationsintervall (0, S.length) mit den \
                #Anfangswerten (phi0, phidot0) integriert, an den\
                    #stellen in s ausgewertet und die Lösung für \
                        #phi und phidot in sol.y geschrieben
        sol = solve_ivp(self.f, (0, self.S.length), (self.phi0, phidot0), \
                        t_eval = s) 
        #Werte für phi und phi' an den \ausgewerteten Stützstellen
        phi, phidot = sol.y 
        #E-Modul des Drahtes in Abhängigkeit des Martensitanteils
        E_sma = (self.S.w.EA * (1 - mf) + self.S.w.EM * mf) 
        M_sma = (sum((-phidot * self.S.dist_sub_sma) / self.n) \
                 - self.S.w.max_strain_zero_load * (mf - self.mf0)) \
            * E_sma * self.A_sma * self.S.dist_sub_sma 
            #Biegemoment aufgrund der Drahtspannung berechnen
        #Rückgabe der Funktion, die Null werden soll
        return phidot[-1] - 1 / (self.S.s.E * self.S.s.I) * M_sma 
    
    def find_phi_phidot(self,mf):
        phidot0 = newton(self.findphidot0, 0, maxiter = 200, tol = 1e-7, args = (mf,))      
        sol = solve_ivp(self.f, (0, self.S.length), (self.phi0, phidot0), t_eval=self.discretization)
        phi, phidot = sol.y
        return phi,phidot

    def solve(self, mf):
        phi, phidot = self.find_phi_phidot(mf)
        s_x = np.zeros(self.n)
        s_z = np.zeros(self.n)
        #Auslenkung in x- und z-Richtung berechnen
        for i in range(1, self.n):
            s_x[i] = self.dx * np.cos(phi[i]) + s_x[i-1]
            s_z[i] = self.dx * np.sin(phi[i]) + s_z[i-1]
            
        stress0 = -self.load * (s_x[-1]/2)/(self.S.dist_sub_sma*self.A_sma)
        deflection = s_z[-1]
        xmax = s_x[-1]
        return stress0, deflection, xmax