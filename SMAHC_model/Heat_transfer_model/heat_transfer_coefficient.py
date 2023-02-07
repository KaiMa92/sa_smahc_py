# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:10:21 2022

@author: kaiser
"""
import numpy as np
import pandas as pd
import os
from project_root import get_project_root
root = get_project_root()

class Heat_transfer_coefficient:
    def __init__(self, ambient_fluid):
        self.air_params_file = root + os.sep + 'Material_library' + os.sep + 'ambient_fluid' + os.sep + ambient_fluid + os.sep + 'characteristics.txt'
        self.air_params = pd.read_csv(self.air_params_file, sep='\ ', decimal = ',', header = 0, skiprows = [1])     
        self.beta_pol = np.poly1d(np.polyfit(self.air_params['T'], self.air_params['beta']*1e-3, deg = 16))
        self.lam_pol = np.poly1d(np.polyfit(self.air_params['T'], self.air_params['lam']*1e-3, deg = 16))
        self.Pr_pol = np.poly1d(np.polyfit(self.air_params['T'], self.air_params['Pr'], deg = 16))
        self.nu_pol = np.poly1d(np.polyfit(self.air_params['T'], self.air_params['nu']*1e-7, deg = 16))
        pass
     
    def Tm(self, T):
        return (T-self.T_inf)/2
        
    def beta(self,T):
        return self.beta_pol(self.Tm(T))
    
    def lam(self,T):
        return self.lam_pol(self.Tm(T))
    
    def Pr(self,T):
        return self.Pr_pol(self.Tm(T))
    
    def nu(self,T):
        return self.nu_pol(self.Tm(T))
    
    def alpha(self, T, phi_deg = 'nan'):
        if phi_deg != 'nan':
            return self.Nu(T,phi_deg)*self.lam(T)/self.L
        else:
            return self.Nu(T)*self.lam(T)/self.L
    
    def Ra(self,T):
        return self.beta(T)*9.81*np.abs(T-self.T_inf)*self.L**3*self.Pr(T)/self.nu(T)**2
    
    def Gr(self, T):
        return 9.81*np.power(self.L,3)*self.beta(T)*(T - self.T_inf)/np.power(self.nu(T),2)
    

class horizontal_cylinder(Heat_transfer_coefficient):
    def __init__(self, d, T_inf, ambient_fluid):
        Heat_transfer_coefficient.__init__(self, ambient_fluid)
        self.T_inf = T_inf
        self.L = 0.5 * np.pi * d
        self.Ra_krit_up = 10**12
        self.Ra_krit_low = 0
        
    def Nu(self, T):
        if self.Ra(T) < self.Ra_krit_up and self.Ra(T) > self.Ra_krit_low:
            #Nu = (0.6 + ((0.387*self.Ra(T)**(1/6))/((1+(0.599/self.Pr(T))**(9/16))**(8/27))))**2
            #Nu = (0.60 + 0.387*np.power(self.Ra(T),1/6)/np.power(1 + np.power(0.559/self.Pr(Tm),9/16),8/27))**2
            Nu = (0.752 + 0.387*(self.Ra(T)*(1+(0.559/self.Pr(T))**(9/16))**(-16/9))**(1/6))**2
        else:
            #print('Ra not in range')
            Nu = (0.752 + 0.387*(self.Ra(T)*(1+(0.559/self.Pr(T))**(9/16))**(-16/9))**(1/6))**2
        return Nu
    

class vertical_cylinder(Heat_transfer_coefficient):
    def __init__(self, d, h, T_inf, ambient_fluid):
        Heat_transfer_coefficient.__init__(self, ambient_fluid)
        self.L = h
        self.T_inf = T_inf
        self.Nu_coef = 0.435*h/d
        
    def Nu(self, T):
        Nu = (0.825 + 0.387*(self.Ra(T)*(1+(0.492/self.Pr(T))**(9/16))**(-16/9))**(1/6))**2 + self.Nu_coef
        return Nu
        
    
class inclined_cylinder(Heat_transfer_coefficient):
    def __init__(self, d, T_inf, ambient_fluid):
        Heat_transfer_coefficient.__init__(self, ambient_fluid)
        self.T_inf = T_inf
        self.L = 0.5 * np.pi * d
        self.Ra_krit_up = 6*10**10
        self.Ra_krit_low = 2.6 * 10 ** -8 
        
    def Nu(self, phi_deg, T): #phi in deg from vertical
        if self.Ra(T) < self.Ra_krit_up and self.Ra(T) > self.Ra_krit_low:
            if phi_deg < 90:
                phi_rad = phi_deg * np.pi/180
                Nu = (-0.04*phi_rad+0.19)+(-0.09*phi_rad**2 + 0.003 * phi_rad + 0.82) * self.Ra(T)**(-0.03*phi_rad+0.16)
            else:
                print('Phi cannot be greater 90 deg')
                Nu = np.nan
        else:
            print('Ra not in range')
            Nu = np.nan
        return Nu
    
    
class horizontal_plate_ehu(Heat_transfer_coefficient): # emitting heat on upper side
    def __init__(self, a,b, T_inf, ambient_fluid):
        Heat_transfer_coefficient.__init__(self, ambient_fluid)
        self.L = a*b/(2*(a+b))
        self.T_inf = T_inf
        
    def Nu(self, T): #phi in deg from vertical
        if self.Ra(T)*(1+(0.322/self.Pr(T))**(11/20))**(-20/11) <= 7*10**4:
            Nu = 0.766*(self.Ra(T)*(1+(0.322/self.Pr(T))**(11/20))**(-20/11))**(1/5)
        else:
            Nu = 0.15*(self.Ra(T)*(1+(0.322/self.Pr(T))**(11/20))**(-20/11))**(1/3)
        return Nu


class horizontal_plate_ehl(Heat_transfer_coefficient): # emitting heat on lower side
    def __init__(self, a,b, T_inf, ambient_fluid):
        Heat_transfer_coefficient.__init__(self, ambient_fluid)
        self.L = a*b/(2*(a+b))
        self.T_inf = T_inf
        
    def Nu(self, T): #phi in deg from vertical
        if self.Ra(T)*(1+(0.492/self.Pr(T))**(9/16))**(-16/9) < 10**10 and self.Ra(T)*(1+(0.492/self.Pr(T))**(9/16))**(-16/9) > 10**3:
            Nu = 0.6*(self.Ra(T)*(1+(0.492/self.Pr(T))**(9/16))**(-16/9))**(1/5)
        else:
            print('Ra not in range')
            Nu = np.nan
        return Nu


class horizontal_plate_ehu_inclined(Heat_transfer_coefficient): # emitting heat on upper side
    def __init__(self, a,b, T_inf, ambient_fluid):
        Heat_transfer_coefficient.__init__(self, ambient_fluid)
        self.L = a*b/(2*(a+b))
        self.T_inf = T_inf
        
    def Nu(self, T, phi_deg): #phi in deg from vertical
        Rac = 10**(7.08-0.00178*phi_deg) 
        Nu = 0.56*(Rac*np.cos(phi_deg*np.pi/180))**(1/4)+0.13*(self.Ra(T)**(1/3)-Rac**(1/3))
        return Nu


class horizontal_plate_ehl_inclined(Heat_transfer_coefficient): # emitting heat on lower side
    def __init__(self, a,b, T_inf, ambient_fluid):
        Heat_transfer_coefficient.__init__(self, ambient_fluid)
        self.L = a*b/(2*(a+b))
        self.T_inf = T_inf
        self.Ra_krit_up = 10**12
        self.Ra_krit_low = 10**-1 
        
    def Nu(self, T, phi_deg): #phi in deg from vertical
        if self.Ra(T) < self.Ra_krit_up and self.Ra(T) > self.Ra_krit_low:
            Nu = (0.825+0.387*(self.Ra(T)*np.cos(phi_deg)*(1+(0.492/self.Pr(T))**(9/16))**(-16/9)))**(2)
        else:
            print('Ra not in range')
            Nu = np.nan
        return Nu