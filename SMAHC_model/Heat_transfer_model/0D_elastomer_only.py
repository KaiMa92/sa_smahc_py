# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:11:17 2021

@author: kaiser
"""

import numpy as np
import time
from pysma.models.heat_transfer_coefficient import *
import pysma
# =============================================================================
# A = pysma.material.SMAHC('A')
# dx = 0.1/1000
# T_init = 23
# a_air = 30
# dt = 0.01
# =============================================================================

class Heat_transfer:
    def __init__(self, SMAHC, dx, T_init, a_air, dt): # a_air 35
        self.name = '0D elastomer only'
        self.S = SMAHC
        self.dx = dx*1000 # dx in mm
        self.dx2 = self.dx**2 # dx^2 in mm^2
        self.dt = dt
        self.l = 1 
        self.b = 0.5 * self.S.w.circumference* 1000
        self.T_init = T_init
        #J/smK * m³/kg * kgK/J --> m²/s
        self.D_s = 1000*1000*self.S.s.lam/(self.S.s.rho*self.S.s.c) #lambda/(rho*c) --> Temperaturleitfähigkeit in mm2/s
        self.D_d = 1000*1000*self.S.d.lam/(self.S.d.rho*self.S.d.c) #lambda/(rho*c) --> Temperaturleitfähigkeit in mm2/s
        #J/sm²K * m³/kg * kgK/J --> m/s
        hpe = horizontal_plate_ehl(self.l, self.b, T_init)
        self.a_air = hpe.alpha(50)
        self.C_d = 1000* self.a_air/(self.S.d.rho*self.S.d.c*self.dx) #--> Temperaturleitfähigkeit in mm/s
        self.C_s = 1000*self.a_air/(self.S.s.rho*self.S.s.c*self.dx) #--> Temperaturleitfähigkeit in mm/s
        
        self.D_s_b = 1000*1000*(((self.S.s.lam+self.S.d.lam)/2)/(self.S.s.rho*self.S.s.c))
        self.D_d_b = 1000*1000*(((self.S.s.lam+self.S.d.lam)/2)/(self.S.d.rho*self.S.d.c))
        
        self.nz_d = int(1000*self.S.d.height/self.dx)+1
        self.nz_s = int(1000*self.S.s.height/self.dx)+1
      
        self.u0_d = T_init * np.ones(1)
        self.u_d = self.u0_d.copy()
        
        self.u0_s = T_init * np.ones(1)
        self.u_s = self.u0_s.copy()
        
        #Konduktion
        #self.E_cond_const = dx* 2 * self.S.d.lam * self.dt  # mal zwei da hier nur ein viertel #Durch 2 da nur halber Weg der Wärme, von aussen nach mitte
        #self.E_cond_const = self.S.d.lam * self.l* self.b * self.dt /(0.5 * self.dx * 1000)
        self.E_cond_const =  self.S.d.lam * np.pi * self.S.w.diameter * 0.5 * dt / (self.S.d.height * 0.5)
        
        self.dTdl_const = self.S.d.lam * np.pi * self.S.w.diameter * 0.5 * dt / (self.S.d.height * 0.5 * self.S.d.c * self.S.d.height * self.S.g.dist * self.S.d.rho)
        
    def do_timestep(self, u0_d, u_d, u0_s, u_s, Tsma0, Tsma):
        E_cond = 0
        T_dif = Tsma - u0_d[0]
        E_cond = T_dif * self.E_cond_const
        u_d[0] = u0_d[0] + T_dif * self.dTdl_const
    
        u0_d = u_d.copy()
        u0_s = u_s.copy()
        
        return u0_d, u_d, u0_s, u_s, E_cond
    
    
    def time_check(self):
        t0 = time.time()
        self.do_timestep(self.u0_d, self.u_d, self.u0_s, self.u_s, self.T_init, self.T_init)
        t1 = time.time()
        calc_time = t1-t0
        virtual_time = self.nsteps * self.dt
        time_for_1s = calc_time/virtual_time
        print(str(time_for_1s), 's for 1 s') 
        return calc_time


#ht = Heat_transfer(A, dx, T_init, a_air, dt)
        
        
        
    
    

                