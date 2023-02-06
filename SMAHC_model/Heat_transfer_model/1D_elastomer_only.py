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
        self.name = '1D elastomer only'
        self.S = SMAHC
        self.dx = dx*1000 # dx in mm
        self.dx2 = self.dx**2 # dx^2 in mm^2
        self.dt = dt
        self.l = 1 
        self.b = self.S.w.circumference* 1000*0.5
        self.T_init = T_init
        #J/smK * m³/kg * kgK/J --> m²/s
        self.D_s = 1000*1000*self.S.s.lam/(self.S.s.rho*self.S.s.c) #lambda/(rho*c) --> Temperaturleitfähigkeit in mm2/s
        self.D_d = 1000*1000*self.S.d.lam/(self.S.d.rho*self.S.d.c) #lambda/(rho*c) --> Temperaturleitfähigkeit in mm2/s
        #J/sm²K * m³/kg * kgK/J --> m/s
        hpe = horizontal_plate_ehl(self.l, self.b, T_init)
        self.a_air = a_air#hpe.alpha(50)
        self.C_d = 1000* self.a_air/(self.S.d.rho*self.S.d.c*self.dx) #--> Temperaturleitfähigkeit in mm/s
        self.C_s = 1000*self.a_air/(self.S.s.rho*self.S.s.c*self.dx) #--> Temperaturleitfähigkeit in mm/s
        
        self.D_s_b = 1000*1000*(((self.S.s.lam+self.S.d.lam)/2)/(self.S.s.rho*self.S.s.c))
        self.D_d_b = 1000*1000*(((self.S.s.lam+self.S.d.lam)/2)/(self.S.d.rho*self.S.d.c))
        
        self.nz_d = int(1000*self.S.d.height/self.dx)+1
        self.nz_s = int(1000*self.S.s.height/self.dx)+1
        
        self.dt_max = 0.5*(self.dx2/self.D_d)
        if self.dt_max > self.dt: 
            self.dt_max = self.dt

        print(self.dt_max)
      
        self.u0_d = T_init * np.ones((self.nz_d))
        self.u_d = self.u0_d.copy()
        
        self.u0_s = T_init * np.ones((self.nz_s))
        self.u_s = self.u0_s.copy()
        
        #Konduktion
        #self.E_cond_const = dx* 2 * self.S.d.lam * self.dt  # mal zwei da hier nur ein viertel #Durch 2 da nur halber Weg der Wärme, von aussen nach mitte
        self.E_cond_const = self.S.d.lam * self.l* self.b * self.dt_max /(self.dx * 1000)#(0.5 * self.dx * 1000)
         
        # Number of timesteps
        self.nsteps = int(self.dt/self.dt_max)
        
        
    def do_timestep(self, u0_d, u_d, u0_s, u_s, Tsma0, Tsma):
        E_cond = 0
        dT = (Tsma-Tsma0)/self.nsteps
        #print('dT: ', dT)
        
        for step in range(1,self.nsteps+1):
            u0_d[0] = Tsma0 + dT * step
            #Konduktion
            T_dif = u0_d[0]-u0_d[1]
            E_cond += self.E_cond_const * T_dif
            # Propagate with forward-difference in time, central-difference in space
            
            u_d[1:-1] = u0_d[1:-1] + (self.dt_max/self.dx2) * self.D_d * (u0_d[2:]  + u0_d[:-2]  - 2*u0_d[1:-1])
            #u_s[1:-1] = u0_s[1:-1] + (self.dt_max/self.dx2) * self.D_s * (u0_s[2:]  + u0_s[:-2]  - 2*u0_s[1:-1])
            
            
            u_d[-1] = u0_d[-1] + (self.dt_max/self.dx2) * self.D_d * (u0_d[-2] - u0_d[-1]) + self.C_d*self.dt_max*(u0_d[-1]-self.T_init)
            #Boundarys distance Layer
            
# =============================================================================
#             u_d[self.u_d_right] = u_d[:,self.ny_d-2]
#             u_d[self.u_d_left] = u_d[:,1]
#             #u[u_up] = u[1,:]
#             #air
#             u_d[0, 1:-1] = u0_d[0, 1:-1] + self.D_d * (self.dt_max/self.dx2) * (u0_d[1, 1:-1]  + u0_d[0, 2:] + u0_d[0, :-2]  - 3*u0_d[0, 1:-1]) - self.C_d*self.dt_max*(u0_d[0, 1:-1]-self.T_init)
#             #boundary to steel
#             u_d[self.nx_d-1, 1:-1] = u0_d[self.nx_d-1, 1:-1] + self.D_d * (self.dt_max/self.dx2) * (u0_d[self.nx_d-2, 1:-1]  + u0_d[self.nx_d-1, :-2] + u0_d[self.nx_d-1, 2:]  - 3*u0_d[self.nx_d-1, 1:-1]) +  (self.D_d_b) * (self.dt_max/self.dx) * (u0_s[0, 1:-1]-u0_d[self.nx_d-1, 1:-1])
#              
#             #boundary_elastomer
#             u_s[0, 1:-1] = u0_s[0, 1:-1] + self.D_s * (self.dt_max/self.dx2) * (u0_s[1, 1:-1]  + u0_s[0, 2:] + u0_s[0, :-2]  - 3*u0_s[0, 1:-1]) +  (self.D_s_b) * (self.dt_max/self.dx) * (u0_d[self.nx_s-1, 1:-1]-u0_s[0, 1:-1])
#             u_s[self.u_s_right] = u_s[:,self.ny_s-2]
#             u_s[self.u_s_left] = u_s[:,1]
#             #ul[ul_down] = ul[nxl-2,:]
#             #air
#             u_s[self.nx_s-1, 1:-1] = u0_s[self.nx_s-1, 1:-1] + self.D_s * (self.dt_max/self.dx2) * (u0_s[self.nx_s-2, 1:-1]  + u0_s[self.nx_s-1, 2:] + u0_s[self.nx_s-1, :-2]  - 3*u0_s[self.nx_s-1, 1:-1]) - self.C_s*self.dt_max*(u0_s[self.nx_s-1, 1:-1]-self.T_init)
# =============================================================================
            
    
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
        
        
        
    
    

                