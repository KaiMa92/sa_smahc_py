# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:11:17 2021

@author: kaiser
"""

import numpy as np
import time

class Heat_transfer:
    def __init__(self, SMAHC, dx, T_init, a_air, dt): # a_air 35
        self.name = '2D elastomer only'
        self.S = SMAHC
        self.dx = dx*1000 # dx in mm
        self.dx2 = self.dx**2 # dx^2 in mm^2
        self.dt = dt
        self.l = self.S.length 
        self.T_init = T_init
        self.u_s = np.nan
        self.u0_s = np.nan
        #J/smK * m³/kg * kgK/J --> m²/s
       
        self.D_d = 1000*1000*self.S.d.lam/(self.S.d.rho*self.S.d.c) #lambda/(rho*c) --> Temperaturleitfähigkeit in mm2/s
        #J/sm²K * m³/kg * kgK/J --> m/s
        self.C_d = 1000* a_air/(self.S.d.rho*self.S.d.c*self.dx) #--> Temperaturleitfähigkeit in mm/s
     
        self.nx_d, self.ny_d = int(1000*self.S.d.height/self.dx)+1, int((1000*self.S.g.dist/2)/self.dx)+2
        self.nx_s, self.ny_s = int(1000*self.S.s.height/self.dx)+1, int((1000*self.S.g.dist/2)/self.dx)+2
        
        self.dt_max = 0.5*(self.dx2/(2*self.D_d))
        if self.dt_max > self.dt: 
            self.dt_max = self.dt

        print(self.dt_max)
      
        self.u0_d = T_init * np.ones((self.nx_d, self.ny_d))
        self.u_d = self.u0_d.copy()
        self.u_d_boundary = np.zeros((self.nx_d, self.ny_d), dtype=bool)
         
        self.E_cond_const = 2 * self.S.d.lam * self.l * self.dt_max #Kf_0p050 /(0.5 * 1000) *2
        #Boundary for wire mask
        self.surface_lst = []
        self.u_heat = self.u_d_boundary.copy()
        
        
        for i in range(self.nx_d):
            for j in range(self.ny_d):
                p = np.sqrt((i*self.dx-0)**2 + (j*self.dx-((self.S.g.dist*1000)/2))**2)
                #p2 = (i*self.dx-0)**2 + (j*self.dx-(self.S.g.dist/2))**2
                
                if p < (self.S.w.radius*1000) +self.dx/2:
                    self.u_heat[i,j] = True

                    
        for i in range(self.nx_d):
            for j in range(self.ny_d):                       
                if self.u_heat[i,j] == True and self.u_heat[i+1,j] == True:
                    self.surface_lst.append([i,j])
                    break
                elif self.u_heat[i,j] == True and self.u_heat[i+1,j] == False:
                    self.surface_lst.append([i,j])
                else:
                    pass
        
        #Boundary symmetrie right
        self.u_d_right = self.u_d_boundary.copy()
        self.u_d_right[:,self.ny_d-1] = np.ones(self.nx_d)    
        
        #Boundary symmetrie left
        self.u_d_left = self.u_d_boundary.copy()
        self.u_d_left[:,0] = np.ones(self.nx_d) 
        
        #Boundary symmetrie down
        self.u_d_down = self.u_d_boundary.copy()
        self.u_d_down[self.nx_d-1,:] = np.ones(self.ny_d)   
        
         #Boundary symmetrie up
        self.u_d_up = self.u_d_boundary.copy()
        self.u_d_up[0,:] = np.zeros(self.ny_d)   
        self.u_d_up[0,:np.where(self.u_heat[0,:]==True)[0][0]] == True
        
        
        # Number of timesteps
        self.nsteps = int(self.dt/self.dt_max)
        if self.nsteps == 0:
            self.nsteps = 1
        

    def do_timestep(self, u0_d, u_d, u0_s, u_s, Tsma0, Tsma):
        E_cond = 0
        dT = (Tsma-Tsma0)/self.nsteps
        #print('dT: ', dT)
        
        
        for step in range(1,self.nsteps+1):
            u0_d[self.u_heat] = Tsma0 + dT * step
            #Konduktion
            T_dif = 0
            for se in self.surface_lst[1:-2]:
                T_dif += u0_d[se[0],se[1]-1] + u0_d[se[0]+1,se[1]]
            T_dif = (2 * len(self.surface_lst[1:-2]) * Tsma) - T_dif
            #print(T_dif)
            E_cond += self.E_cond_const * T_dif
            # Propagate with forward-difference in time, central-difference in space
            
            u_d[1:-1, 1:-1] = u0_d[1:-1, 1:-1] + (self.dt_max/self.dx2) * self.D_d * (u0_d[2:, 1:-1]  + u0_d[:-2, 1:-1] + u0_d[1:-1, 2:] + u0_d[1:-1, :-2]  - 4*u0_d[1:-1, 1:-1])
          
            
            #Boundarys distance Layer
            u_d[self.nx_d-1, 1:-1] = u0_d[self.nx_d-1, 1:-1] + self.D_d * (self.dt_max/self.dx2) * (u0_d[self.nx_d-1, 1:-1]  + u0_d[self.nx_d-1, 2:] + u0_d[self.nx_d-1, :-2]  - 3*u0_d[self.nx_d-1, 1:-1]) - self.C_d*self.dt_max*(u0_d[self.nx_d-1, 1:-1]-self.T_init)
            #u_d[self.nx_d-1, 1:-1] = u0_d[self.nx_d-1, 1:-1] + self.D_d_b * (self.dt_max/self.dx2) * (u0_d[self.nx_d-2, 1:-1]  + u0_d[self.nx_d-1, :-2] + u0_d[self.nx_d-1, 2:]  - 3*u0_d[self.nx_d-1, 1:-1]) - self.C_d*self.dt_max*(u0_d[self.nx_d-1, 1:-1]-self.T_init)
            u_d[self.u_d_right] = u_d[:,self.ny_d-2]
            u_d[self.u_d_left] = u_d[:,1]
            #u[u_up] = u[1,:]
            u_d[0, 1:-1] = u0_d[0, 1:-1] + self.D_d * (self.dt_max/self.dx2) * (u0_d[1, 1:-1]  + u0_d[0, 2:] + u0_d[0, :-2]  - 3*u0_d[0, 1:-1]) - self.C_d*self.dt_max*(u0_d[0, 1:-1]-self.T_init)
    
            u0_d = u_d.copy()
        
        return u0_d, u_d, u0_s, u_s, E_cond