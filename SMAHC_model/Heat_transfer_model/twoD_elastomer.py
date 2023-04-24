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
# ========================================================================================
# Heat transfer class
# ========================================================================================
class Heat_transfer:
    #Initialize Heat transfer problem
    def __init__(self, SMAHC, dx, Tu, a_air, dt): # a_air 35
        self.name = '2D elastomer only'
        self.S = SMAHC
        self.dx = dx*1000 # dx in mm
        self.dx2 = self.dx**2 # dx^2 in mm^2
        self.dt = dt
        self.l = self.S.length 
        self.Tu = Tu
        # Calculate thermal diffusivity td in mm²/s for the interlayer
        self.td = 1000*1000*self.S.d.lam/(self.S.d.rho*self.S.d.c) 
        # Calculate heat transfer coefficient to the surroundings tds in mm/s for interlayer
        self.tds = 1000* a_air/(self.S.d.rho*self.S.d.c*self.dx) 
        # Calculate number of elements of mesh in x-direction nx_d and in y direction ny_d
        self.nx_d, self.ny_d = int(1000*self.S.d.height/self.dx)+\
            1, int((1000*self.S.g.dist/2)/self.dx)+2
        # Calucalte maximum permissible discrete time step according to \
            #Courant-Friedrichs-Lewy stability criteria
        self.dt_max = 0.5*(self.dx2/(2*self.td))
        if self.dt_max > self.dt: 
            self.dt_max = self.dt
        # Initialize array
        self.u0_d = Tu * np.ones((self.nx_d, self.ny_d))
        self.u_d = self.u0_d.copy()
        # Create mask for boundary conditions
        self.u_d_boundary = np.zeros((self.nx_d, self.ny_d), dtype=bool)
        # Calculate constant for heat flux determination of one time step
        self.E_cond_const = 2 * self.S.d.lam * self.l * self.dt_max
        # Create mask for sma wire
        self.u_heat = self.u_d_boundary.copy()
        #Deterimen all cells that belong to SMA-wire
        for i in range(self.nx_d):
            for j in range(self.ny_d):
                p = np.sqrt((i*self.dx-0)**2 + (j*self.dx-((self.S.g.dist*1000)/2))**2)
                if p < (self.S.w.radius*1000) +self.dx/2:
                    self.u_heat[i,j] = True
        #Determine all cells that belong to SMA-wire surface
        self.surface_lst = []          
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
        self.u_tdown = self.u_d_boundary.copy()
        self.u_tdown[self.nx_d-1,:] = np.ones(self.ny_d)   
        #Heat transfer boundary condition to the surroundings
        self.u_d_up = self.u_d_boundary.copy()
        self.u_d_up[0,:] = np.zeros(self.ny_d)   
        self.u_d_up[0,:np.where(self.u_heat[0,:]==True)[0][0]] == True
        #Calculate the number of time steps to do for one time step of the main script
        self.nsteps = int(self.dt/self.dt_max)
        if self.nsteps == 0:
            self.nsteps = 1
    # Do timestep method to do one time step regarding the main script
    def do_timestep(self, u0_d, u_d, Tsma0, Tsma):
        #Initialize time step
        E_cond = 0
        dT = (Tsma-Tsma0)/self.nsteps
        #Run time steps according to number of time steps necessary
        for step in range(1,self.nsteps+1):
            #Linearize temperature change
            u0_d[self.u_heat] = Tsma0 + dT * step
            #Calculate heat flux between SMA-wire surface and interlayer for one time
            #step in the thermal domain
            T_dif = 0
            for se in self.surface_lst[1:-2]:
                T_dif += u0_d[se[0],se[1]-1] + u0_d[se[0]+1,se[1]]
            T_dif = (2 * len(self.surface_lst[1:-2]) * Tsma) - T_dif
            E_cond += self.E_cond_const * T_dif
            # Update temperature field using forward-difference in time, 
            # central-difference in space scheme
            u_d[1:-1, 1:-1] = u0_d[1:-1, 1:-1] + (self.dt_max/self.dx2) * self.td * \
                (u0_d[2:, 1:-1]  + u0_d[:-2, 1:-1] + u0_d[1:-1, 2:] +\
                 u0_d[1:-1, :-2]  - 4*u0_d[1:-1, 1:-1])
            #Update temperature field at boundaries
            u_d[self.nx_d-1, 1:-1] = u0_d[self.nx_d-1, 1:-1] + self.td *\
                (self.dt_max/self.dx2) * (u0_d[self.nx_d-1, 1:-1] +\
                u0_d[self.nx_d-1, 2:] + u0_d[self.nx_d-1, :-2] - 3*u0_d[self.nx_d-1, 1:-1])\
                    - self.tds*self.dt_max*(u0_d[self.nx_d-1, 1:-1]-self.Tu)
            u_d[self.u_d_right] = u_d[:,self.ny_d-2]
            u_d[self.u_d_left] = u_d[:,1]
            u_d[0, 1:-1] = u0_d[0, 1:-1] + self.td * (self.dt_max/self.dx2) * \
                (u0_d[1, 1:-1]  + u0_d[0, 2:] + u0_d[0, :-2]  - 3*u0_d[0, 1:-1]) - \
                    self.tds*self.dt_max*(u0_d[0, 1:-1]-self.Tu)
            #Copy temperature field
            u0_d = u_d.copy()
        return u0_d, u_d, E_cond