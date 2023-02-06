# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:14:56 2020

@author: kaiser
"""
import json
import pysma 
import math
import numpy as np

class SMA_material:
    """
    This class represents a sma wire
    :attribute density: kg/mm³
    :attribute diameter: mm
    """
    def __init__(self, name):
        self.path = pysma.access.wire_characteristics(name)
        with open(self.path) as file:
            att_dct = json.load(file)
        self.material_name = name
        self.rho = float(att_dct['density']) # density in kg/m^3
        self.diameter = float(att_dct['diameter']) # m
        self.radius = self.diameter/2
        self.crosssection_area = np.pi * (self.diameter/2)**2 # m^2
        self.circumference = self.diameter*3.14 # m
        self.CMs = float(att_dct['CMs']) # Anstieg der Umwandlungstemperaturen gradient Spannung ueber Temperatur MPa/K ungefaehr gleich fuer alle Umwandlungstemperaturen
        self.CMf = float(att_dct['CMf']) # Anstieg der Umwandlungstemperaturen gradient Spannung ueber Temperatur MPa/K ungefaehr gleich fuer alle Umwandlungstemperaturen
        self.CAs = float(att_dct['CAs']) # Anstieg der Umwandlungstemperaturen gradient Spannung ueber Temperatur MPa/K ungefaehr gleich fuer alle Umwandlungstemperaturen
        self.CAf = float(att_dct['CAf']) # Anstieg der Umwandlungstemperaturen gradient Spannung ueber Temperatur MPa/K ungefaehr gleich fuer alle Umwandlungstemperaturen
        self.As = float(att_dct['As']) # Austenitstarttemperatur in C°
        self.Af = float(att_dct['Af']) # Austenitfinishtemperatur in C°
        self.Ms = float(att_dct['Ms']) # Martensitstarttemperatur in C°
        self.Mf = float(att_dct['Mf']) # Martensitfinishtemperatur in C°
        self.Rs = float(att_dct['Rs']) # R starttemperatur in C°
        self.Rf = float(att_dct['Rf']) # R finishtemperatur in C°
        self.max_strain_zero_load = float(att_dct['max_strain_zero_load']) # Maximale Lastfreie Transformationsdehnung in -
        self.EA = float(att_dct['EA']) #E-Modul Austenit in Pa
        self.EM = float(att_dct['EM']) #E-Modul Martensit in Pa
        self.dH_M_to_A = float(att_dct['dH_M_to_A']) #latent heat of transformation in J/kg
        self.dH_A_to_R = float(att_dct['dH_A_to_R']) #latent heat of transformation in J/kg
        self.dH_R_to_M = float(att_dct['dH_R_to_M']) #latent heat of transformation in J/kg
        self.cA = float(att_dct['cA']) #specific heat capacity in J/kgK
        self.cM = float(att_dct['cM']) #specific heat capacity in J/kgK
        self.rM = float(att_dct['rM']) #resistivity in ohm*m
        self.rA = float(att_dct['rA']) #resistivity in ohm*m
        self.k = float(att_dct['k']) #Fit parameter for load dependent transformation strain -
        self.max_trans_strain = float(att_dct['max_trans_strain']) #Maximum transformation strain -
        
    def E(self,mf):
        return self.EA + mf*(self.EM-self.EA)
          
    def rhoR(self,mf):
        return self.rhoRA + mf(self.rhoRM-self.rhoRA)
    
    def ceff(self,mf):
        return self.c + ((2*self.deltaH)/(self.Af-self.As))*math.sin(mf*math.pi)
    
       
class SMA_wire:
    def __init__(self, wire_material, length_m):
        SMA_material.__init__(self, wire_material)
        self.length = length_m
        self.volume = self.length * self.crosssection_area
        self.mass = self.rho * self.volume
        self.surface = self.length * self.circumference
        self.resM = self.rM * self.length / self.crosssection_area
        self.resA = self.rA * self.length / self.crosssection_area
        
 
    
class SMA_grid:
    def __init__(self, SMA_wire, n, dist, electric_connection):
        self.wire = SMA_wire
        self.n = n
        self.dist = dist
        self.w = self.n * self.dist
        self.wire.hlt = self.wire.crosssection_area * self.n / self.w #homogenous layer thickness
        if electric_connection == 'series':
            self.resM = self.wire.resM * self.n
            self.resA = self.wire.resA * self.n
        if electric_connection == 'parallel':
            pass
        


class Homogeneous_substrate_material:    
    def __init__(self, name):
        self.path = pysma.access.substrate_characteristics(name)
        with open(self.path) as file:
            att_dct = json.load(file)
        self.E = float(att_dct['E'])
        self.lam = float(att_dct['lambda'])
        self.c = float(att_dct['c'])
        self.rho = float(att_dct['density'])
        
  
      
class Substrate(Homogeneous_substrate_material):   
    def __init__(self,substrate_material, length, height, width):
        Homogeneous_substrate_material.__init__(self, substrate_material)
        self.length = length
        self.height = height
        self.width = width
        self.I = self.width*self.height**3/12



class Homogeneous_distance_material:    
    def __init__(self, name):
        self.path = pysma.access.distance_layer_characteristics(name)
        with open(self.path) as file:
            att_dct = json.load(file)
        self.E = float(att_dct['E'])
        self.lam = float(att_dct['lambda'])
        self.c = float(att_dct['c'])
        self.rho = float(att_dct['density'])
        

        
class Distance_layer(Homogeneous_distance_material):   
    def __init__(self,distance_material, length, height, width):
        Homogeneous_distance_material.__init__(self, distance_material)
        self.length = length
        self.height = height
        self.width = width
        self.I = self.width*self.height**3/12
        


class SMAHC:
    
    def __init__(self, smahc_type):
        self.name = smahc_type
        self.path = pysma.access.smahc_characteristics(smahc_type)
        with open(self.path) as file:
            att_dct = json.load(file) 
              
        self.w = SMA_wire(att_dct['sma_material'] ,float(att_dct['active_length']))
        self.g = SMA_grid(self.w, float(att_dct['nwires']), float(att_dct['width'])/float(att_dct['nwires']), att_dct['electric_connection'])
        self.s = Substrate(att_dct['substrate_material'], float(att_dct['active_length']), float(att_dct['substrate_thickness']), float(att_dct['width']))
        self.d = Distance_layer(att_dct['distance_material'],float(att_dct['active_length']), float(att_dct['distance_layer_thickness']), float(att_dct['width']))
        
        self.width = float(att_dct['width'])#m
        self.length = float(att_dct['active_length']) #m
        
        self.dist_sub_sma = self.d.height + self.s.height/2 + self.w.radius
        self.hslt = self.w.crosssection_area / self.g.dist # homogeneous sma layer thickness
        self.stiffness = self.s.E * self.s.height**3/(12 * self.dist_sub_sma**2 * self.hslt)
        




# =============================================================================
# class SMAHC:
#     
#     def __init__(self, smahc_type):
#         self.name = smahc_type
#         self.path = pysma.access.smahc_characteristics(smahc_type)
#         with open(self.path) as file:
#             att_dct = json.load(file)
#             
#         self.w = float(att_dct['width'])#m
#         self.l = float(att_dct['active_length']) #m
#         self.electric_connection = att_dct['electric_connection']
#         
#         self.sma = pysma.material.sma_material(att_dct['sma_material'])
#         self.sma.number_wires = float(att_dct['nwires'])
#         self.sma.hlt = self.sma.crosssection_area*self.sma.number_wires/self.w #homogenous layer thickness
#         
#         
#         self.sma.single_wire = Single_wire()
#         self.sma.single_wire.mass = self.sma.density * self.l * self.sma.crosssection_area
#         self.sma.single_wire.surface = self.l * self.sma.circumference
#         
#         self.sma.all_wires = All_wires()
#         self.sma.all_wires.mass = self.sma.density * self.l * self.sma.crosssection_area*self.sma.number_wires
#         self.sma.all_wires.surface = self.l * self.sma.circumference*self.sma.number_wires
# 
#         
#         self.lam = pysma.material.homogeneous_laminate_material(att_dct['laminate_material'])
#         self.lam.thickness = float(att_dct['laminate_thickness'])
#         self.lam.I = (self.w*self.lam.thickness**3)/12
#         
#         self.distance_layer_thickness = float(att_dct['distance_layer_thickness'])+(self.lam.thickness/2)+(self.sma.diameter/2)
#         
#         self.max_stress_due_to_laminate = self.sma.max_strain_zero_load/(((self.sma.hlt*self.w*self.distance_layer_thickness**2)/(self.lam.E*self.lam.I))+(1/self.sma.EA))
# #b = bending_beam('Jani_example','steel', 50*10**-3, 500*10**-3, 1.5*10**-3, 10)
#         
#     def max_stress(self,external_force,delta_horicontal_deflection_percent):
#         distance_actuator_to_external_load = 0.005
#         a = external_force * self.distance_layer_thickness *self.sma.EA*(self.l+distance_actuator_to_external_load)*(1-delta_horicontal_deflection_percent)/(self.lam.E*self.lam.I)
#         b = 1+((self.sma.hlt*self.sma.EA*self.w*self.distance_layer_thickness**2)/(self.lam.E*self.lam.I))
#         sigma_sma_max = (a + self.sma.max_strain_zero_load*self.sma.EA)/b
#         return sigma_sma_max
# =============================================================================
    
    





