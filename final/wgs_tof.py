#!/usr/bin/env python

import time
import numpy as np
from micki.db import read_from_db
from micki import Model, Reaction, ModelAnalysis
from micki.reactants import Adsorbate, _Fluid
from ase.units import _k, _Nav, m
import sympy as sym

start = time.time()

#Experimental Reaction Conditions
conditions = [
              (523, 0.154, 0.208, 0.000, 0.000, 102.9),  # 1
              (548, 0.055, 0.208, 0.000, 0.000, 85.9),   # 2
              (548, 0.105, 0.208, 0.000, 0.000, 97.4),   # 3
              (548, 0.137, 0.062, 0.000, 0.000, 118.0),  # 4
              (548, 0.144, 0.104, 0.000, 0.000, 114.8),  # 5
              (548, 0.148, 0.145, 0.000, 0.000, 108.4),  # 6
              (548, 0.145, 0.208, 0.000, 0.000, 101.4),  # 7
              (548, 0.106, 0.208, 0.068, 0.000, 106.9),  # 8
              (548, 0.104, 0.208, 0.109, 0.000, 105.6),  # 9
              (548, 0.140, 0.208, 0.151, 0.000, 110.1),  # 10
              (548, 0.102, 0.208, 0.192, 0.000, 109.1),  # 11
              (548, 0.134, 0.208, 0.000, 0.037, 103.5),  # 12
              (548, 0.156, 0.208, 0.000, 0.037, 102.1),  # 13
              (548, 0.130, 0.208, 0.000, 0.123, 105.9),  # 14
              (548, 0.134, 0.208, 0.177, 0.123, 95.7),   # 15
              (548, 0.132, 0.208, 0.000, 0.173, 103.8),  # 16
              (548, 0.145, 0.208, 0.000, 0.191, 94.4),   # 17
              (548, 0.159, 0.208, 0.000, 0.208, 101.1),  # 18
              (548, 0.198, 0.208, 0.000, 0.000, 102.6),  # 19
              (548, 0.223, 0.208, 0.000, 0.000, 88.3),   # 20
              (573, 0.150, 0.208, 0.000, 0.000, 103.4),  # 21
              ]

#Experimental TOFs
experiment = np.array([3.68,
                       8.56,
                       8.06,
                       3.63,
                       4.81,
                       5.56,
                       7.27,
                       6.05,
                       5.59,
                       6.12,
                       6.03,
                       4.13,
                       2.77,
                       2.67,
                       2.67,
                       2.55,
                       2.28,
                       2.29,
                       7.29,
                       7.09,
                       15.44])

#Theoretical TOFs determined by Grabow et al.
manos = np.array([3.38,
                  10.01,
                  7.51,
                  2.93,
                  4.36,
                  5.31,
                  6.87,
                  7.65,
                  7.63,
                  7.01,
                  7.67,
                  4.94,
                  3.21,
                  3.10,
                  2.95,
                  2.91,
                  2.38,
                  2.54,
                  6.31,
                  5.76,
                  13.20])

#number of catalyst sites calculated by multiplying
# wt% * dispersion / molar mass * catalyst mass
nsites = 47e-6 * 0.1501

#read in database of species
sp = read_from_db('wgs.json', eref=['slab', 'co_g', 'h2o_g', 'h2_g'])
sp['ho-h'].symm = 2
sp['o-h-oh'].symm = 2
sp['o-co'].symm = 2

# Shift CO energy to match experiment binding enthalpy
sp['co'].dE = 0.09496182099234107
# Shift H energy to match experiment binding enthalpy
sp['h'].dE = 0.236689058/2.0

# Shift energy of OH and COOH based on optimization results
sp['oh'].dE = -0.2171335
sp['cooh'].dE = -0.0515426

# Add lateral interactions
sp['co'].lateral = 2*0.784423808 * sp['co'].symbol
sp['o'].lateral = 1.147079243 * sp['co'].symbol + 2*1.11913902 * sp['o'].symbol
sp['co'].lateral += 1.147079243 * sp['o'].symbol

sp['h'].lateral = 0.237728186 * sp['co'].symbol
sp['co'].lateral += 0.237728186 * sp['h'].symbol

sp['h2o'].lateral = 0.10555879 * sp['co'].symbol
sp['co'].lateral += 0.10555879 * sp['h2o'].symbol

sp['oh'].lateral = 0.263160274 * sp['co'].symbol
sp['co'].lateral += 0.263160274 * sp['oh'].symbol

sp['cooh'].lateral = 1.901900269 * sp['co'].symbol
sp['co'].lateral += 1.901900269 * sp['cooh'].symbol

# Not expected to be relevant:
#sp['hcoo'].coverage = 1.082712362 * sp['co'].symbol
#sp['cho'].coverage = 2.540122148 * sp['co'].symbol

# Define reactions in model
rxns = {'co_ads': Reaction(sp['co_g'], sp['co'], method='STICK'),
        'h2o_ads': Reaction(sp['h2o_g'], sp['h2o'], method='STICK'),
        'h2_ads': Reaction(sp['h2_g'], 2*sp['h'], method='STICK'),
        'ho-h': Reaction(sp['h2o'], sp['oh'] + sp['h'], ts=sp['ho-h']),
        'o-h': Reaction(sp['oh'], sp['o'] + sp['h'], ts=sp['o-h']),
        'o-h-oh': Reaction(2*sp['oh'], sp['o'] + sp['h2o'], method='DIEQUIL'),
        'o-co': Reaction(sp['o'] + sp['co'], sp['co2_g'], ts=sp['o-co']),
        'co-oh': Reaction(sp['cooh'], sp['co'] + sp['oh'], ts=sp['co-oh']),
        'oco-h': Reaction(sp['cooh'] + sp['slab'], sp['co2_g'] + sp['h'], ts=sp['oco-h']),
        'oco-h-o': Reaction(sp['cooh'] + sp['o'], sp['co2_g'] + sp['oh'], method='EQUIL'),
        'oco-h-oh': Reaction(sp['cooh'] + sp['oh'], sp['co2_g'] + sp['h2o'], method='EQUIL'),
        # Not expected to be relevant:
        #'oco-h-co': Reaction(sp['cooh'] + sp['co'], sp['co2_g'] + sp['hco'], ts=sp['oco-h-co']),
        #'h-coo': Reaction(sp['hcoo'], sp['co2_g'] + sp['h'], ts=sp['h-coo']),
        #'o-h-coo': Reaction(sp['hcoo'] + sp['o'], sp['co2_g'] + sp['oh'], ts=sp['o-h-coo']),
        #'ho-h-coo': Reaction(sp['hcoo'] + sp['oh'], sp['co2_g'] + sp['h2o'], ts=sp['ho-h-coo']),
        #'h-co': Reaction(sp['cho'], sp['co'] + sp['h'], ts=sp['h-co']),
        }

# Define the area of one site
Asite = np.sqrt(3) * 3.8966**2 / (4 * m**2)

tof = []
r12=[]
r11=[]
r1211=[]
# Set up Loop to calcualate TOF for all 21 conditions
for condition in conditions:
    #set up CSTR to get initial starting surface concentrations
    T = condition[0]
    atm2molar = 101325 / _k / T / _Nav / 1000.
    model = Model(T, Asite, reactor='CSTR')
    model.lattice = {sp['slab']: {sp['slab']: 6}}
    model.add_reactions(rxns)
    model.set_fixed(['co_g', 'h2o_g', 'h2_g', 'co2_g'])
    #initialize gas phase concentrations from flow rates
    U0 = {'co_g': atm2molar * condition[1],
          'h2o_g': atm2molar * condition[2],
          'co2_g': atm2molar * condition[3],
          'h2_g': atm2molar * condition[4],
          'co': 0.56275433599205904,
          'cooh': 1.3744935762217692e-12,
          'h': 1.2798504989912461e-05,
          'h2o': 1.791655916669818e-05,
          'o': 9.4267631566852731e-16,
          'oh': 1.7137327904883546e-10,
          }

    #define the integration space-time dt
    V = 1; dt = 60 * 1000 * nsites / condition[5]
    model.set_initial_conditions(U0)
    #solve CSTR model to get initial starting surface concentrations
    t1, U1, r1 = model.find_steady_state(maxiter=200000, epsilon=1e-8)
    # setup PFR Model
    pfr = Model(T, Asite, reactor='PFR', rhocat=1./V)
    pfr.lattice = model.lattice
    pfr.add_reactions(rxns)
    pfr.set_initial_conditions(U1)
    #solve PFR for steady state results
    U1pfr, r1pfr = pfr.solve(dt, 1000)
    # append current TOF to list of TOFS for all conditions
    tof.append((U1pfr[-1]['co2_g'] - atm2molar * condition[3]) * condition[5] / (1000 * nsites))
    r11.append((r1pfr[-1]['co-oh']))
    r12.append((r1pfr[-1]['oco-h']))
    r1211.append((r1pfr[-1]['oco-h'])/(r1pfr[-1]['co-oh'])) 
    
#evaluate the coverage at U1 and store in dictionary indexed by species
coverage = {}
for name, U in U1.items():
    if name in sp and sp[name].symbol is not None:
        coverage[sp[name].symbol] = U
    
stop = time.time()
print('total time:', stop - start)
