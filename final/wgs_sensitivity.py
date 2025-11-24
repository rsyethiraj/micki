#!/usr/bin/env python

import time
import numpy as np
from micki.db import read_from_db
from micki import Model, Reaction, ModelAnalysis
from micki.reactants import Adsorbate, _Fluid
from ase.units import _k, _Nav, m, kB
import sympy as sym


# This script runs sensitivty analysis for all 21 reaction conditions
# using a PFR reactor model.
# Sensitivy analysis includes CRC, TRC, rate orders, activation energy

start = time.time()

#Experimental Reaction conditions
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

# Read in database
sp = read_from_db('wgs.json', eref=['slab', 'co_g', 'h2o_g', 'h2_g'])
sp['ho-h'].symm = 2
sp['o-h-oh'].symm = 2
sp['o-co'].symm = 2

# Shift CO energy to match experiment binding enthalpy
sp['co'].dE = 0.09496182099234107 # Shift CO energy to match experiment binding enthalpy

# Shift H energy to match experiment binding enthalpy
sp['h'].dE = 0.119535059

# Shift OH and COOH based on optimization results
sp['oh'].dE = -.2171335007
sp['cooh'].dE = -0.0515426001


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


#number of catalyst sites calculated by multiplying
# wt% * dispersion / molar mass * catalyst mass
nsites = 47e-6 * 0.1501

# Define the area of one site
Asite = np.sqrt(3) * 3.8966**2 / (4 * m**2)

def run_pfr(T, flowrate, U0):
    atm2molar = 101325 / _k / T / _Nav / 1000.
    #set up CSTR to get initial starting surface concentrations
    model = Model(T, Asite, reactor='CSTR')
    model.lattice = {sp['slab']: {sp['slab']: 6}}
    model.add_reactions(rxns)
    model.set_fixed(['co_g', 'h2o_g', 'h2_g', 'co2_g'])

    
    V = 1; dt = 60 * 1000 * nsites / flowrate
    model.set_initial_conditions(U0)
    #solve CSTR model to get initial starting surface concentrations
    t1, U1, r1 = model.find_steady_state(maxiter=200000, epsilon=1e-8)
    # setup PFR Model
    pfr = Model(T, Asite, reactor='PFR', rhocat=1./V)
    pfr.lattice = model.lattice
    pfr.add_reactions(rxns)
    pfr.set_initial_conditions(U1)
    # sovle PFR MODEL
    U1pfr, r1pfr = pfr.solve(dt, 1000)
    # Return the TOF
    return U1, (U1pfr[-1]['co2_g'] - U0['co2_g']) * flowrate / (1000 * nsites)

crcs = []
trcs = []
rate_orders = []
Eas = []
for condition in conditions:
    T = condition[0]
    flowrate = condition[5]
    atm2molar = 101325 / _k / T / _Nav / 1000.

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
    # Calculate CRC for a given reaction condition
    U0, tofmid = run_pfr(T, flowrate, U0)

    crc = {}
    scale = 0.001
    dG = scale * kB * T
    for name, rxn in rxns.items():
        rxn.set_scale('kfor', 1 - scale)
        rxn.set_scale('krev', 1 - scale)
        _, toflow = run_pfr(T, flowrate, U0)
        
        rxn.set_scale('kfor', 1 + scale)
        rxn.set_scale('krev', 1 + scale)
        _, tofhigh = run_pfr(T, flowrate, U0)

        rxn.set_scale('kfor', 1)
        rxn.set_scale('krev', 1)

        crc[name] = (tofhigh - toflow) / (2 * scale * tofmid)
    crcs.append(crc)
    
    # Calcualte TRC for a given reaction condition
    trc = {}
    for name, species in sp.items():
        if not isinstance(species, Adsorbate) or species.ts or name == 'slab':
            continue

        species.dE -= dG
        for _, rxn in rxns.items():
            rxn.update(T=T, Asite=Asite, force=True)
        _, toflow = run_pfr(T, flowrate, U0)

        species.dE += 2*dG
        for _, rxn in rxns.items():
            rxn.update(T=T, Asite=Asite, force=True)
        _, tofhigh = run_pfr(T, flowrate, U0)

        species.dE -= dG
        for _, rxn in rxns.items():
            rxn.update(T=T, Asite=Asite, force=True)

        trc[name] = (toflow - tofhigh) * kB * T / (tofmid * dG)
    trcs.append(trc)
    
    # Calculate rate orders for a given reaction condition
    drho = 0.00005
    rate_order = {}
    for name in ['co_g', 'h2o_g', 'h2_g', 'co2_g']:
        species = sp[name]
        rhomid = U0[name]
        if U0[name] - drho < 0:
            continue
        U0[name] -= drho
        _, toflow = run_pfr(T, flowrate, U0)
        U0[name] += 2*drho
        _, tofhigh = run_pfr(T, flowrate, U0)
        U0[name] = rhomid

        rate_order[name] = (rhomid / tofmid) * (tofhigh - toflow) / (2 * drho)
    rate_orders.append(rate_order)
    
    # Calculate activation energy for a given reaction condition
    dT = 0.01
    _, toflow = run_pfr(T - dT, flowrate, U0)
    _, tofhigh = run_pfr(T + dT, flowrate, U0)
    Ea = kB * T**2 * (tofhigh - toflow) / (2 * tofmid * dT)
    Eas.append(Ea)

stop = time.time()
print('total time:', stop - start)
