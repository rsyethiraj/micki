"""Microkinetic modeling objects"""

from __future__ import print_function

import os
import glob
import tempfile
import shutil
import warnings

from collections import OrderedDict

import numpy as np
import sympy as sym
import sksundae as sun

from copy import copy
from ase.units import kB, _hplanck, kg, _k, _Nav, mol

from micki.reactants import _Thermo, _Fluid, _Reactants, Gas, Liquid, Adsorbate
from micki.reactants import Electron

from micki.lattice import Lattice


class Reaction(object):
    def __init__(self, reactants, products, ts=None, method=None, S0=1.,
                 dG_act=None, dground=False, reversible=True):

        # Wrap reactants and products in _Reactants type
        if isinstance(reactants, _Thermo):
            self.reactants = _Reactants([reactants])
        elif isinstance(reactants, _Reactants):
            self.reactants = reactants
        else:
            raise NotImplementedError

        if isinstance(products, _Thermo):
            self.products = _Reactants([products])
        elif isinstance(products, _Reactants):
            self.products = products
        else:
            raise NotImplementedError

        # Determine the number of sites on the LHS and the RHS of the reaction,
        # then add "bare" sites as necessary to balance the site number.
        vacancies = OrderedDict()
        self.species = []
        for species in self.reactants:
            if species not in self.species:
                self.species.append(species)
            if species.sites is None:
                continue
            for site in species.sites:
                if site in vacancies:
                    vacancies[site] -= 1
                else:
                    vacancies[site] = -1
        for species in self.products:
            if species not in self.species:
                self.species.append(species)
            if species.sites is None:
                continue
            for site in species.sites:
                if site in vacancies:
                    vacancies[site] += 1
                else:
                    vacancies[site] = 1

        # If the user supplied "bare" sites, count them too
        for species in self.reactants:
            if species in vacancies:
                vacancies[species] -= 1
        for species in self.products:
            if species in vacancies:
                vacancies[species] += 1

        for vacancy, nvac in vacancies.items():
            # There are extra sites on the RHS, so add some to the LHS
            if nvac > 0:
                self.reactants += nvac * vacancy
            # There are extra sites on the LHS, so add some to the RHS
            elif nvac < 0:
                self.products += abs(nvac) * vacancy

        self.ts = None
        # The user supplied a transition state species
        if ts is not None:
            assert dG_act is None, \
                    "Cannot specify both barrier height and transition state!"
            # Wrap the TS in the _Reactants class
            if isinstance(ts, _Thermo):
                self.ts = _Reactants([ts])
            elif isinstance(ts, _Reactants):
                self.ts = ts
            # Fail if the user supplies something other than a _Thermo
            # or _Reactants
            else:
                raise NotImplementedError
        # FIXME: Add stoichiometry checking to ensure logical reactions.
        # Caveat: Don't fail on unbalanced adsorption sites, since some
        # species take up more than one site.

        self.involves_catalyst = False
        for species in self.reactants:
            if isinstance(species, Adsorbate):
                self.involves_catalyst = True
                break
        if not self.involves_catalyst:
            for species in self.products:
                if isinstance(species, Adsorbate):
                    self.involves_catalyst = True
                    break

        self.method = method
        if self.method is None:
            if self.ts is not None:
                self.method = 'TST'
            else:
                self.method = 'EQUIL'

        if isinstance(self.method, str):
            self.method = self.method.upper()

        self.S0 = S0
        self.keq = None
        self.kfor = None
        self.krev = None
        self.T = None
        self.Asite = None
        self.L = None
        self.scale_params = ['dH', 'dS', 'dH_act', 'dS_act', 'kfor', 'krev']
        self.alpha = None
        self.reversible = reversible

        # Scaling for sensitivity analysis, defaults to 1 (no scaling)
        self.scale = OrderedDict()
        for param in self.scale_params:
            self.scale[param] = 1.0
        self.scale_old = self.scale.copy()

        # If the user supplied a TS, this should be None.
        self.dG_act = dG_act

        # If all reactants are Liquid species, then this reaction can occur
        # at any point of the diffusion grid, not just near the catalyst
        # surface.
        self.all_liquid = True

        # Count up the number of Fluid and Adsorbate species on either side
        # of the reaction. This is necessary to construct a proper Jacobian
        # for the rate of change of Fluid species vs Adsorbate species.
        # The Jacobian is related to the concentration in M of catalytic sites
        # in the model (defaults to 1 M in the Model class).
        self.Nreact_fluid = 0
        self.Nreact_ads = 0
        for species in self.reactants:
            if not isinstance(species, Liquid):
                self.all_liquid = False
            if isinstance(species, _Fluid):
                self.Nreact_fluid += 1
            elif isinstance(species, Adsorbate):
                self.Nreact_ads += 1

        self.Nprod_fluid = 0
        self.Nprod_ads = 0
        for species in self.products:
            if not isinstance(species, Liquid):
                self.all_liquid = False
            if isinstance(species, _Fluid):
                self.Nprod_fluid += 1
            elif isinstance(species, Adsorbate):
                self.Nprod_ads += 1

        self.Nfluid = self.Nreact_fluid + self.Nprod_fluid
        self.Nads = self.Nreact_ads + self.Nprod_ads

        self.dground = dground

    def get_scale(self, param):
        try:
            return self.scale[param]
        except KeyError:
            print("{} is not a valid scaling parameter name!".format(param))
            return None

    def set_scale(self, param, value):
        try:
            self.scale[param] = value
        except KeyError:
            print("{} is not a valid scaling parameter name!".format(param))

    def update(self, T=None, Asite=None, L=None, force=False):
        if not force and not self.is_update_needed(T, Asite, L):
            return

        for species in self.species:
            species.update(T=T, force=True)

        self.T = T
        self.Asite = Asite
        self.L = L
        self.dH = self.products.get_H(T) - self.reactants.get_H(T)
#        self.dH *= self.scale['dH']
        self.dS = self.products.get_S(T) - self.reactants.get_S(T)
#        self.dS *= self.scale['dS']
        self.dG = self.dH - self.T * self.dS
        if self.ts is not None:
            for species in self.ts:
                species.update(T=T, force=True)

            Gts = self.ts.get_G(T)
            Gr = self.reactants.get_G(T)
            Gp = self.products.get_G(T)

            dEr = np.sum([species.lateral + species.dE for species in self.reactants])
            dEp = np.sum([species.lateral + species.dE for species in self.products])

            dGf = Gts - Gr + dEr
            dGr = Gts - Gp + dEp

            if dGf < 0:
                raise RuntimeError('Reaction {} has negative forwards activation barrier!'.format(self))
            if dGr < 0:
                raise RuntimeError('Reaction {} has negative reverse activation barrier!'.format(self))

            all_symbols = set()
            all_symbols.update(sym.sympify(dEr).atoms(sym.Symbol))
            all_symbols.update(sym.sympify(dEp).atoms(sym.Symbol))

            if sym.sympify(dEp - dEr).subs({symbol: 0 for symbol in all_symbols}) == 0:
                self.alpha = dGf / (dGf + dGr)
            else:
                a1 = (2*dEp - 2*dEr - dGf - dGr - sym.sqrt(8*(dEp-dEr)*dGf + (-2*dEp + 2*dEr + dGf + dGr)**2))/(4*(dEp-dEr))
                a1 = sym.sympify(a1).subs({symbol: 0 for symbol in all_symbols})
                if isinstance(a1, sym.Float) and 0. <= a1 <= 1:
                    self.alpha = a1
                else:
                    a2 = (2*dEp - 2*dEr - dGf - dGr + sym.sqrt(8*(dEp-dEr)*dGf + (-2*dEp + 2*dEr + dGf + dGr)**2))/(4*(dEp-dEr))
                    self.alpha = sym.sympify(a2).subs({symbol: 0 for symbol in all_symbols})
                    if not isinstance(self.alpha, sym.Float) or not (0. <= self.alpha <= 1.):
                        raise RuntimeError("Couldn't find alpha parameter for {}!".format(self))

            self.dH_act = self.ts.get_H(T) + (1 - self.alpha) * dEr + self.alpha * dEp - self.reactants.get_H(T)
            self.dH_act *= self.scale['dH_act']
            self.dS_act = self.ts.get_S(T) - self.reactants.get_S(T)
            self.dS_act *= self.scale['dS_act']
            self.dG_act = self.dH_act - self.T * self.dS_act

            # If there is a coverage dependence, assume everything has
            # coverage 0
            if self.dground:
                dG_act = self.dG_act
                if isinstance(dG_act, sym.Basic):
                    subs = {}
                    for atom in dG_act.atoms(sym.Symbol):
                        subs[atom] = 0.
                    dG_act = dG_act.subs(subs)

                if dG_act < 0.:
                    warnings.warn('Negative activation energy found for {}. '
                                  'Rounding to 0.'.format(self),
                                  RuntimeWarning, stacklevel=2)
                    self.dG_act = 0.

                dG_rev = self.dG_act - self.dG
                if isinstance(dG_rev, sym.Basic):
                    subs = {}
                    for atom in dG_rev.atoms(sym.Symbol):
                        subs[atom] = 0.
                    dG_rev = dG_rev.subs(subs)

                if dG_rev < 0.:
                    warnings.warn('Negative activation energy found for {}. '
                                  'Rounding to {}'.format(self, self.dG),
                                  RuntimeWarning, stacklevel=2)
                    self.dG_act = self.dG
        self._calc_keq()
        self._calc_kfor()
        self._calc_krev()
        self.scale_old = self.scale.copy()

    def is_update_needed(self, T, Asite, L):
        for species in self.species:
            if species.is_update_needed(T):
                return True
        if self.keq is None:
            return True
        if T is not None and T != self.T:
            return True
        if Asite is not None and Asite != self.Asite:
            return True
        if L is not None and L != self.L:
            return True
        for param in self.scale_params:
            if self.scale[param] != self.scale_old[param]:
                return True
        return False

    def get_keq(self, T=None, Asite=None, L=None):
        self.update(T, Asite, L)
        return self.keq

    def get_kfor(self, T=None, Asite=None, L=None):
        self.update(T, Asite, L)
        return self.kfor

    def get_krev(self, T=None, Asite=None, L=None):
        self.update(T, Asite, L)
        return self.krev

    def _calc_keq(self):
        self.keq = sym.exp(-self.dG / (kB * self.T)) \
                * self.products.get_reference_state() \
                / self.reactants.get_reference_state() \
                * self.scale['kfor'] / self.scale['krev']

    def _calc_kfor(self):
        barr = 1
        if self.dG_act is not None:
            barr *= sym.exp(-self.dG_act / (kB * self.T)) \
                    / self.reactants.get_reference_state()
#                    * self.ts.get_reference_state() \
        if self.method == 'EQUIL':
            self.kfor = _k * self.T * barr / _hplanck * self.scale['kfor']
            if isinstance(self.keq, sym.Basic):
                subs = {}
                for atom in self.keq.atoms(sym.Symbol):
                    subs[atom] = 0.
                keq = self.keq.subs(subs)
            else:
                keq = self.keq
            if keq < 1:
                self.kfor *= self.keq * self.scale['krev'] / self.scale['kfor']
        elif self.method == 'DIEQUIL':
            kfor1 = _k * self.T * barr / _hplanck * self.scale['kfor']
            kfor2 = kfor1 * self.keq * self.scale['krev'] / self.scale['kfor']
            self.kfor = kfor1 * kfor2 / (kfor1 + kfor2)
        elif self.method == 'STICK':
            # STICK is TST with the transition state being a non-interacting
            # 2D ideal gas.
            found_fluid = False
            for species in self.reactants:
                if isinstance(species, _Fluid):
                    if found_fluid:
                        raise ValueError("At most one fluid "
                                         "can react with STICK!")
                    found_fluid = True
                    fluid = species
            Sfluid = fluid.get_S(self.T)
            fluid._calc_qtrans2D(self.T, self.Asite)
            Strans = Sfluid - fluid.S['elec'] - fluid.S['rot'] - fluid.S['vib']
            Slost = Strans / fluid.S['trans']
            dS = (fluid.S['trans2D'] - fluid.S['trans']) * Slost
            dG = fluid.E['trans2D'] - fluid.E['trans'] - self.T * dS
            self.kfor = barr * _k * self.T / _hplanck * np.exp(-dG / (kB * self.T))
            self.kfor *= self.scale['kfor']
        elif self.method == 'ER':
            # Collision Theory
            # kfor = S0 * Asite / (sqrt(2 * pi * m * kB * T))
            m_react = 0.
            for species in self.reactants:
                if isinstance(species, _Fluid):
                    m_react += species.atoms.get_masses().sum()
            kfor1 = barr * 1000 * self.S0 * _Nav * self.Asite \
                * np.sqrt(_k * self.T * kg / (2 * np.pi * m_react)) \
                * self.scale['kfor']

            m_prod = 0.
            for species in self.products:
                if isinstance(species, _Fluid):
                    m_prod = species.atoms.get_masses().sum()
            # FIXME: barr should be different for reverse reaction
            krev2 = barr * 1000 * self.S0 * _Nav * self.Asite \
                * np.sqrt(_k * self.T * kg / (2 * np.pi * m_prod)) \
                * self.scale['krev']
            kfor2 = self.keq * krev2
            self.kfor = kfor1 * kfor2 / (kfor1 + kfor2)
        elif self.method == 'DIFF':
            if self.L is None:
                raise ValueError("Must provide diffusion length "
                                 "for diffusion reactions!")
            found_fluid = False
            for species in self.reactants:
                if isinstance(species, _Fluid):
                    if found_fluid:
                        raise ValueError("Diffusion reaction must "
                                         "have exactly 1 fluid!")
                    found_fluid = True
                    D = species.D
            sites = 1
            for species in self.reactants:
                if isinstance(species, Adsorbate):
                    sites *= species.symbol
            for species in self.products:
                if not isinstance(species, (Adsorbate, Electron)):
                    raise ValueError("All products must be adsorbates "
                                     "in diffusion reaction!")
            self.kfor = 1000 * D * self.Asite * mol * barr \
                * self.scale['kfor'] / (self.L * sites)
        elif self.method == 'DIFF_LIQ':
            if len(self.reactants) != 2:
                raise ValueError("DIFF_LIQ rate only defined for reactions "
                                 "with exactly two reactants!")
            if not self.all_liquid:
                raise ValueError("DIFF_LIQ rate only defined for all-liquid "
                                 "phase reactions!")
            Rtot = self.reactants[0].R + self.reactants[1].R
            Dtot = self.reactants[0].D + self.reactants[1].D
            self.kfor = 4 * np.pi * Dtot * Rtot * 1e-10 * 1000 * _Nav
            self.kfor *= self.scale['kfor']
        elif self.method == 'TST':
            # Transition State Theory
            self.kfor = (_k * self.T / _hplanck) * barr * self.scale['kfor']
        else:
            raise ValueError("Method {} is not recognized!".format(
                self.method))

    def _calc_krev(self):
        self.krev = self.kfor / self.keq

    def __repr__(self):
        string = self.reactants.__repr__() + ' <-> '
        if self.ts is not None:
            string += self.ts.__repr__() + ' <-> '
        string += self.products.__repr__()
        return string


class Model(object):
    def __init__(self, T, Asite, z=0, lattice=None, reactor='CSTR', rhocat=1):
        self.reactions = OrderedDict()
        self._reactions = []
        self._species = []
        self.species = OrderedDict()
        self.vacancy = []
        self.vacspecies = OrderedDict()
        self.solvent = None
        self.fixed = []
        self.initialized = False
        self.U0 = None
        self.rhocat = rhocat

        self.T = T  # System temperature
        self.Asite = Asite  # Area of adsorption site
        self._z = z  # Diffusion length
        self.lattice = lattice
        self.reactor = reactor
        # SUNDAE-based solver attributes
        self.nvariables = 0          # will be set later when species are defined
        self.y0 = None               # initial variable values
        self.yp0 = None              # initial derivatives
        self.rates = []              # reaction rate expressions
        self.resfn = None            # DAE residual function
        self.solver = None           # SUNDAE solver object

    def add_reactions(self, reactions):
        # Set up list of reactions and species
        for name, reaction in reactions.items():
            assert isinstance(reaction, Reaction)
            if reaction in self._reactions:
                return
            self._reactions.append(reaction)
            self.reactions[name] = reaction
            for species in reaction.species:
                self._add_species(species)
            if reaction.ts is not None:
                for ts in reaction.ts:
                    ts.lattice = self.lattice
            reaction.update(T=self.T, Asite=self.Asite, L=self.z)

    def set_solvent(self, solvent):
        # Solvent will not diffuse even in diffusion system
        if solvent is not None:
            if self.solvent is not None:
                warnings.warn('Overriding old solvent {} with {}.'
                              ''.format(solvent, self.solvent),
                              RuntimeWarning, stacklevel=2)
            if not isinstance(self.species[solvent], Liquid):
                raise ValueError("Solvent must be a Liquid!")
            self.solvent = solvent

    def set_fixed(self, fixed):
        # Fixed species are removed from the differential equations
        if isinstance(fixed, str):
            fixed = [fixed]
        for name in fixed:
            if name not in self.fixed:
                self.fixed.append(name)

    def _add_species(self, species):
        assert isinstance(species, _Thermo)
        # Do nothing if we already know about the species
        if species in self._species or species in self.vacancy:
            return
        # Add the species to the list of known species
        species.lattice = self.lattice
        self.species[species.label] = species
        self._species.append(species)
        # Add the sites that species occupies to the list of known vacancies.
        if species.sites is not None:
            for site in species.sites:
                if site not in self.vacancy:
                    if site in self._species:
                        self._species.remove(site)
                    self.vacancy.append(site)
                    self.vacspecies[site] = [species]
                else:
                    self.vacspecies[site].append(species)

    def set_T(self, T):
        self._T = T
        for reaction in self._reactions:
            reaction.update(T=T, Asite=self.Asite)
        if self.U0 is not None:
            self.set_initial_conditions(self.U0)

    def get_T(self):
        return self._T

    T = property(get_T, set_T, doc='Model temperature')

    def set_Asite(self, Asite):
        self._Asite = Asite
        for reaction in self._reactions:
            reaction.update(T=self.T, Asite=Asite)
        if self.U0 is not None:
            self.set_initial_conditions(self.U0)

    def get_Asite(self):
        return self._Asite

    Asite = property(get_Asite, set_Asite, doc='Area of an adsorption site')

    def set_z(self, z):
        self._z = z
        self.check_diffusion()
        for reaction in self._reactions:
            reaction.update(L=z)
        if self.U0 is not None:
            self.set_initial_conditions(self.U0)

    def get_z(self):
        return self._z

    z = property(get_z, set_z, doc='Diffusion length')

    def set_lattice(self, lattice):
        if isinstance(lattice, Lattice) or lattice is None:
            self._lattice = lattice
        elif isinstance(lattice, dict):
            self._lattice = Lattice(lattice)
        else:
            raise ValueError('Unable to parse lattice!')
        for species in self._species:
            species.set_lattice(self.lattice)
        if self.U0 is not None:
            self.set_initial_conditions(self.U0)

    def get_lattice(self):
        return self._lattice

    lattice = property(get_lattice, set_lattice, doc='Model lattice')

    def set_initial_conditions(self, U0):
        if self.initialized:
            self.finalize()

        # Reorder species such that Liquid -> Gas -> Adsorbate -> Vacancy
        # Steady-state species go to the end.
        newspecies = []
        for species in self._species:
            if isinstance(species, Liquid):
                newspecies.append(species)
        for species in self._species:
            if isinstance(species, (Gas, Electron)):
                newspecies.append(species)
        for species in self._species:
            if isinstance(species, Adsorbate):
                newspecies.append(species)
        self._species = newspecies

        # Also obtain a list of species that will be variables in the
        # differential equations. This excludes fixed species and empty
        # sites.
        self._variable_species = []
        for species in self._species:
            if species.label not in self.fixed + [self.solvent]:
                self._variable_species.append(species)
        self.nvariables = len(self._variable_species)

        # Start with the incomplete user-provided initial conditions
        self.U0 = U0.copy()

        # Initialize counter for vacancies
        occsites = {species: 0 for species in self.vacancy}

        for name in self.U0:
            try:
                species = self.species[name]
            except KeyError:
                for species in self.vacancy:
                    if species.label == name:
                        break
                else:
                    raise ValueError('Species {} is unknown!'.format(name))

            # Ignore all initial conditions for the number of empty sites
            if species in self.vacancy:
                warnings.warn('Initial condition for vacancy concentration '
                              'ignored.', RuntimeWarning, stacklevel=2)
                continue

            # Throw an error if the user provides the concentration for a
            # species we don't know about
            if species not in self._species:
                raise ValueError("Unknown species {}!".format(species))

            # If the species occupies a site, add its concentration to the
            # occupied sites counter
            if species.sites is not None:
                for site in species.sites:
                    occsites[site] += self.U0[name]

        self.dvacdy = np.zeros((len(self.vacancy), self.nvariables), dtype=int)
        self.vactot = {}
        # Determine what the initial vacancy concentration should be
        for i, vac in enumerate(self.vacancy):
            name = vac.label
            # If a vacancy species is part of the lattice, get its maximum
            # concentration from its relative abundance. Otherwise, assume
            # it is 1.
            if self.lattice is not None and vac in self.lattice.sites:
                self.vactot[vac] = self.lattice.ratio[vac]
            else:
                self.vactot[vac] = 1.
            # Make sure there isn't too much stuff occupying each kind of
            # site on the surface.
            assert occsites[vac] <= self.vactot[vac], \
                    "Too many adsorbates on {}!".format(vac)
            # Normalize the concentration of empty sites to match the
            # appropriate site ratio from the lattice.
            self.U0[name] = self.vactot[vac] - occsites[vac]
            for j, species in enumerate(self._variable_species):
                self.dvacdy[i, j] = -species.sites.count(vac)

        # Populate dictionary of initial conditions for all species
        for name, species in self.species.items():
            # Assume concentration of unnamed species is 0
            if name not in self.U0:
                self.U0[name] = 0.

        # The number of variables that will be in our differential equations
        size = len(self._species)

        # This creates a symbol for each species named modelparamX where X
        # is a three-digit numerical identifier that corresponds to its
        # position in the order of species
        self.symbols_all = []
        # symbols_dict a Thermo object and returns its corresponding symbol
        self.symbols_dict = OrderedDict()
        # symbols ONLY includes species that will be in the differential
        # equations. Fixed species are not included in this list
        self.symbols = []

        for species in self._species:
            self.symbols_all.append(species.symbol)
            self.symbols_dict[species] = species.symbol

        self.symbols = [species.symbol for species in self._variable_species]

        # subs converts a species symbol to either its initial value if
        # it is fixed or to a constraint (such as constraining the total
        # number of adsorption sites)
        subs = {}
        
        self.vac_sym = np.zeros(len(self.vacancy), dtype=object)
        # A vacancy will be represented by the total number of sites
        # minus the symbol of each species that occupies one of its sites.
        for i, vacancy in enumerate(self.vacancy):
            self.vac_sym[i] = self.vactot[vacancy]
            for species in self._species:
                self.vac_sym[i] -= species.sites.count(vacancy) * species.symbol

        # known_symbols keeps track of user-provided symbols that the
        # model has seen, so that symbols referring to species not in
        # the model can be later removed.
        known_symbols = set()
        for species in self._species + self.vacancy:
            known_symbols.add(species.symbol)

        # Create the final mass matrix of the proper dimensions
        self.M = np.eye(self.nvariables, dtype=int)
        
        if self.reactor == 'PFR':
            for i, species in enumerate(self._variable_species):
                if isinstance(species, Adsorbate):
                    self.M[i, i] = 0

        # algvar tells the solver which variables are differential
        # and which are algebraic. It is the diagonal of the mass matrix.
        algvar = np.array(self.M.diagonal(), dtype=float)

        # Initialize all rate expressions based on the above symbols
        nrxns = len(self._reactions)
        # Array of symbolic rate expressions
        self.rates = np.zeros(nrxns, dtype=object)
        # Array of rate coefficients.
        self.dypdr = np.zeros((self.nvariables, nrxns), dtype=float)

        for j, rxn in enumerate(self._reactions):
            rate_for = rxn.get_kfor(self.T, self.Asite, self.z)
            rate_rev = rxn.get_krev(self.T, self.Asite, self.z)

            for i, species in enumerate(self._variable_species):
                rcount = rxn.reactants.species.count(species)
                pcount = rxn.products.species.count(species)
                self.dypdr[i, j] = -rcount + pcount
                if isinstance(species, _Fluid) and rxn.involves_catalyst:
                    self.dypdr[i, j] *= self.rhocat

            for species in self._species + self.vacancy:
                rcount = rxn.reactants.species.count(species)
                pcount = rxn.products.species.count(species)
                if not isinstance(species, Electron):
                    rate_for *= species.symbol**rcount
                    rate_rev *= species.symbol**pcount

            # Overall reaction rate (flux)
            self.rates[j] = rate_for
            if rxn.reversible:
                self.rates[j] -= rate_rev


        # All symbols referring to unknown species are going to be replaced
        # by 0
        unknown_symbols = set()
        for rate in self.rates:
            unknown_symbols.update(rate.atoms(sym.Symbol))
        unknown_symbols -= known_symbols
        unknown_symbols -= set(self.symbols_all)
        subs.update({symbol: 0 for symbol in unknown_symbols})

        # Fixed species must have their symbols replaced by their fixed
        # initial values.
        for species in self._species:
            if species.label in self.fixed or species.label == self.solvent:
                label = species.label
                subs[species.symbol] = self.U0[label]

        # Additionally, fixed species concentrations into rate
        # expressions
        for i, r in enumerate(self.rates):
            self.rates[i] = sym.sympify(r).subs(subs)

        # derivative of rate expressions w.r.t. concentrations and vacancies
        self.drdy = np.zeros((nrxns, self.nvariables), dtype=object)
        self.drdvac = np.zeros((nrxns, len(self.vacancy)), dtype=object)
        for i, rate in enumerate(self.rates):
            for j, symbol in enumerate(self.symbols):
                self.drdy[i, j] = sym.diff(rate, symbol)
            for j, vac in enumerate(self.vacancy):
                self.drdvac[i, j] = sym.diff(rate, vac.symbol)
                
        # Convert to ordered initial arrays for SUNDAE
        self.y0 = np.array([self.U0[s.label] for s in self._variable_species])
        self.yp0 = np.zeros_like(self.y0)

        # Set tolerances
        self.atol = np.array([1e-32] * self.nvariables) + 1e-16 * algvar
        self.rtol = 1e-10

        # Initialize SUNDAE solver object
        self.solver = sun.ida.IDA()
        self.solver.init(self.resfn, 0.0, self.y0, self.yp0)
        self.solver.set_tolerances(self.rtol, self.atol)

        # If you want to handle algebraic variables (from M)
        self.solver.set_id(algvar)


        self.initialized = True

    def setup_execs(self):
        from micki.fortran import f90_template, pyf_template
        from numpy import f2py
        """
        Build numeric residual, rate, and Jacobian functions from SymPy expressions,
        and initialize a persistent sksundae.IDA solver attached to self.solver.

        This version lambdifies:
          - residuals: F(y, yp) = M*yp - dypdr * rates(y)
          - Jacobian: J_y = dF/dy (symbolic), and supplies jacfn that returns J = J_y + cj * (dF/dyp)
            where dF/dyp = M (mass matrix).
        """
        import sympy as sym
        import numpy as np
        import sksundae as sun
        import warnings

        # sanity check
        if not hasattr(self, "nvariables") or self.nvariables == 0:
            raise RuntimeError("nvariables must be set before calling setup_execs()")

        # 1) symbolic placeholders
        n = self.nvariables
        y_syms = sym.symbols(f'y0_0:{n}')
        yp_syms = sym.symbols(f'yp0_0:{n}')

        nv = len(self.vacancy)
        if nv > 0:
            vac_syms = sym.symbols(f'vac0_0:{nv}')
        else:
            vac_syms = ()

        # 2) map model symbols -> y placeholders
        subs_map = {}
        for i, s in enumerate(self.symbols):
            subs_map[s] = y_syms[i]
        for i, v in enumerate(self.vacancy):
            # vac_sym expressions (constructed earlier) are in terms of species.symbol,
            # which we map to vac_syms placeholders first; later we'll substitute vac_syms
            # back into y via vac_sym_list which uses y_syms.
            subs_map[v.symbol] = vac_syms[i] if nv > 0 else sym.Integer(0)

        # 3) prepare symbolic rate expressions and vac expressions substituted to placeholders
        rate_sym_list = [sym.sympify(r) for r in self.rates]  # ensure sympy objects
        rate_sub = [r.subs(subs_map) for r in rate_sym_list]

        vac_sym_list = []
        if nv > 0:
            for vexpr in self.vac_sym:
                vac_sym_list.append(sym.sympify(vexpr).subs(subs_map))

        # 4) because vac_sym_list is expressed in terms of species symbols replaced by y_syms,
        # we should now express the rate_sub in terms of y_syms by substituting vac_syms -> vac_sym_list
        if nv > 0:
            rate_for_lamb_sym = [r.subs({vac_syms[i]: vac_sym_list[i] for i in range(nv)}) for r in rate_sub]
        else:
            rate_for_lamb_sym = rate_sub

        # 5) build residual symbolic expressions: F_i = sum_j M[i,j] * yp_j - sum_k dypdr[i,k] * rate_k(y)
        M = np.array(self.M, dtype=float)
        dypdr = np.array(self.dypdr, dtype=float)
        nrates = len(rate_for_lamb_sym)

        residual_sym = []
        for i in range(n):
            mass_term = sum(int(M[i, j]) * yp_syms[j] for j in range(n))
            flux_term = sym.Integer(0)
            for k in range(nrates):
                coeff = dypdr[i, k]
                if coeff == 0:
                    continue
                flux_term += sym.sympify(float(coeff)) * rate_for_lamb_sym[k]
            residual_sym.append(mass_term - flux_term)

        # 6) Jacobian wrt y_syms: J_y = dF/dy
        # Note: dF/dyp = M (constant); J used by IDA is J = dF/dy + cj * dF/dyp
        try:
            J_sym = sym.Matrix(residual_sym).jacobian(list(y_syms))
        except Exception as e:
            raise RuntimeError("Failed to build symbolic Jacobian: " + str(e))

        # 7) Lambdify residuals and Jacobian
        # residuals: function of (y_syms, yp_syms) -> residual vector
        arg_res = list(y_syms) + list(yp_syms)
        try:
            resid_lamb = sym.lambdify(arg_res, residual_sym, modules="numpy")
        except Exception as e:
            raise RuntimeError("Failed to lambdify residual expressions: " + str(e))

        # jacobian lambdify: function of y_syms -> matrix
        try:
            jac_lamb = sym.lambdify(list(y_syms), J_sym, modules="numpy")
        except Exception as e:
            raise RuntimeError("Failed to lambdify Jacobian expressions: " + str(e))

        # rates lambdify (for postprocessing)
        try:
            rate_lamb = sym.lambdify(list(y_syms), rate_for_lamb_sym, modules="numpy")
        except Exception as e:
            raise RuntimeError("Failed to lambdify rate expressions: " + str(e))

        # 8) define Python callback wrappers
        def resfn(t, y_arr, yp_arr, res_out):
            # pack args in order expected by resid_lamb
            y_flat = np.asarray(y_arr).ravel()
            yp_flat = np.asarray(yp_arr).ravel()
            args = tuple(list(y_flat) + list(yp_flat))
            rvals = resid_lamb(*args)
            rnp = np.asarray(rvals, dtype=float).ravel()
            res_out[:] = rnp
            return 0

        # Jacobian function expected by IDA: J = dF/dy + cj * dF/dyp
        # We'll implement jacfn(t, y, yp, cj, J_out) which fills J_out in-place.
        M_sym = np.array(M, dtype=float)  # dF/dyp
        def jacfn(t, y_arr, yp_arr, cj, J_out):
            # compute J_y
            y_flat = np.asarray(y_arr).ravel()
            J_y = np.asarray(jac_lamb(*list(y_flat)), dtype=float)
            # combine
            J_comb = J_y + float(cj) * M_sym
            # copy into provided output (assume shape matches (n, n))
            J_out[:, :] = J_comb
            return 0

        # expose functions on self
        self.resfn = resfn
        self.jacfn = jacfn
        self.rate_func = lambda y: np.asarray(rate_lamb(*list(np.asarray(y).ravel())), dtype=float).ravel()

        # 9) prepare y0 / yp0 and solver initialization (tolerances & algvar)
        if getattr(self, "y0", None) is None:
            self.y0 = np.array([self.U0[s.label] for s in self._variable_species], dtype=float)
        if getattr(self, "yp0", None) is None:
            self.yp0 = np.zeros_like(self.y0)

        algvar = np.array(self.M.diagonal(), dtype=float)
        atol = np.array([1e-32] * n, dtype=float) + 1e-16 * algvar
        rtol = 1e-10

        # 10) create solver and attach jacfn as supported
        created = False
        # Try passing jacfn in constructor first (some sksundae versions accept this)
        try:
            self.solver = sun.ida.IDA(resfn=self.resfn, jacfn=self.jacfn, neq=n)
            created = True
        except Exception:
            try:
                self.solver = sun.ida.IDA(resfn=self.resfn, neq=n)
                created = True
                # try to register jacfn via setter(s) if available
                if hasattr(self.solver, "set_jacobian"):
                    try:
                        self.solver.set_jacobian(self.jacfn)
                    except Exception:
                        pass
                if hasattr(self.solver, "set_jacfn"):
                    try:
                        self.solver.set_jacfn(self.jacfn)
                    except Exception:
                        pass
            except Exception as e:
                raise RuntimeError("Could not create sksundae.IDA solver: " + str(e))

        # try defensive init/tolerance/id setting
        try:
            if hasattr(self.solver, "init"):
                try:
                    # some wrappers expect (resfn, t0, y0, yp0)
                    self.solver.init(self.resfn, 0.0, self.y0, self.yp0)
                except Exception:
                    # maybe constructor already initialized; ignore
                    pass
            if hasattr(self.solver, "set_tolerances"):
                try:
                    self.solver.set_tolerances(rtol, atol)
                except Exception:
                    pass
            if hasattr(self.solver, "set_id"):
                try:
                    self.solver.set_id(algvar)
                except Exception:
                    pass
        except Exception:
            warnings.warn("Could not fully configure sksundae solver (tolerances/algvar). "
                          "Check API and adjust calls.", RuntimeWarning)

        # compatibility shim: simple fsolve wrapper that mimics old behaviour
        def py_fsolve(tspan, y0=None, yp0=None):
            y0_local = self.y0 if y0 is None else np.asarray(y0)
            yp0_local = self.yp0 if yp0 is None else np.asarray(yp0)
            sol = self.solver.solve(tspan, y0_local, yp0_local)
            return sol

        self.fsolve = py_fsolve
        self.initialized = True

    def _out_array_to_dict(self, U, dU, r):
        Ui = {}
        dUi = {}
        ri = {}
        fixed = self.fixed
        if self.solvent is not None:
            fixed += [self.solvent]
        for name in fixed:
            dUi[name] = 0.
            Ui[name] = self.U0[name]
        for j, symbol in enumerate(self.symbols):
            for species, isymbol in self.symbols_dict.items():
                if symbol == isymbol:
                    Ui[species.label] = U[j]
                    dUi[species.label] = dU[j]
        for vacancy in self.vacancy:
            Ui[vacancy.label] = self.vactot[vacancy]
            for species in self.vacspecies[vacancy]:
                Ui[vacancy.label] -= Ui[species.label]
            dUi[vacancy.label] = 0

        j = 0
        rxn_to_name = {}
        for name, reaction in self.reactions.items():
            rxn_to_name[reaction] = name
        for reaction in self._reactions:
            ri[rxn_to_name[reaction]] = r[j]
            j += 1

        return Ui, dUi, ri

    def find_steady_state(self, dt=60, maxiter=2000, epsilon=1e-8):
        if not getattr(self, "initialized", False):
            # Ensure the solver is prepared
            self.setup_execs()

        # initial t, y, yp
        t = 0.0
        y = self.y0.copy()
        yp = self.yp0.copy()

        # preallocate residual array
        res = np.zeros_like(y)

        converged = False
        it = 0

        while not converged and it < maxiter:
            t_next = t + dt
            # call solver to advance to t_next; provide y, yp as initial guesses
            sol = self.solver.solve([t, t_next], y, yp)

            # solver.solve may return an object or arrays; handle common forms
            # prefer returning last timepoint solution
            try:
                t_vals = np.asarray(sol.t)
                y_vals = np.asarray(sol.y)
                yp_vals = np.asarray(sol.yp) if hasattr(sol, "yp") else None
                # sol.y shape often (nvars, n_times)
                y = y_vals[:, -1] if y_vals.ndim == 2 else y_vals
                if yp_vals is not None:
                    yp = yp_vals[:, -1] if yp_vals.ndim == 2 else yp_vals
                else:
                    # if solver didn't provide yp, try to estimate via finite difference or leave prior yp
                    yp = yp
                t = t_vals[-1] if t_vals.ndim >= 1 else float(t_next)
            except Exception:
                # fallback: solver might have returned (t_out, y_out, yp_out)
                try:
                    t, y_full, yp_full = sol
                    y = np.asarray(y_full).ravel()[:, -1] if np.asarray(y_full).ndim == 2 else np.asarray(y_full).ravel()
                    yp = np.asarray(yp_full).ravel()[:, -1] if np.asarray(yp_full).ndim == 2 else np.asarray(yp_full).ravel()
                except Exception:
                    raise RuntimeError("Unexpected solver.solve return type. Inspect sol object.")

            # compute residuals using resfn
            self.resfn(t, y, yp, res)

            # For convergence check: either check derivative yp or residual res
            if np.max(np.abs(yp)) < epsilon or np.max(res**2) < epsilon**2:
                converged = True
                break

            it += 1

        if not converged:
            print("WARNING: did not converge to steady state within maxiter steps")

        # compute rates (post-processing) and convert outputs to your dict format
        rates_vals = self.rate_func(y)

        # _out_array_to_dict in original code consumed arrays shaped (nvars, ntime)
        # Here we will reuse your helper if present; otherwise build simple dict mapping species -> value
        try:
            # your original used U1.T shape etc; replicate similar behaviour
            U_dict, dU_dict, r_dict = self._out_array_to_dict(y.reshape(-1, 1).T.T,
                                                             yp.reshape(-1, 1).T.T,
                                                             rates_vals.reshape(-1, 1).T.T)
            # above keeps backward compatibility with your _out_array_to_dict call sites
        except Exception:
            # fallback: map variable species labels to values
            U_dict = {s.label: float(v) for s, v in zip(self._variable_species, y)}
            dU_dict = {s.label: float(v) for s, v in zip(self._variable_species, yp)}
            # rates: map by index (no names unless you have reaction labels)
            r_dict = {i: float(rv) for i, rv in enumerate(rates_vals)}

        # store results on object like original method did
        self.t = t
        self.U = [U_dict]
        self.dU = [dU_dict]
        self.r = [r_dict]

        # optional check_rates hook
        if hasattr(self, "check_rates"):
            try:
                self.check_rates(U_dict)
            except Exception:
                pass

        return t, U_dict, r_dict
    def newton_steady_state(self, y0=None, tol=1e-10, maxiter=50,
                           lin_solver='direct', damping=True,
                           bt_maxiter=10, bt_reduce=0.5, verbose=False):
        """
        Solve F(y, yp=0) = 0 with Newton's method using the analytic Jacobian.

        Parameters
        ----------
        y0 : array_like or None
            Initial guess for the variable vector. If None, uses self.y0.
        tol : float
            Convergence tolerance on the residual ||F||_inf.
        maxiter : int
            Maximum Newton iterations.
        lin_solver : {'direct','lstsq'}
            Linear solver used for J * delta = -F. 'direct' uses np.linalg.solve,
            'lstsq' uses np.linalg.lstsq for near-singular J.
        damping : bool
            If True, perform simple backtracking line search on the Newton step.
        bt_maxiter : int
            Maximum backtracking reductions.
        bt_reduce : float
            Multiplicative step-length reduction factor (0<bt_reduce<1).
        verbose : bool
            Print iteration diagnostics.

        Returns
        -------
        y_ss : ndarray
            The steady-state vector (solution).
        res_dict : mapping
            Reaction rates or other post-processed rates (like your existing code returns).
        Raises
        ------
        RuntimeError if solver infrastructure isn't present.
        """

        import numpy as np

        # sanity checks
        if not getattr(self, "initialized", False):
            # ensure setup_execs has been called
            try:
                self.setup_execs()
            except Exception as e:
                raise RuntimeError("Model must be initialized (call setup_execs) before Newton solve: " + str(e))

        n = int(self.nvariables)
        # initial guess
        if y0 is None:
            if getattr(self, "y0", None) is None:
                raise RuntimeError("No initial guess available (self.y0 is None). Provide y0.")
            y = self.y0.astype(float).copy()
        else:
            y = np.asarray(y0, dtype=float).ravel().copy()
            if y.size != n:
                raise ValueError("y0 length mismatch: expected {}, got {}".format(n, y.size))

        # zero derivative for steady-state
        yp_zero = np.zeros(n, dtype=float)

        # helper: evaluate residual F(y,0) into array
        def eval_res(y_vec):
            r = np.zeros(n, dtype=float)
            # resfn signature: resfn(t, y, yp, res_out)
            # pass t=0.0 (time not relevant for steady-state)
            self.resfn(0.0, y_vec, yp_zero, r)
            return r

        # helper: evaluate Jacobian dF/dy (analytic) by calling jacfn with cj=0
        def eval_Jy(y_vec):
            J = np.zeros((n, n), dtype=float)
            # jacfn signature (as provided in setup_execs): jacfn(t, y, yp, cj, J_out)
            # passing cj=0 yields J = dF/dy + 0 * dF/dyp = dF/dy
            try:
                # Some wrappers return int status; jacfn fills J_out
                self.jacfn(0.0, y_vec, yp_zero, 0.0, J)
            except Exception as e:
                # If jacfn isn't present or fails, raise and suggest alternate approach
                raise RuntimeError("Could not evaluate analytic Jacobian via self.jacfn: " + str(e))
            return J

        # Newton iterations
        res = eval_res(y)
        res_norm = np.max(np.abs(res))
        if verbose:
            print(f"Newton init: ||F||_inf = {res_norm:.3e}")
        if res_norm < tol:
            # already converged
            rates = self.rate_func(y) if hasattr(self, "rate_func") else None
            return y, rates

        for it in range(1, maxiter + 1):
            J = eval_Jy(y)

            # Solve linear system J * delta = -res
            b = -res
            try:
                if lin_solver == 'direct':
                    delta = np.linalg.solve(J, b)
                elif lin_solver == 'lstsq':
                    delta, *_ = np.linalg.lstsq(J, b, rcond=None)
                else:
                    raise ValueError("Unknown lin_solver '{}'".format(lin_solver))
            except np.linalg.LinAlgError:
                # fallback to least-squares
                delta, *_ = np.linalg.lstsq(J, b, rcond=None)

            # damping/backtracking line search: reduce alpha until residual decreases
            alpha = 1.0
            y_trial = y + alpha * delta
            res_trial = eval_res(y_trial)
            res_trial_norm = np.max(np.abs(res_trial))

            if damping:
                bt = 0
                # accept step if residual decreases sufficiently (here: smaller max-norm)
                while res_trial_norm >= res_norm and bt < bt_maxiter:
                    alpha *= bt_reduce
                    y_trial = y + alpha * delta
                    res_trial = eval_res(y_trial)
                    res_trial_norm = np.max(np.abs(res_trial))
                    bt += 1
                if verbose and bt > 0:
                    print(f"  backtrack: it={it}, bt={bt}, alpha={alpha:.3e}, ||F||_inf={res_trial_norm:.3e}")

            # accept trial
            y = y_trial
            res = res_trial
            res_norm_new = res_trial_norm

            if verbose:
                print(f"Newton it {it:3d}: ||F||_inf = {res_norm_new:.3e}, ||delta||_inf = {np.max(np.abs(delta)):.3e}")

            if res_norm_new < tol:
                if verbose:
                    print("Newton converged.")
                break

            # prepare for next iteration
            res_norm = res_norm_new
        else:
            # loop exhausted
            raise RuntimeError(f"Newton did not converge within {maxiter} iterations; final ||F||_inf = {res_norm:.3e}")

        # post-processing: compute rates & convert to your dict format like find_steady_state did
        rates_vals = self.rate_func(y) if hasattr(self, "rate_func") else None

        # convert y into your U dict
        try:
            U_dict, dU_dict, r_dict = self._out_array_to_dict(y.reshape(-1, 1).T.T,
                                                             yp_zero.reshape(-1, 1).T.T,
                                                             rates_vals.reshape(-1, 1).T.T)
        except Exception:
            U_dict = {s.label: float(v) for s, v in zip(self._variable_species, y)}
            dU_dict = {s.label: 0.0 for s in self._variable_species}
            if rates_vals is None:
                r_dict = {}
            else:
                r_dict = {i: float(rv) for i, rv in enumerate(np.atleast_1d(rates_vals))}

        # store same fields as find_steady_state
        self.t = 0.0
        self.U = [U_dict]
        self.dU = [dU_dict]
        self.r = [r_dict]

        # optional check_rates hook
        if hasattr(self, "check_rates"):
            try:
                self.check_rates(U_dict)
            except Exception:
                pass

        return y, r_dict

    def solve(self, t, ncp):
        self.t, U1, dU1, r1 = self.fsolve(self.nvariables,
                                          len(self.rates), ncp, t)
        self.U1 = U1.T
        self.dU1 = dU1.T
        self.r1 = r1.T
        self.U = []
        self.dU = []
        self.r = []
        for i, t in enumerate(self.t):
            Ui, dUi, ri = self._out_array_to_dict(self.U1[i], self.dU1[i],
                                                  self.r1[i])
            self.U.append(Ui)
            self.dU.append(dUi)
            self.r.append(ri)
        self.check_rates(self.U[-1])
        return self.U, self.r

    def finalize(self):
        self.initialized = False
#        self.ffinalize()

    def check_rates(self, U, epsilon=1e-6):
        symbol_to_coverage = {}
        for name, Ui in U.items():
            if name in self.species and self.species[name].symbol is not None:
                symbol_to_coverage[self.species[name].symbol] = Ui

        for species in self.vacancy:
            if species.label in U and species.symbol is not None:
                symbol_to_coverage[species.symbol] = U[species.label]

        for name, reaction in self.reactions.items():
            kfor = sym.sympify(reaction.kfor).subs(symbol_to_coverage)
            krev = sym.sympify(reaction.krev).subs(symbol_to_coverage)
            kmax = _k * self.T / _hplanck
            for k, word in [(kfor, "Forwards"), (krev, "Reverse")]:
                ratio = k / kmax
                if (ratio - 1.0) > 1e-6:
                    warnings.warn(word + " rate constant for {} is too large! "
                                  "Value is {} kB T / h (should be <= 1)."
                                  "".format(reaction, ratio),
                                  RuntimeWarning, stacklevel=2)

    def copy(self, initialize=True):
        newmodel = Model(self.T, self.Asite, self.z, self.lattice, self.rhocat)
        newmodel.add_reactions(self.reactions)
        newmodel.set_fixed(self.fixed)
        newmodel.set_solvent(self.solvent)
        if initialize:
            newmodel.set_initial_conditions(self.U0)
        return newmodel
