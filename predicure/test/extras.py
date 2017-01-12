#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Some random tests for the semimarkov module.

These are interactive 'tests'. The require an interactive Python console
such as a Jupyter Notebook, QtConsole, IPython console, Hydrogen session
within Atom, or an nteract Python Notebook.
'''

# Copyright © 2017 Carson J. Q. Farmer <carson@set.gl>
# All rights reserved. BSD-2 Licensed.

from semimarkov import SemiMarkov
from scipy.stats import poisson, uniform, norm, gaussian_kde
import pykov  # This is really just to generate the simulation data...


def simulate(T=7*24*60):
    '''Not all that realistic semi-Markov model simulation.'''
    C = pykov.Chain({('H','W'): .70, ('H','H'): .00, ('H','O'): .30,
                     ('W','H'): .70, ('W','W'): .00, ('W','O'): .30,
                     ('O','O'): .10, ('O','W'): .45, ('O','H'): .45})
    waits = {'H': 11*60, 'W': 9*60, 'O': 4*60}
    prev_ = 'H'
    dur = poisson.rvs(waits[prev_])
    t = dur
    while t < T:
        next_ = C.move(prev_)
        yield [(prev_, next_, dur)]
        prev_ = next_
        dur = poisson.rvs(waits[prev_])
        t += dur
    yield [(prev_, next_, dur)]

t = SemiMarkov(False)
sim = list(simulate(24*60*30))
t.update_all(sim)
t.query([("H", "W", 663)], True)

# t.transitions["H"]["W"].holding.print_breaks(20)

t.transitions["H"]["W"].duration_density(662)
t.transition_probability("H", "W")

vals = [b.value for b in t.transitions["H"]["W"].holding.bins]

# 'H': 11*60, 'W': 9*60, 'O': 4*60
# Compare to what we get from a kernel density estimator and...
# the theoretical Poisson version
gaussian_kde(vals)(662)
poisson(11*60).pmf(662)

# What does the 'pure' probabilistic version say?
assert t.query([("H", "W", 663)], True)[0] - (poisson(11*60).pmf(663) * .7) < 0.005

# Let's plot it for shits and giggles
f = t.plot()
f.savefig("test.png")

t.transitions['H']['W'].holding.to_dict()

print(t)
