#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A sketch-based semi-Markov learner with approximate duration estimation.

Testing module for predicure.
"""

# Copyright © 2017 Carson J. Q. Farmer <carson@set.gl>
# All rights reserved. BSD-2 Licensed.

from __future__ import print_function, division, absolute_import

import random
import pytest
from scipy.stats import poisson
from addc import AddC, Centroid, KernelCentroid
from math import isinf, isnan

from predicure import MarkovSketch

import numpy as np

# Only works on *nix systems
word_file = "/usr/share/dict/words"
WORDS = open(word_file).read().splitlines()

def contains_same(s, t):
    s, t = set(s), set(t)
    return s >= t and s <= t


def all_close(s, t, tol=1e-8):
    # Ignores inf and nan values...
    return all(abs(a - b) < tol for a, b in zip(s, t)
               if not isinf(a) and not isinf(b) and
                  not isnan(a) and not isnan(b))

@pytest.fixture(scope="module")
def Events(state_count=10):
    '''Return sequence of random 'states' with random 'times'.

    This is really lazy/messy, but it works for now.
    '''
    states = [random.choice(WORDS) for _ in range(state_count)]
    # Random holding times between 12 hours and 1 hour
    holds = [random.randint(1*60, 5*60) for _ in range(state_count)]
    total_time = 0
    events = []
    while total_time < 60*72: # 72 hours for now
        i = random.randint(0, len(states)-1)
        dwell = poisson.rvs(holds[i])
        state = states[i]
        time = 0
        while time < dwell:
            events.append((state, total_time))
            step = random.gauss(0.5, .05)  # Basicaly one event every 30 seconds
            time += step
            total_time += step
    return events


class TestMarkovSketch:
    """Main test class."""

    def test_init(self):
        ac = MarkovSketch()
        assert len(ac.states) < 1
        # More asserts need to be added here

    def test_add(self):
        ps = Events()
        ac = MarkovSketch()
        for p in ps:
            ac += p
        assert len(ac) == len(set(i[0] for i in ps))
        assert ac.npoints == len(ps)
        # More asserts need to be added here

    def test_batch(self):
        ps = Events()
        ac = MarkovSketch()
        assert len(ac) == 0
        ac.batch(ps)
        assert ac.npoints == len(ps)
        assert len(ac) == len(set(i[0] for i in ps))

    def test_compare_batch_add(self):
        ps = Events()
        ac1 = MarkovSketch()
        ac2 = MarkovSketch()
        for p in ps:
            ac1 += p
        ac2.batch(ps)
        for a, b in zip(ac1.states, ac2.states):
            assert contains_same(a, b)
        for a, b in zip(ac1, ac2):
            assert a[0] == b[0]
            assert a[1].keys() == b[1].keys()

    def test_len(self):
        ps = Events()
        ac = MarkovSketch()
        assert len(ac) == 0
        ac.batch(ps)
        print(ac.states)
        assert len(ac) == len(set(i[0] for i in ps))

    def test_contains(self):
        ps = Events()
        ac = MarkovSketch()
        assert ps[0][0] not in ac
        ac.batch(ps)
        assert ps[0][0] in ac

    def test_call_and_states(self):
        ps = Events()
        ac = MarkovSketch()
        # assert len(ac()) == len(ac.states) == 0
        ac.batch(ps)
        # assert len(ac()) == len(ac.states) == len(set(i[0] for i in ps))

    def test_iter(self):
        ps = Events()
        ac = MarkovSketch().batch(ps)
        assert hasattr(ac, '__iter__')
        assert ac.npoints == len(ps)

    def test_npoints(self):
        ps = Events()
        ac = MarkovSketch().batch(ps)
        assert ac.npoints == len(ps)
        assert ac.npoints > len(ac)
        assert isinstance(ac.npoints, int)

    def test_durations(self):
        pass
