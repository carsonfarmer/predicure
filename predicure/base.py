#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A sketch-based semi-Markov learner with approximate duration estimation."""

# Copyright © 2017 Carson J. Q. Farmer <carson@set.gl>
# Some ideas borrowed from https://github.com/aijunbai/user-modelling
# Copyright © 2015 Aijun Bai, Alibaba Inc.
# All rights reserved. BSD-2 Licensed.

from __future__ import division

from collections import defaultdict
from functools import partial, reduce, lru_cache
import copy
from math import sqrt

from streamhist import StreamHist

_all__ = ["SemiMarkov"]

EPS = 1e-6


def linspace(start, stop, num=50):
    '''Return evenly spaced numbers over a specified interval.

    Parameters
    ----------
    start : scalar
        The starting value of the sequence.
    end : scalar
        The end value of the sequence.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.

    Notes
    -----
    Returns num evenly spaced samples, calculated over the interval
    [start, stop]. This is a pure python version of numpy's linspace to avoid
    a numpy depenency.
    '''
    if num == 1:
        return stop
    h = (stop - start) / float(num)
    values = [start + h * i for i in range(num+1)]
    return values


def smoothed(hist, x, window=None):
    '''Return mean probability over `window` centered at `x`.

    Parameters
    ----------
    x : scalar
        Location at which to estimate the (smoothed) PDF value.
    window : scalar
        Smoothing window, in minutes. If window is False, we
        return the 'unsmoothed' PDF value at x. If window is True or None,
        the standard deviation of durations is used as the window size.
    '''
    if window is False:
        return hist.pdf(x)
    elif window is None or window is True:
        window = sqrt(hist.var())  # Use standard deviation?
    W = linspace(x-window/2, x+window/2, int(window))
    return sum(hist.pdf(i) for i in W)/len(W)


class Transition:
    __slots__ = ['counter', 'holding']

    def __init__(self):
        self.counter = 0
        self.holding = StreamHist(20)

    def __str__(self):
        return 'counter={}, holding=\n{}'.format(self.counter, self.holding)

    def __iadd__(self, other):
        self.counter += other.counter
        self.holding += other.holding  # Merge histograms
        return self

    def update(self, duration, count):
        if isinstance(duration, StreamHist):
            assert count is None
            self.holding += duration
        else:
            assert count is not None
            self.counter += count
            self.holding.update(duration, count)

    def duration_density(self, x):
        return smoothed(self.holding, x)  # window? defaults to std dev


class SemiMarkov:
    __slots__ = ['transitions']

    def __init__(self, density=False):
        self.transitions = defaultdict(partial(defaultdict, Transition))

    def __add__(self, event):
        '''Add an event/path to the sketch.

        Parameters
        ----------
        event : tuple
            Should be of the form: (state_i, state_j, duration_at_i)

        Notes
        -----
        1. If memoizing return results, this would have to reset the caches.
        2. Currently, there is no limit on the number of states this model will
           track. This could be changed so that only the most recent k states
           are tracked, or we could use an approximate set of states via a
           count-min sketch. This would add additional uncertainty though.
        '''
        self.update(*event, count=1)
        return self

    def update(self, state1, state2, duration, count=1):
        self.transitions[state1][state2].update(duration, count)

    def __contains__(self, state):
        """Test if a given state is in the sketch."""
        return state in self.transitions

    def __iter__(self):
        return iter(self.transitions.items())

    @property
    def states(self):
        '''Current set of all possible states.

        Notes
        -----
        1. This property could be memoized to speed calculations.
        2. In python, this property should be computed via list comprehensions.
        '''
        ret = set()
        for s1 in self.transitions:
            ret.add(s1)
            for s2 in self.transitions[s1]:
                ret.add(s2)
        return {i for i in ret if i is not None}

    def clone(self):
        '''Return a deep copy of this model.'''
        return copy.deepcopy(self)

    def batch(self, paths):
        '''Update model with multiple paths at once.

        Parameters
        ----------
        paths: list or iterable of tuples
            List of tuples describing individual transition paths similar to
            the required parameters for the `update` method:
            (state_i, state_j, duration_at_i)
        '''
        # No checks, no nothing... just batch processing, pure and simple
        for path in paths:
            self += path
        return self

    def merge(self, other):
        '''Merge this model with other.

        Notes
        -----
        1. If memoizing return results, this would have to reset the caches.
        '''
        for s1 in other.transitions:
            for s2 in other.transitions[s1]:
                self.transitions[s1][s2] += other.transitions[s1][s2]
        return self

    def visitation_count(self, state):
        '''Visitation count from and to state.

        Notes
        -----
        1. This property could be memoized to speed calculations.
        '''
        return self.visitation_from(state) + self.visitation_to(state)

    def visitation_from(self, state):
        '''Visitation count from state.

        Notes
        -----
        1. This property could be memoized to speed calculations.
        '''
        return sum(self.transition_count(state, s) for s in self.states)

    def visitation_to(self, state):
        '''Visitation count to state.

        Notes
        -----
        1. This property could be memoized to speed calculations.
        '''
        return sum(self.transition_count(s, state) for s in self.states)

    def transition_count(self, i, j):
        '''Transition count from state1 to state2.'

        Notes
        -----
        1. This property could be memoized to speed calculations.
        '''
        if i in self.transitions and j in self.transitions[i]:
            return self.transitions[i][j].counter
        return 0

    def transition_probability(self, i, j):
        '''Transition probability from state1 to state2.

        Notes
        -----
        1. This property could be memoized to speed calculations.
        '''
        if i in self.transitions and j in self.transitions[i]:
            prob = self.transitions[i][j].counter/self.visitation_from(i)
        else:
            prob = 0.0
        return max(prob, EPS)

    def duration(self, i, j):
        '''Duration array from state1 to state2.

        Notes
        -----
        1. This property could be memoized to speed calculations.
        '''
        if i in self.transitions and j in self.transitions[i]:
            return self.transitions[i][j].holding
        return None

    def duration_density(self, i, j, t):
        '''Probability of waiting at state i for time t before transition to j.

        Notes
        -----
        1. This property could be memoized to speed calculations.
        2. This value could be smoothed to account for uncertainty in times.
        '''
        if i in self.transitions and j in self.transitions[i]:
            prob = self.transitions[i][j].duration_density(t)
        else:
            prob = 0.0
        return max(prob, EPS)

    def to_dict(self):
        res = defaultdict(partial(defaultdict, dict))
        for a in self.states:
            for b in self.states:
                d = self.transitions[a][b].holding.to_dict()
                c = self.transitions[a][b].counter
                res[a][b]['counter'] = c
                res[a][b]['holding'] = d
        return res

    def query(self, path, use_duration=True):
        '''Return query result of path.

        Parameters
        ----------
        path: tuple
            Path of state transitions (state_i, state_j, duration)
        use_duration: boolean
            Whether use duration or not

        Returns
        -------
        result: tuple
            Contains the join probability of all paths (multiple paths lead to
            multiplied probabilities) and each individual path component
            probability: (joint_probability, [(transition: probability), ...])
        '''
        probs = []
        for s1, s2, d in path:
            prob = self.transition_probability(s1, s2)
            if use_duration:
                prob *= self.duration_density(s1, s2, d)
            probs.append(prob)
        return reduce(lambda x, y: x * y, probs, 1.0), list(zip(path, probs))

    def plot(self):
        # We hide the plotting modules here so that we don't need them to
        # testing and play with models... only to plot
        import pygraphviz
        import tempfile
        from matplotlib.image import imread
        import matplotlib.pyplot as plt
        G = pygraphviz.AGraph(directed=True)
        for state in self.states:
            color = 'blue'
            G.add_node(state, color=color)

        for i, a in enumerate(self.states):
            for j, b in enumerate(self.states):
                p = round(self.transition_probability(a, b), 4)
                if p > 0:  # Only include 'valid' edges
                    G.add_edge(a, b, label=p)

        with tempfile.NamedTemporaryFile() as tf:
            G.draw(tf.name, format='png', prog='dot')
            img = imread(tf.name)
            plt.imshow(img)
            plt.axis('off')
        return plt.gcf()

    def _remove_state(self, toremove):
        '''Remove the given state from the model, ajusting counts accordingly.
        '''
        if toremove in self.states:
            trans_from = defaultdict(float)
            trans_to = defaultdict(float)

            for s in self.states:
                if s != toremove:
                    trans_from[s] = self.transition_count(s, toremove)
                    trans_to[s] = self.transition_count(toremove, s)

            total = sum(trans_to[s] for s in trans_to)

            if total > 0.0:
                for s1 in self.transitions:
                    for s2 in self.transitions[s1]:
                        if s1 != toremove and s2 != toremove:
                            count = trans_from[s1] * trans_to[s2] / total
                            # TODO: update durations/holding times?
                            self.transitions[s1][s2].counter += count

            self._discard_state(toremove)

    def _remove_all(self, toremove):
        '''Remove states from the resulting model.'''
        for s in toremove:
            self._remove_state(s)

    def _discard_transition(self, state1, state2):
        if state1 in self.transitions:
            if state2 in self.transitions[state1]:
                self.transitions[state1].pop(state2)
        assert state2 not in self.transitions[state1]

    def _discard_state(self, state_name):
        for s in self.states:
            self._discard_transition(state_name, s)
            self._discard_transition(s, state_name)
        if state_name in self.transitions:
            self.transitions.pop(state_name)
        assert state_name not in self.states

    def clean(self, topn=None, least=None):
        '''Clean model by removing unlikely transitions and infrequent states.

        Parameters
        ----------
        topn : int
            If topn is not None, then only keep the topn states in terms
            of total visitation count.
        least : int
            If least is not None, then only keep the states with a total
            visitation count above least.
        '''
        for s1 in self.transitions:
            for s2 in self.transitions[s1].keys():
                if self.transition_probability(s1, s2) <= EPS:
                    self._discard_transition(s1, s2)

        least = least if least else 0

        frequent = filter(
            lambda x: self.visitation_count(x) > least, self.states)
        if topn:
            frequent = sorted(frequent, key=lambda x: self.visitation_count(x),
                              reverse=True)[0:topn]
        toremove = filter(lambda x: x not in set(frequent), self.states)
        self._remove_all(toremove)
        # TODO: May be sufficient to simply 'discard' states instead of
        # 'safely' removing them? Discarding throws away data... but probably
        # not worth the effort of a) maintaining removal code, and b) overhead
        # of removing things nicely?


class MarkovSketch:
    def __init__(self):
        '''This is just a prototype sketch that uses the SemiMarkov model in
        the background, but takes on points in continuous time.
        '''
        self.model = SemiMarkov()
        self.npoints = 0
        self.current_state = None
        self.current_time = 0

    def __add__(self, event):
        '''Add an event to the sketch.

        Parameters
        ----------
        event : tuple
            Event must of the form: (state, time)
        '''
        state, time = event
        if state is not self.current_state:
            self.model.update(state, self.current_state, time-self.current_time)
            self.current_time = state
            self.current_time = time
        # Otherwise, no new state, so no event has taken place...
        self.npoints += 1  # Update count of points seen so far
        return self

    def __len__(self):
        '''Number of states in the underlying SemiMarkov model.'''
        return len(self.model.states)

    def __call__(self, path, use_duration):
        '''Query the underlying SemiMarkov model.'''
        return self.model.query(path, use_duration)

    def __contains__(self, state):
        '''Test if a given state is in the underlying SemiMarkov model.'''
        return state in self.model

    def __iter__(self):
        return iter(self.model)

    def batch(self, events):
        # No checks, no nothing... just batch processing, pure and simple
        for event in events:
            self += event
        return self

    def trim(self, freq=1):
        '''Return SemiMarkov model with state visitation counts < freq removed.
        '''
        result = self.model.clone().clean(least=freq)
        return result

    @property
    def states(self):
        return self.model.states
