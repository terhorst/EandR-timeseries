#!/usr/bin/python2.7 -O
from __future__ import division
import sys
sys.path.append("lib/")

import random
import tempfile
import math
import os
import numpy as np
import random
import shutil

from simupopulator import SimuPopulator as Slimulator
from ms import MsParser

SLIM_PATH = os.environ['SLIM_PATH']
SLIM_TMP_PATH = os.environ['SLIM_TMP_PATH']

class ERSimulation:
    'Simulate an evolve-and-resequence experiment.'
    def __init__(self, N, r, h, s, L, replicates, F, sampling, times):
        '''Initialize with census population size N, recombination fraction r,
        dominance parameter h, selection coefficient s, segment length L,
        number of replicates R, number of founder haplotypes F, sampling depth
        sampling.
        '''
        self.N = N
        self.r = r
        self.h = h
        self.s = s
        self.L = L
        self.replicates = replicates
        self.F = F
        self.sampling = sampling
        self.times = times
        self.selected_position = None
        # For now, # founders must evenly divide population
        assert N % F == 0
        self.msp = MsParser(F, L, np.random.randint(1000000, size=3))
        Lh = self.msp.haps.shape[1]
        msh = self.msp.haps.sum(axis=0) / F
        # Set some other variables for future use
        self.TD = tempfile.mkdtemp(dir=SLIM_TMP_PATH)
        self.selected_site = None
        self.recomb_map = None

    @property
    def positions(self):
        return self.msp.positions

    def __del__(self):
        # Clean up generated files afterwards
        shutil.rmtree(self.TD)
        # print(self.TD)

    def simulate(self):
        # Run slim based on ms output
        self.slim = Slimulator(self.TD, self.msp.positions, self.times,
                self.msp.haps, self.selected_position, self.replicates, self.L,
                self.N, self.r, self.h, self.s, self.recomb_map)
        self.slim.run()

    def normld(self, i, j):
        # Calculate LD between sites i and j
        haps = self.msp.haps
        pXY = (haps[:, i] * haps[:, j]).sum() / self.F
        pX = haps[:, i].sum() / self.F
        pY = haps[:, j].sum() / self.F
        return (pXY - pX * pY) / math.sqrt(pX * pY * (1. - pX) * (1. - pY))


if __name__=="__main__":
    # Perform a test simulation
    sim = ERSimulation(500, 1e-8, 0.5, 0.01, 10000, 3, 10, None, (10, 20, 30, 40, 50))
    sim.simulate(random.sample(sim.msp.positions, 1)[0])
    subs = sim.slim.subset((10, 20, 30, 40, 50), sim.msp.positions[10:40], sim.msp.positions[15])
    print(subs.data.shape)
