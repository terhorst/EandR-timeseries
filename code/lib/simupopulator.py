from __future__ import division, print_function
import itertools as it
import numpy as np
from collections import namedtuple, Counter
import os
import random

from simuOpt import setOptions
setOptions(optimized=True, alleleType='binary', quiet=False)
import simuPOP as sim

SimulationResult = namedtuple("SimulationResult", "positions times dists data init_haps init_haps2")


class SimuPopulator:
    Z = [ 
            (False, True, True), # zL
            (True, False, True), # zS
            (True, True, False), # zR
            (False, False, True), # zLS
            (False, True, False), # zLR
            (True, False, False), # zRS
            (False, False, False) # zLRS
        ]
    Z = [tuple(not x for x in y) for y in Z]

    Z2 = [
            (False, False), # AB
            (False, True), # Ab
            (True, False), # aB
            (True, True) # ab
        ]
    Z2 = [tuple(not x for x in y) for y in Z2]

    def __init__(self, base_dir, positions, times, initial_haplotypes,
            selected_position, replicates, L, N, r, h, s,
            recomb_map=None):
        self.base_dir = base_dir
        self.positions = positions
        self.times = times
        self.init_haps = initial_haplotypes
        self.selected_position = selected_position
        self.replicates = replicates
        self.N = N
        self.r = r
        self.s = s
        self.h = h
        self.L = L
        print(N, r, s, h, L)
        self.seed = random.randint(0, 1000000000)

        if recomb_map is None:
            self.recomb_map = [self.r * (self.positions[i] - self.positions[i - 1]) 
                               for i in range(1, len(self.positions))]
        else:
            self.recomb_map = recomb_map

        self.simu = sim.Simulator(
                sim.Population(size=N, infoFields='fitness', loci=len(self.positions)),
                rep=replicates)

    def run(self):
        # Run the simulator
        ftns = {(0, 0): 1.0, (0, 1): 1.0 + self.h * self.s, (1, 1): 1.0 + self.s}
        if self.selected_position is not None:
            si = self.positions.index(self.selected_position)
            po = [sim.MapSelector(loci=si, fitness=ftns)]
        else:
            po = []
        self.simu.evolve(
            initOps=[
                sim.InitSex(),
                sim.InitGenotype(
                    haplotypes=self.init_haps.astype(int).tolist(),
                    prop=[1. / self.init_haps.shape[0]] * self.init_haps.shape[0]
                    ),
                sim.Stat(alleleFreq=sim.ALL_AVAIL),
                sim.PyExec('traj = [[alleleFreq[k][1] for k in sorted(alleleFreq)]]'),
                ],
            preOps=po,
            matingScheme=sim.RandomMating(
                ops=sim.Recombinator(rates=self.recomb_map + [0], loci=range(len(self.positions)))
                ),
            postOps=[sim.Stat(alleleFreq=sim.ALL_AVAIL),
                     sim.PyExec('traj.append([alleleFreq[k][1] for k in sorted(alleleFreq)])'),
                     #traj.append([af[1] for af in alleleFreq])')
            ],
            gen=max(self.times)
        )
        # data: times x sites x replicates
        # this: reps x times x sites
        self.data = np.transpose([p.dvars().traj for p in self.simu.populations()], (1, 2, 0))
        self.data = self.data[np.ix_([0] + self.times)]
        # for i in range(self.replicates):
            # print(self.data[..., (si - 2):(si + 3), i])


    def subset(self, times, positions, selected_position, sampling):
        dta = self.data.copy()
        if sampling:
            # Replace each entry of data with a binomial sample:
            def bi(x):
                nr = sampling
                return 1.0 * np.random.binomial(nr, x) / nr
            bia = np.frompyfunc(bi, 1, 1)
            # Assume initial haplotypes are still known; only sample for gens t>0
            dta[1:] = bi(self.data[1:]).astype(float)
        indices = [self.positions.index(p) for p in positions]
        ih = np.zeros([len(indices), len(indices), 7], dtype=np.double)
        ih2 = np.zeros([len(indices), len(indices), 4], dtype=np.double)
        for (i1, sl1), (i2, sl2) in it.product(enumerate(indices), repeat=2):
            if selected_position:
                sp = self.positions.index(selected_position)
            else:
                sp = sl1
            cc = Counter(map(tuple, self.init_haps[:, (sl1, sp, sl2)]))
            F = self.init_haps.shape[0]
            ih[i1, i2] = [cc[k] / F for k in self.Z]
            cc2 = Counter(map(tuple, self.init_haps[:, (sl1, sl2)]))
            ih2[i1, i2] = [cc2[k] / F for k in self.Z2]
        tinds = [1 + self.times.index(t) for t in times]
        return SimulationResult(
                data=dta[np.ix_(tinds, indices)],
                positions=positions,
                dists=np.abs(np.subtract.outer(positions, positions)),
                init_haps=ih,
                init_haps2=ih2,
                times=times)
