#!/usr/bin/python2.7 -O
from __future__ import division
import sys
sys.path.append("lib/")

import math
import argparse
import psycopg2
import os
import numpy as np
import random
np.set_printoptions(linewidth=140, precision=5, suppress=True)

from er_simulation import ERSimulation
from likelihood import Likelihood
from db import *

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Test the model.')
    parser.add_argument('--L', type=int, required=True)
    parser.add_argument('--times', type=int, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument('--nsites', type=int, nargs="+", default=[0])
    parser.add_argument('--s', type=float, required=True)
    parser.add_argument('--r', type=float, required=True)
    parser.add_argument('--h', type=float, required=True)
    parser.add_argument('--N', type=int, required=True)
    parser.add_argument('--sampling', type=int, default=None)
    parser.add_argument('--F', type=int, help="Number of founder haplotypes", required=True)
    parser.add_argument('--replicates', default=3, type=int)
    parser.add_argument('--comment')

    args = parser.parse_args()
    L = args.L
    s = args.s
    r = args.r
    h = args.h
    N = args.N
    F = args.F
    replicates = args.replicates
    times = args.times
    sampling = args.sampling

    sim = ERSimulation(N, r, h, s, L, replicates, F, sampling, times)
    sim.selected_position = random.sample(sim.positions, 1)[0]
    ssi = sim.positions.index(sim.selected_position)
    sim.simulate()

    liks = []
    # Consider nearby (strongly) linked sites which are bounded 
    # in frequency away from the edges of the simplex
    m = np.min(np.min(sim.slim.data, axis=0), axis=1)
    M = np.max(np.max(sim.slim.data, axis=0), axis=1)
    LDs = list(reversed(sorted([(abs(sim.normld(ssi, j)), sim.positions[j]) 
                                for j in range(len(sim.positions)) 
                                if j != ssi
                                and 0.2 <= m[j] <= M[j] <= 0.8])))
    for ns in args.nsites:
        print(LDs[:ns])
        positions = sorted(set([sim.selected_position] + [b for a, b in LDs[:ns]]))
        subsets = sim.slim.subset(times, positions, sim.selected_position, args.sampling)
        lik = Likelihood(subsets, times, positions, sim.selected_position, args.sampling, verbose=True)
        fit = lik.maxlik(N=N, log10r=math.log10(r), h=h, bounds=(-.15, .15), tol=.005)
        s_hat = fit.x
        ml = -fit.fun
        ml0 = lik.likelihood(N, r, h, s)
        print("mlhat: %g\tml0: %g" %(ml, ml0))
        x0 = sim.slim.data[0, tuple([sim.positions.index(pos) for pos in positions]), 0].tolist()
        tup = MaxLik(positions=positions, selected_position=sim.selected_position, x0=x0, mle=s_hat, loglik=ml, lr=-2 * (ml0 - ml))
        liks.append(tup)
        print(tup)

    print(liks)
    res = Result(N=N, positions=sim.positions,
            selected_position=sim.selected_position, s=s, r=r, h=h,
            replicates=replicates, F=F, L=L, sampling=args.sampling,
            times=times, comment=args.comment, command_line=" ".join(sys.argv),
            ms_seeds=sim.msp.seeds.tolist(), slim_seed=sim.slim.seed)
