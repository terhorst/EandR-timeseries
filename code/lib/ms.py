from __future__ import print_function

import numpy as np
from subprocess import Popen, PIPE
import os

MS_PATH = os.environ['MS_PATH']

class MsParser:
    def __init__(self, n, L, seeds, Ne=1e6, mu=1e-9, r=1e-8):
        # Default population genetic parameters correspond to Drosophila.
        self.r = r
        self.seeds = seeds
        cmd = [MS_PATH, str(n), "1", "-t", str(4 * Ne * L * mu), "-r", str(4 * Ne * (L - 1) * r), str(L), "-seeds"] + list(map(str, seeds))
        ms = Popen(cmd, stdout=PIPE)
        ms_out, _ = ms.communicate()
        lines = ms_out.split("\n")
        self.positions = []
        indices = []
        for i, pos in enumerate(lines[5].split()[1:]):
            pp = int(float(pos) * L)
            if pp not in self.positions:
                self.positions.append(pp)
                indices.append(i)
        assert len(set(self.positions)) == len(self.positions)
        haps = [[int(x) for x in h] for h in lines[6:] if h]
        self.haps = np.array(haps, dtype=bool)[:, indices]
        assert self.haps.shape[0] == n
        assert self.haps.shape[1] == len(self.positions)
