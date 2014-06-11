#!/usr/bin/env python3

import unittest
import pyexp
import numpy as np
from itertools import combinations_with_replacement as cwr
from pprint import pprint
import math


def mkexp(paramdict):
    args = [paramdict[x] for x in 
            "n h s zL zS zR zLS zLR zRS zLRS rLR rLS rRS rLS_R rRS_L rLR_S".split()]
    return pyexp.Expectation(*args)


class TestPyExp(unittest.TestCase):
    '''A smattering of test cases that this model ought to pass.'''
    n = np.random.randint(200, 5000)
    A = np.array(np.random.dirichlet((1,)*8), dtype=np.double)
    zL, zS, zR, zLS, zLR, zRS, zLRS, _ = A
    h = np.random.uniform()
    s = math.copysign(np.random.exponential(.05), np.random.standard_normal())
    d = dict(
            n=n, 
            h=h,
            s=s,
            zL=zL, # zL
            zS=zS, # zS
            zR=zR, # zR
            zLS=zLS, # zLS
            zLR=zLR, # zLR
            zRS=zRS, # zRS
            zLRS=zLRS, # zLRS
            # Configuration: LSR
            rLR=2e-6, # rLR
            rLS=1e-6, # rLS
            rRS=1e-6, # rRS
            rLS_R=1e-6, # rLS_R
            rRS_L=1e-6, # rRS_L,
            rLR_S=0.0 # rLR_S
            )
    pprint(d)


    def assertAlmostEqual(self, a, b, **kwargs):
        kwargs.update({'places': 4})
        return super().assertAlmostEqual(a, b, **kwargs)


    def testNoSelection(self):
        d = self.d.copy()
        d['s'] = 0.0
        E = mkexp(d)
        # No selection, constant expectation
        for t in [1, 5, 10, 50]:
            self.assertAlmostEqual(E.EL(t), d['zL'] + d['zLS'] + d['zLR'] + d['zLRS'], msg=t)
            self.assertAlmostEqual(E.ES(t), d['zS'] + d['zLS'] + d['zRS'] + d['zLRS'])
            self.assertAlmostEqual(E.ER(t), d['zR'] + d['zLR'] + d['zRS'] + d['zLRS'])


    def testSMarginal(self):
        d2 = self.d.copy()
        E = mkexp(d2)
        for t in [1, 5, 10, 50]:
            e1 = mkexp(d2).ES(t)
            for zL in [0, 100, 1000]:
                d2['zL'] = zL
                self.assertAlmostEqual(mkexp(d2).ES(t), e1)


    def testCovariance(self):
        E = mkexp(self.d)
        for t1 in [1, 5, 10, 20, 30, 40, 50]:
            for t2 in [1, 5, 10, 20, 30, 40, 50]:
                for fn in 'LL RR SS LR RS LS'.split():
                    f = getattr(E, 'cov' + fn)
                    if fn[0] == fn[1]:
                        lb = 0.0
                    else:
                        lb = -1.0
                    self.assertLessEqual(lb, f(t1, t2), msg="cov%s(%i, %i)" % (fn, t1, t2))
                    self.assertLessEqual(f(t1, t2), 1.0, msg="cov%s(%i, %i)" % (fn, t1, t2))


    def testVarianceMarginal(self):
        # Test variance when left and right sites are the same
        d2 = self.d.copy()
        d2['zL'] = 0
        d2['zR'] = 0
        E = mkexp(d2)
        for t in [1, 5, 10, 50]:
            for zLR in [.2, .3]:
                d2['zLR'] = zLR 
                d2['zLS'] = d2['zRS'] = zLR
                E = mkexp(d2)
                self.assertAlmostEqual(E.covLL(t, t + 10), E.covLL(t + 10, t))
                self.assertAlmostEqual(E.covLS(5, t), E.covRS(5, t))
                self.assertAlmostEqual(E.covLS(t, 10), E.covRS(t, 10))



    def testVarianceS(self):
        # Selected variance does not care about what's going on at other sites
        d2 = self.d.copy()
        d2['zL'] = 0
        d2['zS'] = 300
        d2['zLS'] = 200
        E = mkexp(d2)
        e1 = E.ES(10)
        c1 = E.covSS(10, 10)

        d2['zL'] = 100
        d2['zS'] = 300
        d2['zLS'] = 200
        E = mkexp(d2)
        e2 = E.ES(10)
        c2 = E.covSS(10, 10)

        d2['zL'] = 300
        d2['zS'] = 400
        d2['zLS'] = 100
        E = mkexp(d2)
        e3 = E.ES(10)
        c3 = E.covSS(10, 10)
        
        self.assertAlmostEqual(e1, e2)
        self.assertAlmostEqual(e2, e3)
        self.assertAlmostEqual(c1, c2)
        self.assertAlmostEqual(c2, c3)


    def testTimeSymmetry(self):
        E = mkexp(self.d)
        for t1, t2 in cwr([1, 5, 10, 20, 30, 40, 50], 2):
            self.assertAlmostEqual(E.covLR(t1, t2), E.covLR(t2, t1))
            self.assertAlmostEqual(E.covLS(t1, t2), E.covLS(t2, t1))
            self.assertAlmostEqual(E.covRS(t1, t2), E.covRS(t2, t1))


    def testCauchySchwarz(self):
        E = mkexp(self.d)
        for t1, t2 in ((1, 1), (1, 2)): # , (1, 5), (5, 10), (10, 20)):
            for a, b in ['LS', 'LR', 'RS', 'LL', 'SS', 'RR']:
                f, g, h = [getattr(E, 'cov' + x) for x in (a+b, a+a, b+b)]
                for s1, s2 in ((t1, t2), (t2, t1)):
                    r = (f(s1, s2), g(s1, s1), h(s2, s2))
                    m = " ".join(["%s%s(%i,%i) = %g"]*3) % (a, b, s1, s2, r[0],
                            a, a, s1, s1, r[1], b, b, s2, s2, r[2])
                    self.assertLessEqual(r[0]**2, r[1]*r[2], msg=m)


    def testDeterministic(self):
        E = mkexp(self.d)
        for t in [1, 5, 10, 20, 30, 40, 50]:
            f = E.deterministic(t)
            vals = f.values()
            self.assertTrue(all(v >= 0 for v in vals), msg=self.d)
            self.assertLessEqual(sum(vals), 1.0)


    def testNoRecombination(self):
        d2 = self.d.copy()
        for r in 'rLR rLS rRS rLR_S rRS_L rLS_R'.split():
            d2[r] = 0.0
        d2['zL'] = d2['zLS'] = d2['zLR'] = d2['zLRS'] = 0.0
        E = mkexp(d2)
        for t in [1, 5, 10, 20, 30, 40, 50]:
            self.assertEqual(E.EL(t), 0.0)
            self.assertEqual(E.varL(t), 0.0)


class TestCase1(unittest.TestCase):
    def test(self):
        d = {'h': 0.026663260333421146,
                'n': 1929,
                'rLR': 2e-06,
                'rLR_S': 0.0,
                'rLS': 1e-06,
                'rLS_R': 1e-06,
                'rRS': 1e-06,
                'rRS_L': 1e-06,
                's': -0.1373728926757938,
                'zL': 0.082085069073362965,
                'zLR': 0.036333461359365167,
                'zLRS': 0.026360386934593669,
                'zLS': 0.38126967630443831,
                'zR': 0.016262477700528705,
                'zRS': 0.22412427563046333,
                'zS': 0.091550978595241111}
        E = mkexp(d)
        for t in [1, 5, 10, 20, 30, 40, 50]:
            self.assertLessEqual(E.covLL(t, t + 10), 1.0)
            self.assertLessEqual(-1.0, E.covLL(t, t + 10))


if __name__ == '__main__':
    unittest.main()
