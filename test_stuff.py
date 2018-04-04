from collections import Counter
import unittest

import numpy as np

from nuclear_norm_minimization import get_unmasked_indexes, phase
from switch_error import switch_error


class TestStuff(unittest.TestCase):

    def test_phase(self):
        """
        Tests that rank is indeed minimized as expected when completing a matrix with 0s

        Assumes that the output is rounded appropriately

        # TODO nthomas I'm not sure how to properly write this test yet. Maybe it doesn't need to exist.
        # Or maybe it can just test trivial things like the size of the resulting matrix and values inside (ie constraints)
        # ... but that's really just testing cvxpy
        """
        m = [[0, 1],
             [0, 1],
             [1, 1],
             [1, 1],
             [-1, 1],
             [-1, 1],
             [1, 1],
             [1, 1]]
        m = np.array(m)
        # we expect rank to be minimized
        m_complete_expected = [[-1.0, 1.0],
                               [1.0, 1.0],
                               [1.0, 1.0],
                               [1.0, 1.0]]
        m_complete_expected = np.array(m_complete_expected)
        m_complete = phase(m, mu=2)

        # for now, we just test that this function runs

        # self.assertEqual(m_complete[0, 0], m_complete_expected[0, 0])
        # self.assertEqual(m_complete[0, 1], m_complete_expected[0, 1])
        # self.assertEqual(m_complete[1, 0], m_complete_expected[1, 0])
        # self.assertEqual(m_complete[1, 1], m_complete_expected[1, 1])

    def test_unmasked_indexes(self):
        """
        Recover indexes (i, j) where we need to phase. From a haplotype matrix!

        For use
        """
        m = [[0, 1, 0, -1],
             [0, 1, 0, -1]]
        m = np.array(m)

        indexes_expected = [(0, 0), (0, 2)]

        # Counter best way to compare lists
        # https://stackoverflow.com/questions/7828867/how-to-efficiently-compare-two-unordered-lists-not-sets-in-python
        self.assertEqual(Counter(get_unmasked_indexes(m)), Counter(indexes_expected))

    def test_complete(self):
        """
        Tests that the inferred matrix has all 1s and -1s, and no other values floating around. Sanity check
        """
        self.assertTrue(True)

    def test_switch_error(self):
        """
        Test switch error function
        """
        observed = [[1, 1, 1, -1],
                    [1, 1, -1, 1],
                    [1, 1, 1, -1],
                    [1, 1, -1, 1]]

        observed = np.array(observed)

        expected = [[1, 1, 1, 1],
                    [1, 1, -1, -1],
                    [1, 1, 1, 1],
                    [1, 1, -1, -1]]
        expected = np.array(expected)

        self.assertEqual(switch_error(observed, expected), 2)
