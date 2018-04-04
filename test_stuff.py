from collections import Counter
import unittest

import numpy as np

from nuclear_norm_minimization import get_unmasked_indexes, phase


class TestNucNormMin(unittest.TestCase):

    def test_phase(self):
        """
        Tests that rank is indeed minimized as expected when completing a matrix with 0s

        Assumes that the output is rounded appropriately
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

        self.assertEqual(m_complete[0, 0], m_complete_expected[0, 0])
        self.assertEqual(m_complete[0, 1], m_complete_expected[0, 1])
        self.assertEqual(m_complete[1, 0], m_complete_expected[1, 0])
        self.assertEqual(m_complete[1, 1], m_complete_expected[1, 1])

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
