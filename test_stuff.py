from collections import Counter
import unittest

import numpy as np

from msprime_simulator import compress_to_genotype_matrix, get_incomplete_phasing_matrix
from nuclear_norm_minimization import get_unmasked_even_indexes, phase, get_mask
from switch_error import switch_error


class TestStuff(unittest.TestCase):

    def test_phase(self):
        """
        Tests that constraints are satisfied, and that no positions are unphased
        """
        m = [[-1, 1, 1, 1, -1, 1],
             [-1, 1, 1, 1, -1, 1],
             [0, 1, 1, 1, 1, 1],
             [0, 1, 1, 1, 1, 1],
             [1, -1, 0, 1, 1, 1],
             [1, -1, 1, 1, 1, 1],
             [1, 1, 1, 1, 0, 1],
             [1, 1, 1, 1, 0, 1]]
        m = np.array(m)

        m_complete = phase(m, mu=2)

        # nothing should be -1 (i.e. unphased)
        self.assertTrue((1 - (m_complete == -1)).all())
        # symmetry-broken positions should sum to 1
        self.assertEqual(m_complete[0, 0] + m_complete[1, 0], 1)
        self.assertEqual(m_complete[4, 1] + m_complete[5, 1], 1)

    def test_unmasked_even_indexes(self):
        """
        Recover indexes (i, j) where we need to phase. From a haplotype matrix!

        For use in constructing sum to 1 constraints
        """
        m = [[-1, 1, -1, 0],
             [-1, 1, -1, 0]]
        m = np.array(m)

        mask = get_mask(m)
        indexes_expected = [(0, 0), (0, 2)]

        # Counter best way to compare lists
        # https://stackoverflow.com/questions/7828867/how-to-efficiently-compare-two-unordered-lists-not-sets-in-python
        self.assertEqual(Counter(get_unmasked_even_indexes(mask)), Counter(indexes_expected))

    # def test_complete(self):
    #     """
    #     Tests that the inferred matrix has all 1s and -1s, and no other values floating around. Sanity check
    #     """
    #     self.assertTrue(True)

    def test_switch_error(self):
        """
        Test switch error function
        """
        observed = [[1, 1, 1, 0],
                    [1, 1, 0, 1],
                    [1, 1, 1, 0],
                    [1, 1, 0, 1]]
        observed = np.array(observed)

        expected = [[1, 1, 1, 1],
                    [1, 1, 0, 0],
                    [1, 1, 1, 1],
                    [1, 1, 0, 0]]
        expected = np.array(expected)

        self.assertEqual(switch_error(observed, expected), 2)

    def test_switch_error_2(self):
        """
        Test switch error function
        """
        observed = [[1, 1, 1, 0],
                    [1, 1, 1, 1],
                    [1, 1, 1, 0],
                    [1, 1, 1, 1]]
        observed = np.array(observed)

        expected = [[1, 1, 1, 1],
                    [1, 1, 1, 0],
                    [1, 1, 1, 1],
                    [1, 1, 1, 0]]
        expected = np.array(expected)

        self.assertEqual(switch_error(observed, expected), 0)

    def test_switch_error_3(self):
        """
        Test switch error function
        """
        observed = [[1, 1, 0, 0],
                    [1, 0, 1, 1],
                    [1, 1, 1, 1],
                    [1, 0, 0, 0]]
        observed = np.array(observed)

        expected = [[1, 0, 1, 1],
                    [1, 1, 0, 0],
                    [1, 0, 0, 1],
                    [1, 1, 1, 0]]
        expected = np.array(expected)

        self.assertEqual(switch_error(observed, expected), 1)

    def test_switch_error_4(self):
        """
        Test switch error function
        """
        observed = [[1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1]]
        observed = np.array(observed)

        expected = [[1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1]]
        expected = np.array(expected)

        self.assertEqual(switch_error(observed, expected), 0)

    def test_compress_to_genotypes(self):
        haplotypes = [[1, 0, 1, 0],
                      [1, 0, 0, 0],
                      [1, 1, 1, 0],
                      [1, 1, 0, 1]]
        haplotypes = np.array(haplotypes)

        genotypes = [[2, 0, 1, 0],
                     [2, 2, 1, 1]]
        genotypes = np.array(genotypes)

        compressed = compress_to_genotype_matrix(haplotypes)
        self.assertEqual(tuple(compressed[0]), tuple(genotypes[0]))
        self.assertEqual(tuple(compressed[1]), tuple(genotypes[1]))

    def test_get_incomplete_phasing_matrix(self):
        genotypes = [[2, 0, 1, 0],
                     [2, 2, 1, 1]]
        genotypes = np.array(genotypes)

        haplotypes = [[1, 0, -1, 0],
                      [1, 0, -1, 0],
                      [1, 1, -1, -1],
                      [1, 1, -1, -1]]
        haplotypes = np.array(haplotypes)

        incomplete_haplotypes = get_incomplete_phasing_matrix(genotypes)

        self.assertEqual(tuple(incomplete_haplotypes[0]), tuple(haplotypes[0]))
        self.assertEqual(tuple(incomplete_haplotypes[1]), tuple(haplotypes[1]))
        self.assertEqual(tuple(incomplete_haplotypes[2]), tuple(haplotypes[2]))
        self.assertEqual(tuple(incomplete_haplotypes[3]), tuple(haplotypes[3]))
