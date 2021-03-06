from collections import Counter
import os
import unittest

import numpy as np

from msprime_simulator import compress_to_genotype_matrix, get_incomplete_phasing_matrix
from phission import get_unmasked_even_indexes, phission_phase, get_mask
from utils import read_haplotype_matrix_from_vcf, switch_error, flip_columns, write_vcf_from_haplotype_matrix


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

        m_complete = phission_phase(m)

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

    def test_haplotype_matrix_from_vcf(self):
        """
        Test that reading from vcf you get back individual by SNP matrix
        """
        haplotypes = [[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 0]]
        haplotypes = np.array(haplotypes)

        filepath = 'fixtures/test.vcf'
        read_haplotypes = read_haplotype_matrix_from_vcf(filepath)

        self.assertEqual(tuple(read_haplotypes[0]), tuple(haplotypes[0]))
        self.assertEqual(tuple(read_haplotypes[1]), tuple(haplotypes[1]))
        self.assertEqual(tuple(read_haplotypes[2]), tuple(haplotypes[2]))
        self.assertEqual(tuple(read_haplotypes[3]), tuple(haplotypes[3]))

    def test_vcf_from_haplotype_matrix(self):
        haplotypes = [[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 0]]
        haplotypes = np.array(haplotypes)

        test_filepath = 'fixtures/test_vcf_from_haplotype_matrix.vcf'
        write_vcf_from_haplotype_matrix(test_filepath, haplotypes)
        read_haplotypes = read_haplotype_matrix_from_vcf(test_filepath)

        try:
            self.assertEqual(tuple(read_haplotypes[0]), tuple(haplotypes[0]))
            self.assertEqual(tuple(read_haplotypes[1]), tuple(haplotypes[1]))
            self.assertEqual(tuple(read_haplotypes[2]), tuple(haplotypes[2]))
            self.assertEqual(tuple(read_haplotypes[3]), tuple(haplotypes[3]))
        except Exception as e:
            raise e
        finally:
            os.remove(test_filepath)

    def test_flip_columns(self):
        """
        Test flipping the 0/1 convention
        """
        haplotypes = [[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 0]]
        haplotypes = np.array(haplotypes)

        expected_haplotypes = [[0, 1, 0, 1],
                               [0, 1, 0, 1],
                               [0, 1, 0, 1],
                               [0, 1, 1, 1],
                               [0, 0, 0, 1],
                               [0, 1, 0, 1],
                               [0, 0, 0, 1],
                               [0, 1, 0, 1]]
        expected_haplotypes = np.array(expected_haplotypes)

        flipped_haplotypes = flip_columns([1, 3], haplotypes)
        flipped_back = flip_columns([1, 3], flipped_haplotypes)
        for i in range(len(haplotypes)):
            self.assertEqual(tuple(flipped_haplotypes[i]), tuple(expected_haplotypes[i]))
            self.assertEqual(tuple(flipped_back[i]), tuple(haplotypes[i]))

