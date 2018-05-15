import msprime
import numpy as np


def simulate_haplotype_matrix(num_haps,
                              num_snps,
                              Ne=1e4,
                              length=5e3,
                              recombination_rate=0,
                              mutation_rate=2e-8,
                              random_seed=None):
    """
    Returns a num_haps x num_snps haplotype matrix of 0s and 1s.
    """
    cur_num_haps = num_haps
    while expected_num_snps(cur_num_haps, Ne, length, mutation_rate) < 2 * num_snps:
        cur_num_haps = 2*cur_num_haps  # just double it until the expectation is pretty high

    tree_sequence = msprime.simulate(
         sample_size=cur_num_haps,
         Ne=Ne,
         length=length,
         recombination_rate=recombination_rate,
         mutation_rate=mutation_rate,
         random_seed=random_seed)

    haplotype_matrix = tree_sequence.genotype_matrix().T
    print(haplotype_matrix.shape) # TODO REMOVE DEBUGGING NTHOMAS
    # we take the transpose to get individuals in rows, SNPs in columns
    return haplotype_matrix[:num_haps, :num_snps]


def expected_num_snps(num_haps, Ne, length, mutation_rate):
    return 4 * mutation_rate * length * Ne * np.sum([1/i for i in range(1, num_haps)])


def compress_to_genotype_matrix(haplotypes):
    """
    Assumes 0s and 1s in haplotype matrix (output of msprime)

    e.g.
    array([[1, 1, 1, ..., 1, 1, 0],
           [1, 1, 0, ..., 1, 1, 0],
           [1, 0, 1, ..., 0, 0, 0],
           ...,
           [1, 1, 1, ..., 1, 1, 0],
           [1, 0, 1, ..., 0, 0, 0],
           [1, 1, 1, ..., 1, 1, 0]])
    """
    return haplotypes[::2] + haplotypes[1::2]


def get_incomplete_phasing_matrix(genotypes):
    """
    Assumes 0s, 1s, and 2s in genotype matrix

    array([[0, 0, 0, ..., 0, 0, 2],
           [0, 0, 1, ..., 0, 0, 2],
           [0, 1, 0, ..., 2, 2, 2],
           ...,
           [0, 0, 0, ..., 0, 0, 2],
           [0, 1, 0, ..., 1, 1, 2],
           [0, 0, 0, ..., 0, 0, 2]], dtype=uint8)

    Returns:
        a matrix with 0s and 1s in homozygous positions. -1s in unphased, heterozygous positions
    """
    to_duplicate = 0*(genotypes == 0).astype(int) + -1*(genotypes == 1).astype(int) + 1*(genotypes == 2).astype(int)
    n, m = to_duplicate.shape

    incomplete_haplotypes = np.zeros((2*n, m))
    incomplete_haplotypes[::2] = to_duplicate
    incomplete_haplotypes[1::2] = to_duplicate
    return incomplete_haplotypes
