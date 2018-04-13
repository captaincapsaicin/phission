import sys

import numpy as np
from tabulate import tabulate

from msprime_simulator import simulate_haplotype_matrix, compress_to_genotype_matrix, get_incomplete_phasing_matrix
from nuclear_norm_minimization import get_mask, nuclear_norm_solve
from switch_error import switch_error


def main(num_snps):
    # simulate with msprime
    all_haplotypes = simulate_haplotype_matrix(100)
    while all_haplotypes.shape[1] < num_snps:
        all_haplotypes = simulate_haplotype_matrix(100)
    true_haplotypes = all_haplotypes[:, 0:num_snps]

    print('haplotype dimension')
    print(true_haplotypes.shape)

    genotypes = compress_to_genotype_matrix(true_haplotypes)
    unphased_haplotypes = get_incomplete_phasing_matrix(genotypes)

    # this is what the ".phase" method does
    mask = get_mask(unphased_haplotypes)
    phased_haplotypes = nuclear_norm_solve(unphased_haplotypes, mask, mu=1)
    rounded = np.matrix.round(phased_haplotypes).astype(int)

    headers = ['nuclear norm', 'rank', 'normalized frob distance']
    data = [['true', np.linalg.norm(true_haplotypes, 'nuc'), np.linalg.matrix_rank(true_haplotypes), np.linalg.norm(true_haplotypes - true_haplotypes)/np.linalg.norm(true_haplotypes)],
            ['unphased', np.linalg.norm(unphased_haplotypes, 'nuc'), np.linalg.matrix_rank(unphased_haplotypes), np.linalg.norm(unphased_haplotypes - true_haplotypes)/np.linalg.norm(true_haplotypes)],
            ['phased', np.linalg.norm(phased_haplotypes, 'nuc'), np.linalg.matrix_rank(phased_haplotypes), np.linalg.norm(phased_haplotypes - true_haplotypes)/np.linalg.norm(true_haplotypes)],
            ['rounded', np.linalg.norm(rounded, 'nuc'), np.linalg.matrix_rank(rounded), np.linalg.norm(rounded - true_haplotypes)/np.linalg.norm(true_haplotypes)]]
    print(tabulate(data, headers=headers))

    headers = ['-1s phased', 'Nonzero set to -1s', '-1s remaining']
    zeros = [np.sum(np.logical_and(unphased_haplotypes == -1, rounded != -1)),
             np.sum(np.logical_and(unphased_haplotypes != -1, rounded == -1)),
             np.sum(rounded == -1)]

    row_format = '{:>20}' * (len(headers))
    print('\n')
    print(row_format.format(*headers))
    print(row_format.format(*zeros))

    print('switch error')
    print(switch_error(phased_haplotypes, true_haplotypes))


if __name__ == '__main__':
    main(int(sys.argv[1]))
