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
    haplotypes = all_haplotypes[:, 0:num_snps]

    print('haplotype dimension')
    print(haplotypes.shape)

    genotypes = compress_to_genotype_matrix(haplotypes)
    unphased_haplotypes = get_incomplete_phasing_matrix(genotypes)

    # this is what the ".phase" method does
    mask = get_mask(unphased_haplotypes)
    phased_haplotypes = nuclear_norm_solve(unphased_haplotypes, mask, mu=1)
    rounded = np.matrix.round(phased_haplotypes)

    # converting 0/1 representation to -1/1
    true_haplotypes = -1*(haplotypes == 0).astype(int) + 1*(haplotypes == 1).astype(int)
    # maybe we'll use something like thresholding instead of rounding
    thresh = -1*(phased_haplotypes < 0) + 1*(phased_haplotypes > 0)

    headers = ['nuclear norm', 'rank']
    data = [['true', np.linalg.norm(true_haplotypes, 'nuc'), np.linalg.matrix_rank(true_haplotypes)],
            ['unphased', np.linalg.norm(unphased_haplotypes, 'nuc'), np.linalg.matrix_rank(unphased_haplotypes)],
            ['phased', np.linalg.norm(phased_haplotypes, 'nuc'), np.linalg.matrix_rank(phased_haplotypes)],
            ['rounded', np.linalg.norm(rounded, 'nuc'), np.linalg.matrix_rank(rounded)],
            ['thresholded', np.linalg.norm(thresh, 'nuc'), np.linalg.matrix_rank(thresh)]]
    print(tabulate(data, headers=headers))

    headers = ['0s phased', 'Nonzero set to 0s', '0s remaining']
    zeros = [np.sum(np.logical_and(unphased_haplotypes == 0, rounded != 0)),
             np.sum(np.logical_and(unphased_haplotypes != 0, rounded == 0)),
             np.sum(rounded == 0)]

    row_format = '{:>20}' * (len(headers))
    print('\n')
    print(row_format.format(*headers))
    print(row_format.format(*zeros))

if __name__ == '__main__':
    main(int(sys.argv[1]))

# TODO nthomas: add switch error
# we can't really mess with switch error until we've set all the 0s to nonzero values
# print(switch_error(phased_haplotypes, true_haplotypes))
# print('switch error')
