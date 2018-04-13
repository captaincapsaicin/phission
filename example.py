import sys

import numpy as np
from tabulate import tabulate

from msprime_simulator import simulate_haplotype_matrix, compress_to_genotype_matrix, get_incomplete_phasing_matrix
from nuclear_norm_minimization import get_mask, nuclear_norm_solve
from switch_error import switch_error


def main(num_snps, mu=2, num_ref=0):
    # simulate with msprime
    all_haplotypes = simulate_haplotype_matrix(200)
    while all_haplotypes.shape[1] < num_snps:
        all_haplotypes = simulate_haplotype_matrix(200)
    haplotypes = all_haplotypes[:, 0:num_snps]

    # converting 0/1 representation to -1/1
    true_haplotypes = -1*(haplotypes == 0).astype(int) + 1*(haplotypes == 1).astype(int)
    # just using half
    # true_haplotypes = true_haplotypes[0::2]

    print('haplotype dimension')
    print(haplotypes.shape)

    genotypes = compress_to_genotype_matrix(haplotypes)
    unphased_haplotypes = get_incomplete_phasing_matrix(genotypes)

    # what if we just used half of them
    unphased_haplotypes = unphased_haplotypes[0::2]

    # supplement with reference (true) haplotypes
    print('{} reference haplotypes added'.format(num_ref))
    unphased_haplotypes = np.vstack([unphased_haplotypes, true_haplotypes[0:num_ref]])
    true_with_ref = np.vstack([true_haplotypes[0::2], true_haplotypes[0:num_ref]])
    print('unphased dimension {}'.format(unphased_haplotypes.shape))
    # unphased_haplotypes[0:num_ref] = true_haplotypes[0:num_ref]

    # this is what the ".phase" method does
    mask = get_mask(unphased_haplotypes)
    phased_haplotypes = nuclear_norm_solve(unphased_haplotypes, mask, mu=mu)
    rounded = np.matrix.round(phased_haplotypes)

    # maybe we'll use something like thresholding instead of rounding
    # thresh = -1*(phased_haplotypes < 0) + 1*(phased_haplotypes > 0)

    headers = ['nuclear norm', 'rank']
    data = [['true', np.linalg.norm(true_with_ref, 'nuc'), np.linalg.matrix_rank(true_with_ref)],
            ['unphased', np.linalg.norm(unphased_haplotypes, 'nuc'), np.linalg.matrix_rank(unphased_haplotypes)],
            ['phased', np.linalg.norm(phased_haplotypes, 'nuc'), np.linalg.matrix_rank(phased_haplotypes)],
            ['rounded', np.linalg.norm(rounded, 'nuc'), np.linalg.matrix_rank(rounded)]]
    print(tabulate(data, headers=headers))

    # histogram of absolute values
    bins = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2])
    headers = bins[1:]
    data = [['true', *np.histogram(abs(true_with_ref), bins=bins)[0]],
            ['unphased', *np.histogram(abs(unphased_haplotypes), bins=bins)[0]],
            ['phased', *np.histogram(abs(phased_haplotypes), bins=bins)[0]],
            ['rounded', *np.histogram(abs(rounded), bins=bins)[0]]]
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
    num_snps = int(sys.argv[1])
    mu = float(sys.argv[2])
    num_ref = int(sys.argv[3])
    main(num_snps, mu, num_ref)

# TODO nthomas: add switch error
# we can't really mess with switch error until we've set all the 0s to nonzero values
# print(switch_error(phased_haplotypes, true_haplotypes))
# print('switch error')
