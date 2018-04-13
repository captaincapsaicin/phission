import sys

import numpy as np
from tabulate import tabulate

from msprime_simulator import simulate_haplotype_matrix, compress_to_genotype_matrix, get_incomplete_phasing_matrix
from nuclear_norm_minimization import get_mask, nuclear_norm_solve
from switch_error import switch_error


def main(num_snps, mu, num_ref):
    # simulate with msprime
    all_haplotypes = simulate_haplotype_matrix(100)
    while all_haplotypes.shape[1] < num_snps:
        all_haplotypes = simulate_haplotype_matrix(100)
    true_haplotypes = all_haplotypes[:, 0:num_snps]

    print('haplotype dimension')
    print(true_haplotypes.shape)

    genotypes = compress_to_genotype_matrix(true_haplotypes)
    unphased_haplotypes = get_incomplete_phasing_matrix(genotypes)

    print('{} reference haplotypes added'.format(num_ref))
    unphased_haplotypes = np.vstack([unphased_haplotypes, true_haplotypes[0:num_ref]])
    true_with_ref = np.vstack([true_haplotypes, true_haplotypes[0:num_ref]])
    print('unphased dimension {}'.format(unphased_haplotypes.shape))

    # this is what the ".phase" method does
    mask = get_mask(unphased_haplotypes)
    phased_haplotypes = nuclear_norm_solve(unphased_haplotypes, mask, mu=1)
    rounded = np.matrix.round(phased_haplotypes).astype(int)

    headers = ['nuclear norm', 'rank', 'normalized frob distance']
    data = [['true', np.linalg.norm(true_with_ref, 'nuc'), np.linalg.matrix_rank(true_with_ref), np.linalg.norm(true_with_ref - true_with_ref)/np.linalg.norm(true_with_ref)],
            ['unphased', np.linalg.norm(unphased_haplotypes, 'nuc'), np.linalg.matrix_rank(unphased_haplotypes), np.linalg.norm(unphased_haplotypes - true_with_ref)/np.linalg.norm(true_with_ref)],
            ['phased', np.linalg.norm(phased_haplotypes, 'nuc'), np.linalg.matrix_rank(phased_haplotypes), np.linalg.norm(phased_haplotypes - true_with_ref)/np.linalg.norm(true_with_ref)],
            ['rounded', np.linalg.norm(rounded, 'nuc'), np.linalg.matrix_rank(rounded), np.linalg.norm(rounded - true_with_ref)/np.linalg.norm(true_with_ref)]]
    print(tabulate(data, headers=headers))

    # histogram of absolute values
    bins = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2])
    headers = bins[1:]
    data = [['true', *np.histogram(abs(true_with_ref), bins=bins)[0]],
            ['unphased', *np.histogram(abs(unphased_haplotypes), bins=bins)[0]],
            ['phased', *np.histogram(abs(phased_haplotypes), bins=bins)[0]],
            ['rounded', *np.histogram(abs(rounded), bins=bins)[0]]]
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
    print(switch_error(rounded, true_with_ref))
    print('percent switch error')
    num_phased = (np.sum(unphased_haplotypes == -1)) / 2
    print(switch_error(rounded, true_with_ref) / num_phased)


if __name__ == '__main__':
    num_snps = int(sys.argv[1])
    mu = int(sys.argv[2])
    num_ref = int(sys.argv[3])
    main(num_snps, mu, num_ref)
