import numpy as np

from msprime_simulator import simulate_haplotype_matrix, compress_to_genotype_matrix, get_incomplete_phasing_matrix
from nuclear_norm_minimization import phase, get_mask, nuclear_norm_solve
from switch_error import switch_error


def main():
    # simulate with msprime
    haplotypes = simulate_haplotype_matrix(100)
    print('haplotype dimension')
    print(haplotypes.shape)
    genotypes = compress_to_genotype_matrix(haplotypes)
    unphased_haplotypes = get_incomplete_phasing_matrix(genotypes)

    mask = get_mask(unphased_haplotypes)
    phased_haplotypes = nuclear_norm_solve(unphased_haplotypes, mask, mu=0.1)
    rounded = np.matrix.round(phased_haplotypes)
    # converting 0/1 representation to -1/1
    true_haplotypes = -1*(haplotypes == 0).astype(int) + 1*(haplotypes == 1).astype(int)

    # maybe we'll use something like thresholding later
    thresh = -1*(phased_haplotypes < 0) + 1*(phased_haplotypes > 0)

    print('comparing nuclear norm...')
    print('true haplotypes:')
    print(np.linalg.norm(true_haplotypes, 'nuc'))
    print('unphased haplotypes:')
    print(np.linalg.norm(unphased_haplotypes, 'nuc'))
    print('phased haplotypes (post optimization):')
    print(np.linalg.norm(phased_haplotypes, 'nuc'))
    print('And if we round')
    print(np.linalg.norm(rounded, 'nuc'))

    print('comparing rank...')
    print('true haplotypes:')
    print(np.linalg.matrix_rank(true_haplotypes))
    print('unphased haplotypes:')
    print(np.linalg.matrix_rank(unphased_haplotypes))
    print('phased haplotypes (post optimization):')
    print(np.linalg.matrix_rank(phased_haplotypes))
    print('And if we round')
    print(np.linalg.matrix_rank(rounded))

    print('For sanity, lets see how many 0s we set to phased values')
    print(np.sum(np.logical_and(unphased_haplotypes == 0, rounded != 0)))
    print('and how many nonzeros we set to 0')
    print(np.sum(np.logical_and(unphased_haplotypes != 0, rounded == 0)))
    print('and how many 0s overall in our answer')
    print(np.sum(rounded == 0))


if __name__ == '__main__':
    main()

# we can't really mess with switch error until we've set all the 0s to nonzero values
# print(switch_error(phased_haplotypes, true_haplotypes))
# print('switch error')
