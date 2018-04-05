import numpy as np

from msprime_simulator import simulate_haplotype_matrix, compress_to_genotype_matrix, get_incomplete_phasing_matrix
from nuclear_norm_minimization import phase, get_mask, nuclear_norm_solve
from switch_error import switch_error

# simulate with msprime
haplotypes = simulate_haplotype_matrix(100)
print('haplotype dimension')
print(haplotypes.shape)
genotypes = compress_to_genotype_matrix(haplotypes)
unphased_haplotypes = get_incomplete_phasing_matrix(genotypes)

mask = get_mask(unphased_haplotypes)
phased_haplotypes = nuclear_norm_solve(unphased_haplotypes, mask, mu=20)

true_haplotypes = haplotypes
# converting 0/1 representation to -1/1
# true_haplotypes = -1*(haplotypes == 0).astype(int) + 1*(haplotypes == 1).astype(int)

print('comparing nuclear norm...')
print('true haplotypes:')
print(np.linalg.norm(true_haplotypes, 'nuc'))
print('unphased haplotypes:')
print(np.linalg.norm(unphased_haplotypes, 'nuc'))
print('phased haplotypes (post optimization):')
print(np.linalg.norm(phased_haplotypes, 'nuc'))
print('And if we round')
print(np.linalg.norm(np.matrix.round(phased_haplotypes), 'nuc'))

print('comparing rank...')
print('true haplotypes:')
print(np.linalg.matrix_rank(true_haplotypes))
print('unphased haplotypes:')
print(np.linalg.matrix_rank(unphased_haplotypes))
print('phased haplotypes (post optimization):')
print(np.linalg.matrix_rank(phased_haplotypes))
print('And if we round')
print(np.linalg.matrix_rank(np.matrix.round(phased_haplotypes)))

print('For sanity, lets see how many 2s we set to phased values')
print(np.sum(unphased_haplotypes == 2) - np.sum(np.matrix.round(phased_haplotypes) == 2))
print('and how many 2s remain')
print(np.sum(np.matrix.round(phased_haplotypes) == 2))


phased_haplotypes = phase(unphased_haplotypes, mu=2)
print('switch error')
print(switch_error(phased_haplotypes, true_haplotypes))
