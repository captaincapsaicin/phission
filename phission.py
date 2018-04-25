import argparse
import time

import numpy as np
from tabulate import tabulate

from msprime_simulator import simulate_haplotype_matrix, compress_to_genotype_matrix, get_incomplete_phasing_matrix
from nuclear_norm_minimization import phase
from switch_error import switch_error


def main(num_haps, num_snps, num_ref, Ne, length, recombination_rate, mutation_rate, random_seed):
    # simulate with msprime
    all_haplotypes = simulate_haplotype_matrix(num_haps,
                                               Ne=Ne,
                                               length=length,
                                               recombination_rate=recombination_rate,
                                               mutation_rate=mutation_rate,
                                               random_seed=random_seed)
    while all_haplotypes.shape[1] < num_snps:
        print('resimulating...')
        all_haplotypes = simulate_haplotype_matrix(num_haps,
                                                   Ne=Ne,
                                                   length=length,
                                                   recombination_rate=recombination_rate,
                                                   mutation_rate=mutation_rate,
                                                   random_seed=random_seed)

    print('haplotype dimension')
    true_haplotypes = all_haplotypes[:, 0:num_snps]
    print(true_haplotypes.shape)

    genotypes = compress_to_genotype_matrix(true_haplotypes)
    unphased_haplotypes = get_incomplete_phasing_matrix(genotypes)

    print('{} reference haplotypes added'.format(num_ref))
    unphased_haplotypes = np.vstack([unphased_haplotypes, true_haplotypes[0:num_ref]])
    true_with_ref = np.vstack([true_haplotypes, true_haplotypes[0:num_ref]])
    print('unphased dimension {}'.format(unphased_haplotypes.shape))

    phased_haplotypes = phase(unphased_haplotypes)

    headers = ['nuclear norm', 'rank', 'normalized frob distance']
    data = [['true', np.linalg.norm(true_with_ref, 'nuc'), np.linalg.matrix_rank(true_with_ref), np.linalg.norm(true_with_ref - true_with_ref)/np.linalg.norm(true_with_ref)],
            ['unphased', np.linalg.norm(unphased_haplotypes, 'nuc'), np.linalg.matrix_rank(unphased_haplotypes), np.linalg.norm(unphased_haplotypes - true_with_ref)/np.linalg.norm(true_with_ref)],
            ['phased', np.linalg.norm(phased_haplotypes, 'nuc'), np.linalg.matrix_rank(phased_haplotypes), np.linalg.norm(phased_haplotypes - true_with_ref)/np.linalg.norm(true_with_ref)]]
    print(tabulate(data, headers=headers))

    num_phased = np.sum(np.logical_and(unphased_haplotypes == -1, phased_haplotypes != -1))
    print('Positions phased:')
    print(num_phased)

    print('switch error')
    print(switch_error(phased_haplotypes, true_with_ref))
    print('percent switch error')
    num_phased = (np.sum(unphased_haplotypes == -1)) / 2
    print(switch_error(phased_haplotypes, true_with_ref) / num_phased)

    return true_haplotypes, phased_haplotypes

if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Phase!')
    parser.add_argument('--num-haps', type=int, help='number of haplotypes to simulate (2*individuals)')
    parser.add_argument('--num-snps', type=int, help='number of snps to include')
    parser.add_argument('--num-ref', type=int, default=0, help='number of true reference haplotypes to append')
    parser.add_argument('--Ne', type=float, default=1e5, help='effective population size (msprime parameter)')
    parser.add_argument('--length', type=float, default=5e3, help='haplotype length (msprime parameter)')
    parser.add_argument('--recombination-rate', type=float, default=2e-8, help='recombination rate (msprime parameter)')
    parser.add_argument('--mutation-rate', type=float, default=2e-8, help='mutation rate (msprime parameter)')
    parser.add_argument('--seed', type=int, default=None, help='random seed (msprime parameter)')

    args = parser.parse_args()
    main(args.num_haps,
         args.num_snps,
         args.num_ref,
         args.Ne,
         args.length,
         args.recombination_rate,
         args.mutation_rate,
         args.seed)

    print('time elapsed:')
    print(time.time() - start_time)
