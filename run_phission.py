import argparse
import time

import numpy as np

from msprime_simulator import simulate_haplotype_matrix, compress_to_genotype_matrix, get_incomplete_phasing_matrix
from phission import phission_phase
from utils import print_stats


def main(num_haps,
         num_snps,
         num_ref=0,
         Ne=1e5,
         length=5e3,
         recombination_rate=2e-8,
         mutation_rate=2e-8,
         random_seed=None,
         verbose=False):
    # simulate with msprime
    all_haplotypes = simulate_haplotype_matrix(num_haps,
                                               Ne=Ne,
                                               length=length,
                                               recombination_rate=recombination_rate,
                                               mutation_rate=mutation_rate,
                                               random_seed=random_seed)
    while all_haplotypes.shape[1] < num_snps:
        if verbose:
            print('resimulating...')
        all_haplotypes = simulate_haplotype_matrix(num_haps,
                                                   Ne=Ne,
                                                   length=length,
                                                   recombination_rate=recombination_rate,
                                                   mutation_rate=mutation_rate,
                                                   random_seed=random_seed)

    true_haplotypes = all_haplotypes[:, 0:num_snps]

    genotypes = compress_to_genotype_matrix(true_haplotypes)
    unphased_haplotypes = get_incomplete_phasing_matrix(genotypes)

    unphased_haplotypes = np.vstack([unphased_haplotypes, true_haplotypes[0:num_ref]])
    true_with_ref = np.vstack([true_haplotypes, true_haplotypes[0:num_ref]])

    phased_haplotypes = phission_phase(unphased_haplotypes)

    if verbose:
        print_stats(true_with_ref, unphased_haplotypes, phased_haplotypes)
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
         args.seed,
         verbose=True)

    print('time elapsed:')
    print(time.time() - start_time)
