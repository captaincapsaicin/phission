import argparse
import random
import time

import numpy as np

from msprime_simulator import simulate_haplotype_matrix, compress_to_genotype_matrix, get_incomplete_phasing_matrix
from phission import phission_phase
from utils import print_stats, flip_columns


def main(num_haps,
         num_snps,
         num_ref=0,
         Ne=1e5,
         length=5e3,
         recombination_rate=2e-8,
         mutation_rate=2e-8,
         random_seed=None,
         flip=False,
         verbose=False,
         print_matrices=False):
    # simulate with msprime
    true_haplotypes = np.array([[]])
    while true_haplotypes.shape[1] < num_snps:
        if verbose:
            print('simulating...')
        true_haplotypes = simulate_haplotype_matrix(num_haps,
                                                    num_snps,
                                                    Ne=Ne,
                                                    length=length,
                                                    recombination_rate=recombination_rate,
                                                    mutation_rate=mutation_rate,
                                                    random_seed=random_seed)

    if flip:
        random.seed(a=random_seed)
        column_list = random.choices([0, 1], k=num_haps)
        true_haplotypes = flip_columns(column_list, true_haplotypes)

    genotypes = compress_to_genotype_matrix(true_haplotypes)
    unphased_haplotypes = get_incomplete_phasing_matrix(genotypes)

    unphased_haplotypes = np.vstack([unphased_haplotypes, true_haplotypes[0:num_ref]])
    true_with_ref = np.vstack([true_haplotypes, true_haplotypes[0:num_ref]])

    phased_haplotypes = phission_phase(unphased_haplotypes)

    if verbose:
        print_stats(true_with_ref, unphased_haplotypes, phased_haplotypes, print_matrices=print_matrices)
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
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--print-matrices', action='store_true')
    args = parser.parse_args()

    main(args.num_haps,
         args.num_snps,
         args.num_ref,
         args.Ne,
         args.length,
         args.recombination_rate,
         args.mutation_rate,
         args.seed,
         args.flip,
         verbose=True,
         print_matrices=args.print_matrices)

    print('time elapsed:')
    print(time.time() - start_time)
