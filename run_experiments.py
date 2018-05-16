import argparse
import pickle
import random
import time

import numpy as np

from msprime_simulator import simulate_haplotype_matrix, compress_to_genotype_matrix, get_incomplete_phasing_matrix
from utils import switch_error, flip_columns, write_vcf_from_haplotype_matrix

from phission import phission_phase
from run_beagle import (beagle_phase,
                        INPUT_VCF,
                        BEAGLE_JAR_PATH,
                        BEAGLE_OUTPUT_PATH)


def main(num_experiments,
         num_haps_snps_list,
         Ne,
         length,
         recombination_rate,
         mutation_rate):
    phission_stats = {}
    beagle_stats = {}
    for num_haps, num_snps in num_haps_snps_list:
        phission_stats[(num_haps, num_snps)] = {}
        beagle_stats[(num_haps, num_snps)] = {}
        phission_stats[(num_haps, num_snps)]['rank_true'] = []
        phission_stats[(num_haps, num_snps)]['rank_phased'] = []
        phission_stats[(num_haps, num_snps)]['nuclear_norm_true'] = []
        phission_stats[(num_haps, num_snps)]['nuclear_norm_phased'] = []
        phission_stats[(num_haps, num_snps)]['switch_error'] = []
        phission_stats[(num_haps, num_snps)]['positions_phased'] = []
        phission_stats[(num_haps, num_snps)]['time_to_phase'] = []
        beagle_stats[(num_haps, num_snps)]['rank_true'] = []
        beagle_stats[(num_haps, num_snps)]['rank_phased'] = []
        beagle_stats[(num_haps, num_snps)]['nuclear_norm_true'] = []
        beagle_stats[(num_haps, num_snps)]['nuclear_norm_phased'] = []
        beagle_stats[(num_haps, num_snps)]['switch_error'] = []
        beagle_stats[(num_haps, num_snps)]['positions_phased'] = []
        beagle_stats[(num_haps, num_snps)]['time_to_phase'] = []

    for num_haps, num_snps in num_haps_snps_list:
        print((num_haps, num_snps))
        phission_dict = phission_stats[(num_haps, num_snps)]
        beagle_dict = beagle_stats[(num_haps, num_snps)]
        for random_seed in range(1, num_experiments + 1):
            random.seed(a=random_seed)
            print('Running {}'.format(random_seed))
            true_original_haplotypes = simulate_haplotype_matrix(num_haps,
                                                                 num_snps,
                                                                 Ne=Ne,
                                                                 length=length,
                                                                 recombination_rate=recombination_rate,
                                                                 mutation_rate=mutation_rate,
                                                                 random_seed=random_seed)
            for i in range(10):
                # randomly flip some number of column conventions
                column_list = random.choices([0, 1], k=num_haps)
                true_haplotypes = flip_columns(column_list, true_original_haplotypes)
                write_vcf_from_haplotype_matrix(INPUT_VCF, true_haplotypes, phased=False)
                genotypes = compress_to_genotype_matrix(true_haplotypes)
                unphased_haplotypes = get_incomplete_phasing_matrix(genotypes)

                time_start = time.time()
                phased_haplotypes = phission_phase(unphased_haplotypes)

                phission_dict['rank_true'].append(np.linalg.matrix_rank(true_haplotypes))
                phission_dict['rank_phased'].append(np.linalg.matrix_rank(phased_haplotypes))
                phission_dict['nuclear_norm_true'].append(np.linalg.norm(true_haplotypes, 'nuc'))
                phission_dict['nuclear_norm_phased'].append(np.linalg.norm(phased_haplotypes, 'nuc'))
                phission_dict['switch_error'].append(switch_error(phased_haplotypes, true_haplotypes))
                phission_dict['positions_phased'].append(np.sum(unphased_haplotypes == -1) / 2)
                phission_dict['time_to_phase'].append(time.time() - time_start)

                time_start = time.time()
                phased_haplotypes = beagle_phase(BEAGLE_JAR_PATH, INPUT_VCF, BEAGLE_OUTPUT_PATH)

                beagle_dict['rank_true'].append(np.linalg.matrix_rank(true_haplotypes))
                beagle_dict['rank_phased'].append(np.linalg.matrix_rank(phased_haplotypes))
                beagle_dict['nuclear_norm_true'].append(np.linalg.norm(true_haplotypes, 'nuc'))
                beagle_dict['nuclear_norm_phased'].append(np.linalg.norm(phased_haplotypes, 'nuc'))
                beagle_dict['switch_error'].append(switch_error(phased_haplotypes, true_haplotypes))
                beagle_dict['positions_phased'].append(np.sum(unphased_haplotypes == -1) / 2)
                beagle_dict['time_to_phase'].append(time.time() - time_start)

        # convert to numpy arrays
        phission_dict['rank_true'] = np.array(phission_dict['rank_true'])
        phission_dict['rank_phased'] = np.array(phission_dict['rank_phased'])
        phission_dict['nuclear_norm_true'] = np.array(phission_dict['nuclear_norm_true'])
        phission_dict['nuclear_norm_phased'] = np.array(phission_dict['nuclear_norm_phased'])
        phission_dict['switch_error'] = np.array(phission_dict['switch_error'])
        phission_dict['positions_phased'] = np.array(phission_dict['positions_phased'])
        phission_dict['time_to_phase'] = np.array(phission_dict['time_to_phase'])
        beagle_dict['rank_true'] = np.array(beagle_dict['rank_true'])
        beagle_dict['rank_phased'] = np.array(beagle_dict['rank_phased'])
        beagle_dict['nuclear_norm_true'] = np.array(beagle_dict['nuclear_norm_true'])
        beagle_dict['nuclear_norm_phased'] = np.array(beagle_dict['nuclear_norm_phased'])
        beagle_dict['switch_error'] = np.array(beagle_dict['switch_error'])
        beagle_dict['positions_phased'] = np.array(beagle_dict['positions_phased'])
        beagle_dict['time_to_phase'] = np.array(beagle_dict['time_to_phase'])
    return phission_stats, beagle_stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase!')
    parser.add_argument('--num-experiments', type=int, help='number of experiments to run for each (hap, snp) pair')
    parser.add_argument('--Ne', type=float, default=1e5, help='effective population size (msprime parameter)')
    parser.add_argument('--length', type=float, default=5e3, help='haplotype length (msprime parameter)')
    parser.add_argument('--recombination-rate', type=float, default=2e-8, help='recombination rate (msprime parameter)')
    parser.add_argument('--mutation-rate', type=float, default=2e-8, help='mutation rate (msprime parameter)')
    args = parser.parse_args()

    # num_haps_snps_list = [(4, 4),
    #                       (10, 10),
    #                       (10, 20),
    #                       (20, 10),
    #                       (20, 20),
    #                       (20, 40),
    #                       (40, 10),
    #                       (40, 20),
    #                       (40, 40),
    #                       (40, 80),
    #                       (80, 10),
    #                       (80, 20),
    #                       (80, 40),
    #                       (80, 80),
    #                       (80, 160),
    #                       (160, 10),
    #                       (160, 20),
    #                       (160, 40),
    #                       (160, 80),
    #                       (160, 160)]

    # I'm just setting a default for this for now.
    num_haps_snps_list = [(100, 10), (100, 20), (500, 10), (500, 20), (1000, 10), (1000, 20)]
    phission_stats, beagle_stats = main(args.num_experiments,
                                        num_haps_snps_list,
                                        args.Ne,
                                        args.length,
                                        args.recombination_rate,
                                        args.mutation_rate)
    with open('phission_stats.pkl', 'wb') as f:
        pickle.dump(phission_stats, f)
    with open('beagle_stats.pkl', 'wb') as f:
        pickle.dump(beagle_stats, f)
