import argparse
import pickle
import time

import numpy as np

from msprime_simulator import compress_to_genotype_matrix, get_incomplete_phasing_matrix
from utils import switch_error

from phission import main as run_phission
from beagle import cleanup as beagle_cleanup, main as run_beagle


def main(num_experiments, num_haps_snps_list, recombination_rate=0.0):
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
        for seed in range(1, num_experiments):
            print('Running {}'.format(seed))
            time_start = time.time()
            true_haplotypes, phased_haplotypes = run_phission(num_snps,
                                                              num_haps,
                                                              recombination_rate=recombination_rate,
                                                              random_seed=seed)
            genotypes = compress_to_genotype_matrix(true_haplotypes)
            unphased_haplotypes = get_incomplete_phasing_matrix(genotypes)

            phission_dict = phission_stats[(num_haps, num_snps)]
            phission_dict['rank_true'].append(np.linalg.matrix_rank(true_haplotypes))
            phission_dict['rank_phased'].append(np.linalg.matrix_rank(phased_haplotypes))
            phission_dict['nuclear_norm_true'].append(np.linalg.norm(true_haplotypes, 'nuc'))
            phission_dict['nuclear_norm_phased'].append(np.linalg.norm(phased_haplotypes, 'nuc'))
            phission_dict['switch_error'].append(switch_error(phased_haplotypes, true_haplotypes))
            phission_dict['positions_phased'].append(np.sum(unphased_haplotypes == -1) / 2)
            phission_dict['time_to_phase'].append(time.time() - time_start)

            time_start = time.time()
            true_haplotypes, phased_haplotypes = run_beagle(num_snps,
                                                            num_haps,
                                                            recombination_rate=recombination_rate,
                                                            random_seed=seed)

            genotypes = compress_to_genotype_matrix(true_haplotypes)
            unphased_haplotypes = get_incomplete_phasing_matrix(genotypes)

            beagle_dict = beagle_stats[(num_haps, num_snps)]
            beagle_dict['rank_true'].append(np.linalg.matrix_rank(true_haplotypes))
            beagle_dict['rank_phased'].append(np.linalg.matrix_rank(phased_haplotypes))
            beagle_dict['nuclear_norm_true'].append(np.linalg.norm(true_haplotypes, 'nuc'))
            beagle_dict['nuclear_norm_phased'].append(np.linalg.norm(phased_haplotypes, 'nuc'))
            beagle_dict['switch_error'].append(switch_error(phased_haplotypes, true_haplotypes))
            beagle_dict['positions_phased'].append(np.sum(unphased_haplotypes == -1) / 2)
            beagle_dict['time_to_phase'].append(time.time() - time_start)
            beagle_cleanup()

    return phission_stats, beagle_stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase!')
    parser.add_argument('--num-experiments', type=int, help='number of experiments to run for each (hap, snp) pair')
    args = parser.parse_args()

    num_haps_snps_list = [(4, 4),
                          (10, 10),
                          (10, 20),
                          (20, 10),
                          (20, 20),
                          (20, 40),
                          (40, 10),
                          (40, 20),
                          (40, 40),
                          (40, 80),
                          (80, 10),
                          (80, 20),
                          (80, 40),
                          (80, 80),
                          (80, 160),
                          (160, 10),
                          (160, 20),
                          (160, 40),
                          (160, 80),
                          (160, 160)]

    phission_stats, beagle_stats = main(args.num_experiments, num_haps_snps_list)
    with open('phission_stats.pkl', 'wb') as f:
        pickle.dump(phission_stats, f)
    with open('beagle_stats.pkl', 'wb') as f:
        pickle.dump(beagle_stats, f)


# Can get pickle data back with
# with open('beagle_stats.pkl', 'rb') as f:
#     beagle_stats = pickle.load(f)
