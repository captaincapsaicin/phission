import argparse
import pickle
import random
import time
import multiprocessing
from functools import partial
import uuid

import numpy as np

from msprime_simulator import simulate_haplotype_matrix, compress_to_genotype_matrix, get_incomplete_phasing_matrix
from phission import phission_phase
from run_beagle import beagle_phase
from utils import switch_error, flip_columns, write_vcf_from_haplotype_matrix


BEAGLE_OUTPUT_DIR = 'beagle'
BEAGLE_JAR_PATH = 'beagle.27Jan18.7e1.jar'


def run_for_params(random_seed,
                   num_haps,
                   num_snps,
                   Ne,
                   length,
                   recombination_rate,
                   mutation_rate):
        """
        return dictionary of stats
        """
        beagle_dict = {}
        phission_dict = {}
        time_start = time.time()

        random.seed(a=random_seed)
        print('Running {}'.format(random_seed))
        true_original_haplotypes = simulate_haplotype_matrix(num_haps,
                                                             num_snps,
                                                             Ne=Ne,
                                                             length=length,
                                                             recombination_rate=recombination_rate,
                                                             mutation_rate=mutation_rate,
                                                             random_seed=random_seed)
        # randomly flip some number of column conventions
        column_list = random.choices([0, 1], k=num_haps)
        true_haplotypes = flip_columns(column_list, true_original_haplotypes)
        uuid_suffix = str(uuid.uuid4())
        beagle_input_vcf_filename = '/'.join([BEAGLE_OUTPUT_DIR, 'input_{}.vcf'.format(uuid_suffix)])
        write_vcf_from_haplotype_matrix(beagle_input_vcf_filename, true_haplotypes, phased=False)
        genotypes = compress_to_genotype_matrix(true_haplotypes)
        unphased_haplotypes = get_incomplete_phasing_matrix(genotypes)

        time_start = time.time()
        phased_haplotypes = phission_phase(unphased_haplotypes)

        phission_dict['rank_true'] = np.linalg.matrix_rank(true_haplotypes)
        phission_dict['rank_phased'] = np.linalg.matrix_rank(phased_haplotypes)
        phission_dict['nuclear_norm_true'] = np.linalg.norm(true_haplotypes, 'nuc')
        phission_dict['nuclear_norm_phased'] = np.linalg.norm(phased_haplotypes, 'nuc')
        phission_dict['switch_error'] = switch_error(phased_haplotypes, true_haplotypes)
        phission_dict['positions_phased'] = np.sum(unphased_haplotypes == -1) / 2
        phission_dict['time_to_phase'] = time.time() - time_start

        time_start = time.time()
        beagle_output_path = '/'.join([BEAGLE_OUTPUT_DIR, 'beagle_output_{}'.format(uuid_suffix)])
        phased_haplotypes = beagle_phase(BEAGLE_JAR_PATH, beagle_input_vcf_filename, beagle_output_path)

        beagle_dict['rank_true'] = np.linalg.matrix_rank(true_haplotypes)
        beagle_dict['rank_phased'] = np.linalg.matrix_rank(phased_haplotypes)
        beagle_dict['nuclear_norm_true'] = np.linalg.norm(true_haplotypes, 'nuc')
        beagle_dict['nuclear_norm_phased'] = np.linalg.norm(phased_haplotypes, 'nuc')
        beagle_dict['switch_error'] = switch_error(phased_haplotypes, true_haplotypes)
        beagle_dict['positions_phased'] = np.sum(unphased_haplotypes == -1) / 2
        beagle_dict['time_to_phase'] = time.time() - time_start

        return (phission_dict, beagle_dict)


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
        phission_stats[(num_haps, num_snps)]['rank_true'] = np.zeros(num_experiments)
        phission_stats[(num_haps, num_snps)]['rank_phased'] = np.zeros(num_experiments)
        phission_stats[(num_haps, num_snps)]['nuclear_norm_true'] = np.zeros(num_experiments)
        phission_stats[(num_haps, num_snps)]['nuclear_norm_phased'] = np.zeros(num_experiments)
        phission_stats[(num_haps, num_snps)]['switch_error'] = np.zeros(num_experiments)
        phission_stats[(num_haps, num_snps)]['positions_phased'] = np.zeros(num_experiments)
        phission_stats[(num_haps, num_snps)]['time_to_phase'] = np.zeros(num_experiments)
        beagle_stats[(num_haps, num_snps)]['rank_true'] = np.zeros(num_experiments)
        beagle_stats[(num_haps, num_snps)]['rank_phased'] = np.zeros(num_experiments)
        beagle_stats[(num_haps, num_snps)]['nuclear_norm_true'] = np.zeros(num_experiments)
        beagle_stats[(num_haps, num_snps)]['nuclear_norm_phased'] = np.zeros(num_experiments)
        beagle_stats[(num_haps, num_snps)]['switch_error'] = np.zeros(num_experiments)
        beagle_stats[(num_haps, num_snps)]['positions_phased'] = np.zeros(num_experiments)
        beagle_stats[(num_haps, num_snps)]['time_to_phase'] = np.zeros(num_experiments)

    for num_haps, num_snps in num_haps_snps_list:
        print((num_haps, num_snps))
        phission_dict = phission_stats[(num_haps, num_snps)]
        beagle_dict = beagle_stats[(num_haps, num_snps)]

        run = partial(run_for_params,
                      num_haps=num_haps,
                      num_snps=num_snps,
                      Ne=Ne,
                      length=length,
                      recombination_rate=recombination_rate,
                      mutation_rate=mutation_rate)
        with multiprocessing.Pool() as pool:
            results = pool.map(run, range(1, num_experiments + 1))
        for i in range(num_experiments):
            phission_dict, beagle_dict = results[i]
            phission_stats[(num_haps, num_snps)]['rank_true'][i] = phission_dict['rank_true']
            phission_stats[(num_haps, num_snps)]['rank_phased'][i] = phission_dict['rank_phased']
            phission_stats[(num_haps, num_snps)]['nuclear_norm_true'][i] = phission_dict['nuclear_norm_true']
            phission_stats[(num_haps, num_snps)]['nuclear_norm_phased'][i] = phission_dict['nuclear_norm_phased']
            phission_stats[(num_haps, num_snps)]['switch_error'][i] = phission_dict['switch_error']
            phission_stats[(num_haps, num_snps)]['positions_phased'][i] = phission_dict['positions_phased']
            phission_stats[(num_haps, num_snps)]['time_to_phase'][i] = phission_dict['time_to_phase']
            beagle_stats[(num_haps, num_snps)]['rank_true'][i] = beagle_dict['rank_true']
            beagle_stats[(num_haps, num_snps)]['rank_phased'][i] = beagle_dict['rank_phased']
            beagle_stats[(num_haps, num_snps)]['nuclear_norm_true'][i] = beagle_dict['nuclear_norm_true']
            beagle_stats[(num_haps, num_snps)]['nuclear_norm_phased'][i] = beagle_dict['nuclear_norm_phased']
            beagle_stats[(num_haps, num_snps)]['switch_error'][i] = beagle_dict['switch_error']
            beagle_stats[(num_haps, num_snps)]['positions_phased'][i] = beagle_dict['positions_phased']
            beagle_stats[(num_haps, num_snps)]['time_to_phase'][i] = beagle_dict['time_to_phase']
    return phission_stats, beagle_stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase!')
    parser.add_argument('--num-experiments', type=int, help='number of experiments to run for each (hap, snp) pair')
    parser.add_argument('--Ne', type=float, default=1e5, help='effective population size (msprime parameter)')
    parser.add_argument('--length', type=float, default=5e3, help='haplotype length (msprime parameter)')
    parser.add_argument('--recombination-rate', type=float, default=2e-8, help='recombination rate (msprime parameter)')
    parser.add_argument('--mutation-rate', type=float, default=2e-8, help='mutation rate (msprime parameter)')
    parser.add_argument('--haps-snps', type=str, nargs='+', help='haps,snps for each experiment separated by a space')
    args = parser.parse_args()

    # I'm just setting a default for this for now.
    num_haps_snps_list = [(int(pair.split(',')[0]), int(pair.split(',')[1])) for pair in args.haps_snps]
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
