import argparse
import os
import subprocess
import time

import gzip
import msprime

from msprime_simulator import compress_to_genotype_matrix, get_incomplete_phasing_matrix
from utils import read_haplotype_matrix_from_vcf, print_stats

BEAGLE_OUTPUT_DIR = 'beagle'
INPUT_VCF = '/'.join([BEAGLE_OUTPUT_DIR, 'input.vcf'])
NEW_INPUT_VCF = 'beagle/new_input_vcf.vcf'
BEAGLE_OUTPUT_PATH = '/'.join([BEAGLE_OUTPUT_DIR, 'beagle_output'])
BEAGLE_JAR_PATH = 'beagle.27Jan18.7e1.jar'
BEAGLE_OUTPUT_VCF = BEAGLE_OUTPUT_PATH + '.vcf'


def unphase_vcf(input_vcf):
    # make it fully unphased replace | -> /
    subprocess.check_call('sed -i.bak "s/|/\//g" {}'.format(input_vcf), shell=True)


def beagle_phase(beagle_jar_path, input_vcf, beagle_output_path, verbose=False):
    # call beagle, suppress stdout
    beagle_command = 'java -jar {} gt={} out={}'.format(beagle_jar_path, input_vcf, beagle_output_path)
    if not verbose:
        beagle_command += ' >/dev/null'
    subprocess.check_call(beagle_command, shell=True)
    beagle_output = beagle_output_path + '.vcf.gz'
    with gzip.open(beagle_output, 'rb') as f:
        phased_haplotypes = read_haplotype_matrix_from_vcf(f)
    return phased_haplotypes


def remove_n_lines(vcf_file, n, output_file):
    with open(vcf_file, 'r') as f:
        lines = list(f)
    lines = lines[:-n]
    with open(output_file, 'w') as f:
        f.write(''.join(lines))


def main(num_haps,
         num_snps,
         Ne=1e5,
         length=5e3,
         recombination_rate=2e-8,
         mutation_rate=2e-8,
         random_seed=None,
         verbose=False):
    tree_sequence = msprime.simulate(
         sample_size=num_haps,
         Ne=Ne,
         length=length,
         recombination_rate=recombination_rate,
         mutation_rate=mutation_rate,
         random_seed=random_seed)
    true_haplotypes = tree_sequence.genotype_matrix().T
    while true_haplotypes.shape[1] < num_snps:
        if verbose:
            print('resimulating...')
        tree_sequence = msprime.simulate(
             sample_size=num_haps,
             Ne=Ne,
             length=length,
             recombination_rate=recombination_rate,
             mutation_rate=mutation_rate,
             random_seed=random_seed)
        true_haplotypes = tree_sequence.genotype_matrix().T

    with open(INPUT_VCF, 'w') as f:
        tree_sequence.write_vcf(f, ploidy=2)

    num_simulated_snps = true_haplotypes.shape[1]
    num_snps_to_remove = num_simulated_snps - num_snps

    unphase_vcf(INPUT_VCF)
    remove_n_lines(INPUT_VCF, num_snps_to_remove, NEW_INPUT_VCF)
    phased_haplotypes = beagle_phase(BEAGLE_JAR_PATH, NEW_INPUT_VCF, BEAGLE_OUTPUT_PATH)

    true_haplotypes = true_haplotypes[:, 0:num_snps]

    # constructed for statistics later, not used by beagle, which works directly on vcf
    genotypes = compress_to_genotype_matrix(true_haplotypes)
    unphased_haplotypes = get_incomplete_phasing_matrix(genotypes)

    # TODO nthomas: add some facility to contribute reference haplotypes to beagle

    if verbose:
        print_stats(true_haplotypes, unphased_haplotypes, phased_haplotypes)
    return true_haplotypes, phased_haplotypes

if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Phase!')
    parser.add_argument('--num-snps', type=int, help='number of snps to include')
    parser.add_argument('--num-haps', type=int, default=100, help='number of haplotypes to simulate (2*individuals) (msprime parameter)')
    parser.add_argument('--Ne', type=float, default=1e5, help='effective population size (msprime parameter)')
    parser.add_argument('--length', type=float, default=5e3, help='haplotype length (msprime parameter)')
    parser.add_argument('--recombination-rate', type=float, default=2e-8, help='recombination rate (msprime parameter)')
    parser.add_argument('--mutation-rate', type=float, default=2e-8, help='mutation rate (msprime parameter)')
    parser.add_argument('--seed', type=int, default=None, help='random seed (msprime parameter)')
    args = parser.parse_args()

    # make sure there is a place for beagle output
    try:
        os.mkdir(BEAGLE_OUTPUT_DIR)
    except FileExistsError:
        pass

    main(args.num_haps,
         args.num_snps,
         args.Ne,
         args.length,
         args.recombination_rate,
         args.mutation_rate,
         args.seed,
         verbose=True)

    print('time elapsed:')
    print(time.time() - start_time)
