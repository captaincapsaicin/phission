import argparse
import os
import subprocess

import msprime

from switch_error import switch_error
from utils import read_haplotype_matrix_from_vcf

beagle_output_dir = 'beagle'
input_vcf = '/'.join([beagle_output_dir, 'input.vcf'])
beagle_output_path = '/'.join([beagle_output_dir, 'beagle_output'])
beagle_jar_path = 'beagle.27Jan18.7e1.jar'


def cleanup():
    """
    Cleanup files so we don't have to do any overrite (especially beagle)
    May want to remove the cleanup step for debugging
    """
    os.remove(input_vcf)
    os.remove(beagle_output_path + '.vcf')


def unphase_vcf(input_vcf):
    # make it fully unphased replace | -> /
    subprocess.check_call('sed -i.bak "s/|/\//g" {}'.format(input_vcf), shell=True)


def run_beagle(beagle_jar_path, input_vcf, beagle_output_path, verbose):
    # call beagle, suppress stdout
    beagle_command = 'java -jar {} gt={} out={}'.format(beagle_jar_path, input_vcf, beagle_output_path)
    if not verbose:
        beagle_command += ' >/dev/null'
    subprocess.check_call(beagle_command, shell=True)
    beagle_output = beagle_output_path + '.vcf.gz'
    # probably don't need this and can read directly from gzipped .vcf.gz
    subprocess.check_call('gunzip {}'.format(beagle_output), shell=True)


def remove_n_lines(vcf_file, n, output_file):
    with open(vcf_file, 'r') as f:
        lines = list(f)
    lines = lines[:-n]
    with open(output_file, 'w') as f:
        f.write(''.join(lines))


def main(num_haps, num_snps, verbose=False, seed=None):
    tree_sequence = msprime.simulate(
         sample_size=num_haps,
         Ne=1e5,
         length=5e3,
         recombination_rate=2e-8,
         mutation_rate=2e-8,
         random_seed=seed)

    with open(input_vcf, 'w') as f:
        tree_sequence.write_vcf(f, ploidy=2)

    true_haplotypes = tree_sequence.genotype_matrix().T
    num_simulated_snps = true_haplotypes.shape[1]
    num_snps_to_remove = num_simulated_snps - num_snps

    unphase_vcf(input_vcf)
    new_input_vcf = 'beagle/new_input_vcf.vcf'
    remove_n_lines(input_vcf, num_snps_to_remove, new_input_vcf)
    run_beagle(beagle_jar_path, new_input_vcf, beagle_output_path, verbose)
    beagle_output_vcf = beagle_output_path + '.vcf'
    phased_haplotypes = read_haplotype_matrix_from_vcf(beagle_output_vcf)

    print('haplotype dimension')
    true_haplotypes = true_haplotypes[:, 0:num_snps]
    print(true_haplotypes.shape)
    print('switch error')
    print(switch_error(phased_haplotypes, true_haplotypes))
    return true_haplotypes, phased_haplotypes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase!')
    parser.add_argument('--num-haps', type=int, default=100, help='number of haplotypes to simulate (2*individuals)')
    parser.add_argument('--num-snps', type=int, help='number of snps to include')

    args = parser.parse_args()

    # make sure there is a place for beagle output
    try:
        os.mkdir(beagle_output_dir)
    except FileExistsError:
        pass

    main(args.num_haps, args.num_snps)
    cleanup()
