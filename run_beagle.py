import os
import subprocess

import msprime

from switch_error import switch_error
from utils import read_haplotype_matrix_from_vcf

input_vcf = 'beagle/input.vcf'
output_file = 'beagle/beagle_output'

beagle_path = 'beagle.27Jan18.7e1.jar'


def cleanup():
    """
    Cleanup files so we don't have to do any overrite (especially beagle)
    May want to remove the cleanup step for debugging
    """
    os.remove(input_vcf)
    os.remove(output_file + '.vcf')


def main(verbose=False, seed=None):
    tree_sequence = msprime.simulate(
         sample_size=100,
         Ne=1e5,
         length=5e3,
         recombination_rate=2e-8,
         mutation_rate=2e-8,
         random_seed=seed)

    with open(input_vcf, 'w') as f:
        tree_sequence.write_vcf(f, ploidy=2)

    # make it fully unphased replace | -> /
    subprocess.check_call('sed -i.bak "s/|/\//g" {}'.format(input_vcf), shell=True)
    # call beagle, suppress stdout
    beagle_command = 'java -jar {} gt={} out={}'.format(beagle_path, input_vcf, output_file)
    if not verbose:
        beagle_command += ' >/dev/null'
    subprocess.check_call(beagle_command, shell=True)

    beagle_output = output_file + '.vcf.gz'
    # probably don't need this and can read directly from gzipped .vcf.gz
    subprocess.check_call('gunzip {}'.format(beagle_output), shell=True)
    beagle_output_vcf = output_file + '.vcf'

    true_haplotypes = tree_sequence.genotype_matrix().T
    phased_haplotypes = read_haplotype_matrix_from_vcf(beagle_output_vcf)

    print('switch error')
    print(switch_error(phased_haplotypes, true_haplotypes))
    return true_haplotypes, phased_haplotypes

if __name__ == '__main__':
    # make sure there is a place for beagle output
    try:
        os.mkdir('beagle')
    except FileExistsError:
        pass
    main()
    cleanup()
