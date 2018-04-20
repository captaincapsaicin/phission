import subprocess

import msprime

from utils import read_haplotype_matrix_from_vcf

input_vcf = 'input.vcf'
output_file = 'beagle_output'

beagle_path = 'beagle.27Jan18.7e1.jar'


def main():
    tree_sequence = msprime.simulate(
         sample_size=100,
         Ne=1e5,
         length=5e3,
         recombination_rate=2e-8,
         mutation_rate=2e-8,
         random_seed=10)

    with open(input_vcf, 'w') as f:
        tree_sequence.write_vcf(f, ploidy=2)

    subprocess.check_call('java -jar {} gt={} out={}'.format(beagle_path, input_vcf, output_file), shell=True)
    beagle_output = output_file + '.vcf.gz'
    # probably don't need this and can read directly from gzipped .vcf.gz
    subprocess.check_call('gunzip {}'.format(beagle_output), shell=True)
    beagle_output_vcf = output_file + '.vcf'

    haplotype_matrix = tree_sequence.genotype_matrix().T
    recovered_haplotype_matrix = read_haplotype_matrix_from_vcf(beagle_output_vcf)




if __name__ == '__main__':
    main()
