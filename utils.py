import allel
import numpy as np


def read_haplotype_matrix_from_vcf(vcf_file):
    """"
    Read the vcf file into a haplotype matrix (numpy)
    assume everything is phased,

    """
    callset = allel.read_vcf(vcf_file)
    gt = callset['calldata/GT']
    hap0 = gt[:, :, 0].T
    hap1 = gt[:, :, 1].T

    haps = np.empty((hap0.shape[0]*2, hap0.shape[1]), dtype=hap0.dtype)
    haps[0::2] = hap0
    haps[1::2] = hap1

    return haps
