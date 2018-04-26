import allel
import numpy as np

from tabulate import tabulate


def switch_error(observed, expected):
    """
    Takes in observed haplotype matrix and expected haplotype matrix
    of -1s and 1s and returns

    Returns:
    switch_errors
        the # of switch errors made (a scalar)
    """
    # just start with evens
    observed_i = [i for i in range(observed.shape[0]) if i % 2 == 0]
    expected_i = [i for i in range(expected.shape[0]) if i % 2 == 0]
    switch_errors = 0
    used_free_pass = set()
    for j in range(observed.shape[1]):
        # which individual are we looking at (index = 0 ... n/2)
        for index in range(len(expected_i)):
            o_i = observed_i[index]
            e_i = expected_i[index]
            if observed[o_i, j] != expected[e_i, j]:
                # if we're wrong, we need to switch
                # switch between odd and even, e.g. 0 <-> 1, 24 <-> 25
                observed_i[index] = (o_i // 2) * 2 + 1 - o_i % 2
                # but we give everyone one free pass
                if o_i not in used_free_pass:
                    # we know the first time this occurs,
                    # it will be with an even individual index
                    used_free_pass.add(o_i)
                    used_free_pass.add(o_i + 1)
                else:
                    switch_errors += 1
            # we also revoke your free pass if you've passed your first heterozygous site
            # again which will only occur while o_i is still even
            if o_i not in used_free_pass and observed[o_i, j] != observed[o_i + 1, j]:
                used_free_pass.add(o_i)
                used_free_pass.add(o_i + 1)

    return switch_errors


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


def print_stats(true_haplotypes, unphased_haplotypes, phased_haplotypes):
    print('haplotype dimension')
    print(true_haplotypes.shape)
    headers = ['nuclear norm', 'rank', 'normalized frob distance']
    data = [['true', np.linalg.norm(true_haplotypes, 'nuc'), np.linalg.matrix_rank(true_haplotypes), np.linalg.norm(true_haplotypes - true_haplotypes)/np.linalg.norm(true_haplotypes)],
            ['unphased', np.linalg.norm(unphased_haplotypes, 'nuc'), np.linalg.matrix_rank(unphased_haplotypes), np.linalg.norm(unphased_haplotypes - true_haplotypes)/np.linalg.norm(true_haplotypes)],
            ['phased', np.linalg.norm(phased_haplotypes, 'nuc'), np.linalg.matrix_rank(phased_haplotypes), np.linalg.norm(phased_haplotypes - true_haplotypes)/np.linalg.norm(true_haplotypes)]]
    print(tabulate(data, headers=headers))

    num_phased = np.sum(np.logical_and(unphased_haplotypes == -1, phased_haplotypes != -1))
    print('Positions phased:')
    print(num_phased)

    print('switch error')
    print(switch_error(phased_haplotypes, true_haplotypes))
    print('percent switch error')
    num_phased = (np.sum(unphased_haplotypes == -1)) / 2
    print(switch_error(phased_haplotypes, true_haplotypes) / num_phased)

    # uncomment to print out comparable matrices
    # print('\n')
    # border = -8*np.ones((true_haplotypes.shape[0], 1))
    # print(np.hstack([true_haplotypes, border, phased_haplotypes]).astype(int))
