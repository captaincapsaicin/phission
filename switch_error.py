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
        for index in range(len(expected_i)):
            o_i = observed_i[index]
            e_i = expected_i[index]
            if observed[o_i, j] != expected[e_i, j]:
                # if we're wrong, we need to switch
                # but we give everyone one free pass
                if o_i in used_free_pass:
                    switch_errors += 1
                # we know the first time this occurs,
                # it will be with an even individual index
                used_free_pass.add(o_i)
                used_free_pass.add(o_i + 1)
                # follow along the other phased haplotype
                # switch between odd and even, e.g. 0 <-> 1, 24 <-> 25
                observed_i[index] = (o_i // 2) * 2 + 1 - o_i % 2
            # we also revoke your free pass if you've passed your first heterozygous site
            # again which will only occur while o_i is still even
            if observed[o_i, j] != observed[o_i + 1, j]:
                used_free_pass.add(o_i)
                used_free_pass.add(o_i + 1)

    return switch_errors
