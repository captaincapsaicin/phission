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
    expected_i = [i for i in range(observed.shape[0]) if i % 2 == 0]
    switch_errors = 0
    for j in range(observed.shape[1]):
        for index in range(len(expected_i)):
            o_i = observed_i[index]
            e_i = expected_i[index]
            if observed[o_i, j] != expected[e_i, j]:
                # we need to switch
                if j != 0:
                    # but we don't count it if it's only the first position.
                    # Just do the switch but don't penalize
                    switch_errors += 1
                # follow along the other phased haplotype
                # switch between odd and even, e.g. 0 <-> 1, 24 <-> 25
                observed_i[index] = (o_i // 2) * 2 + 1 - o_i % 2

    return switch_errors
