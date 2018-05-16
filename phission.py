import cvxpy as cvx
import numpy as np

import time


def phission_phase(unphased):
    """
    Matrix completion with phasing
    """
    mask = get_mask(unphased)
    X = nuclear_norm_solve(unphased, mask)
    # round to the nearest integer
    return np.matrix.round(X).astype(int)


def nuclear_norm_solve(unphased, mask):
    """
    Parameters:
    -----------
    unphased : m x n array
        matrix we want to complete
    mask : m x n array
        matrix with entries zero (if missing) or one (if present)

    Returns:
    --------
    X: m x n array
        completed matrix
    """
    start_time = time.time()
    X = cvx.Variable(*unphased.shape)
    objective = cvx.Minimize(cvx.norm(X, "nuc"))
    # equality constraints
    # print('equality constraints')
    constraints = [cvx.mul_elemwise(mask, X - unphased) == np.zeros(unphased.shape)]
    # print(time.time() - start_time)
    # print('# of equality constraints: {}'.format(np.count_nonzero(mask)))
    # print('sum to 1 constraints')
    constraints += get_sum_to_1_constraints(mask, X)
    # print(time.time() - start_time)
    # print('# of sum_to_1 constraints: {}'.format(len(get_sum_to_1_constraints(mask, X))))
    # print('symmetry breaking constraints')
    constraints += get_symmetry_breaking_constraints(mask, X)
    # print(time.time() - start_time)
    # print('# of symmetry_breaking constraints: {}'.format(len(get_symmetry_breaking_constraints(mask, X))))

    problem = cvx.Problem(objective, constraints)

    problem.solve(solver='SCS', verbose=False)
    # print('solve')
    # print(time.time() - start_time)
    return X.value


def get_symmetry_breaking_constraints(mask, X):
    """
    mask : m x n array
        matrix with entries zero (if missing) or one (if present)
    X  :
        cvxpy variable that we're solving for.

    We want the first set of indexes for every 0/0 mask pair for each individual
    """
    constraints = []
    indexes = get_unmasked_even_indexes(mask)
    seen_individuals = set()
    for i, j in indexes:
        if i not in seen_individuals:
            constraints.append(X[i, j] == 1)
            constraints.append(X[i + 1, j] == 0)
            seen_individuals.add(i)
    return constraints


def get_sum_to_1_constraints(mask, X):
    """
    A is our starting matrix, it has 0s in the spot we need to phase
    X is our cvxpy variable that we're solving for.

    We need each pair of phased haplotypes to sum to 0 (i.e. one is -1 and the other is 1)
    """
    constraints = []
    indexes = get_unmasked_even_indexes(mask)
    for i, j in indexes:
        constraints.append((X[i, j] + X[i + 1, j]) == 1)
    return constraints


def get_mask(A):
    """
    Gets a mask indicating non-homozygous elements from haplotype matrix A
    """
    return A != -1


def get_unmasked_even_indexes(mask):
    """
    mask : m x n array
        matrix with entries zero (if missing) or one (if present)

    For use in setting up constraints.
    """
    i, j = np.nonzero(1 - mask)
    new_i = []
    new_j = []
    # we're going to filter out all of the odds, since we know about those already
    # and maybe it'll be confusing later on? I'm not sure. I guess we could just remove them later.
    # but I'd rather have a set where I can be like (i, j) + (i+1, j) == 1. basically.
    for index in range(len(i)):
        if i[index] % 2 == 0:
            new_i.append(i[index])
            new_j.append(j[index])

    return zip(new_i, new_j)
