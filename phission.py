import cvxpy as cvx
import numpy as np


def phission_phase(unphased):
    """
    Matrix completion with phasing
    """
    mask = get_mask(unphased)
    X = nuclear_norm_solve(unphased, mask)
    # round to the nearest integer
    return np.matrix.round(X).astype(int)


def nuclear_norm_solve(unphased, mask, mu=1):
    """
    Parameters:
    -----------
    unphased : m x n array
        matrix we want to complete
    mask : m x n array
        matrix with entries zero (if missing) or one (if present)
    mu : scalar
        constant tradeoff between rank and 4 gamete test default = 1

    Returns:
    --------
    X: m x n array
        completed matrix
    """
    mu_param = cvx.Parameter(nonneg=True, value=mu)

    X_columns = []
    rows, columns = unphased.shape
    for i in range(columns):
        X_columns.append(cvx.Variable(rows, 1))
    X = cvx.Variable(unphased.shape)

    objective = cvx.Minimize(cvx.norm(X, p='nuc') + mu_param*four_gamete_loss(X_columns))

    # equality constraints
    constraints = [X == cvx.vstack(X_columns).T]  # matrix building constraints. hstack doesn't work. vstack needed
    constraints += [cvx.multiply(mask, (X - unphased)) == np.zeros(unphased.shape)]  # equality of unphased positions
    constraints += get_symmetry_breaking_constraints(mask, X)
    constraints += get_sum_to_1_constraints(mask, X)

    problem = cvx.Problem(objective, constraints)
    problem.solve(solver='SCS')  # other solvers are failing all over the place
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


def four_gamete_loss(X_columns):
    num_columns = len(X_columns)
    all_losses = []
    for i in range(num_columns):
        for j in range(i + 1, num_columns):
            # TODO : figure out warning nonconvex
            all_losses.append(-cvx.log1p(X_columns[i].T*X_columns[j]))
            all_losses.append(-cvx.log1p((1 - X_columns[i]).T*X_columns[j]))
            all_losses.append(-cvx.log1p(X_columns[i].T*(1 - X_columns[j])))
            all_losses.append(-cvx.log1p((1 - X_columns[i]).T*(1 - X_columns[j])))
    return np.sum(all_losses)  # this should return a cvx expression


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
