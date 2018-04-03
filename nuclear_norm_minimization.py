from cvxpy import Minimize, Problem, Variable, SCS, mul_elemwise, norm, sum_squares
import numpy as np


# a random 100x100 matrix of -1, 1
A = 2*np.random.random_integers(0, 1, (100, 100)) - 1

m = [[-1, 1],
     [0, 1]]
m = np.array(m)
mask = [[1, 1],
        [0, 1]]
mask = np.array(mask)


def nuclear_norm_solve(A, mask, mu):
    """
    Taken from https://github.com/tonyduan/matrix-completion

    Solve using a nuclear norm approach, using CVXPY.
    [ Candes and Recht, 2009 ]
    Parameters:
    -----------
    A : m x n array
        matrix we want to complete
    mask : m x n array
        matrix with entries zero (if missing) or one (if present)
    mu : float
        hyper-parameter controlling trade-off between nuclear norm and square loss
        Returns:
    --------
    X: m x n array
    completed matrix
    """
    X = Variable(*A.shape)
    objective = Minimize(norm(X, "nuc") +
                         mu * sum_squares(mul_elemwise(mask, X - A)))
    problem = Problem(objective, [])
    problem.solve(solver=SCS)
    return X.value
