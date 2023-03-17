# Standard libraries
import os
import numpy as np

# Loading .mat files
import scipy.io

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

def admm_algorithm(
    _lambda : float,
    _rho : float,
    a : np.ndarray,
    b : np.ndarray,
    q : np.ndarray,
    P : np.ndarray,
    max_iterations : int = 100,
    _epsilon : float = 1e-10
) -> tuple[float]:
    '''
    Implements the Alternating Direction Method of Multipliers (ADMM) algorithm to solve the convex optimization problem:
    
        minimize_x 1/2 * x.T * P * x + q.T * x + lambda / 2 * ||z||^{2}_{2}
        subject to a <= z <= b
        
    where P is a positive semi-definite matrix, q is a vector, and a and b are lower and upper bounds on the decision variable x.
    
    Parameters:
    -----------
    _lambda : float
        A scalar parameter controlling the strength of the constraint violation penalty term.
    _rho : float
        A scalar parameter controlling the step size in the augmented Lagrangian method.
    a : np.ndarray
        A (n x 1) numpy array specifying the lower bounds on the decision variable x.
    b : np.ndarray
        A (n x 1) numpy array specifying the upper bounds on the decision variable x.
    q : np.ndarray
        A (n x 1) numpy array specifying the linear objective term in the optimization problem.
    P : np.ndarray
        A (n x n) numpy array specifying the quadratic objective term in the optimization problem.
    max_iterations : int, optional
        Maximum number of iterations for the algorithm. Default is 100.
    _epsilon : float, optional
        A scalar parameter controlling the stopping condition of the algorithm. The algorithm stops if the relative 
        difference between the previous and current values of x is less than epsilon. Default is 1e-10.
        
    Returns:
    --------
    A tuple containing the solution to the optimization problem in the form of a (n x 1) numpy array.
    '''
    x = np.zeros((P.shape[0], 1))
    z = np.zeros((P.shape[0], 1))

    u = np.zeros((P.shape[0], 1))
    w = u / _rho

    for iteration in range(max_iterations):
        _inv = np.linalg.pinv(
                P + P.T + 2 * _rho * np.eye(P.shape[0])
        )
        x = -np.dot((q + _rho * (-z + w)).T, (2 * _inv)).T
        z = -(_rho * (x + w))/(_lambda - _rho)
        
        z[z > b] = a[z > b]
        z[z < a] = a[z < a]
        
        w += x - z

        stopping_condition = np.linalg.norm(x - z)
        if stopping_condition < _epsilon:
            break

    return (x)

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    file_directory = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Question1.mat'))
    mat_file_data = scipy.io.loadmat(file_directory)

    a = mat_file_data['a']
    b = mat_file_data['b']
    q = mat_file_data['q']
    P = mat_file_data['P']
    
    _lambda = 0.5
    _rho = 1.1

    x = admm_algorithm(_lambda, _rho, a, b, q, P)

    # print(x)