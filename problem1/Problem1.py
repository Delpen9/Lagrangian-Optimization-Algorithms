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
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    '''
    '''
    x = np.zeros(P.shape[0])
    z = np.zeros(P.shape[0])

    u = np.zeros(x.shape)
    w = u / _rho
    
    for iteration in range(max_iterations):
        print('Do something here.')
    return None

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


