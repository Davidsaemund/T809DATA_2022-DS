import numpy as np
import matplotlib.pyplot as plt


def normal(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    # Part 1.1
    delta = np.finfo(np.float).eps
    return np.exp(-np.power(x-mu, 2)/(2*np.power(sigma+delta, 2)))/\
        np.sqrt(2*np.pi*np.power(sigma+delta, 2))
    
def plot_normal(sigma: float, mu:float, x_start: float, x_end: float):
    # Part 1.2
    return 0
def _plot_three_normals():
    # Part 1.2
    return 0
def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    # Part 2.1
    return 0
def _compare_components_and_mixture():
    # Part 2.2
    return 0
def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    # Part 3.1
    return 0
def _plot_mixture_and_samples():
    # Part 3.2
    return 0
if __name__ == '__main__':
    # select your function to test here and do `python3 template.py`
    p_normal = normal(np.array([-1,0,1]),1,0)
    print(p_normal)