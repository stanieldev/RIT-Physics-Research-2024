import numpy as np
import itertools


# 2Pi-Periodic Kernel Function Heuristic [Verified]
def paper_equation_11(theta_1: np.ndarray, theta_2: np.ndarray, M: list, const: float = 1) -> float:

    # Assure all lists are the same length
    assert len(theta_1) == len(theta_2) == len(M)

    # Do a for loop to verify the answer
    result = 0
    for k in itertools.product(*[range(-m, m + 1) for m in M]):
        result += np.exp(1j * np.dot(theta_1 - theta_2, k))

    # # Take the dot product of each k vector with the theta difference using numpy's einsum
    # result = np.einsum("i,ki->k", theta_1 - theta_2, list(itertools.product(*[range(-m, m + 1) for m in M])), optimize=True)
    # result = np.exp(1j * result).sum()

    # Multiply the result by the constant and return the real part
    assert (np.all(abs(result.imag) < 1e-12))
    return const * result.real


# 2Pi-Periodic Kernel Function Heuristic Gradient
def paper_equation_11_gradient(theta_1: np.ndarray, theta_2: np.ndarray, M: list, const: float = 1) -> np.ndarray:

    # Assure all lists are the same length
    assert len(theta_1) == len(theta_2) == len(M)

    # Do with a loop since I don't know how to do it with numpy
    gradient_vector = np.zeros(len(M))
    for _ in range(len(M)):
        result = 0
        for k in itertools.product(*[range(-m, m + 1) for m in M]):
            result += 1j * k[_] * np.exp(1j * np.dot(theta_1 - theta_2, k))
        gradient_vector[_] = const * result.real

    # Return the gradient
    assert (np.all(abs(gradient_vector.imag) < 1e-12))
    return const * gradient_vector


# Pi-Periodic Kernel Function Heuristic
def paper_equation_12(theta_1: np.ndarray, theta_2: np.ndarray, M_half: list, const: float = 1) -> float:

    # Assure all lists are the same length
    assert len(theta_1) == len(theta_2) == len(M_half)

    # Do a for loop to verify the answer
    result = 0
    for k in itertools.product(*[range(-m, m + 1) for m in M_half]):
        result += np.exp(2j * np.dot(theta_1 - theta_2, k))

    # # Take the dot product of each k vector with the theta difference using numpy's einsum
    # result = np.einsum("i,ki->k", theta_1 - theta_2, list(itertools.product(*[range(-m, m + 1) for m in M])), optimize=True)
    # result = np.exp(1j * result).sum()

    # Multiply the result by the constant and return the real part
    assert (np.all(abs(result.imag) < 1e-12))
    return const * result.real


# Pi-Periodic Kernel Function Heuristic Gradient
def paper_equation_12_gradient(theta_1: np.ndarray, theta_2: np.ndarray, M_half: list, const: float = 1) -> np.ndarray:

    # Assure all lists are the same length
    assert len(theta_1) == len(theta_2) == len(M_half)

    # Do with a loop since I don't know how to do it with numpy
    gradient_vector = np.zeros(len(M_half))
    for _ in range(len(M_half)):
        result = 0
        for k in itertools.product(*[range(-m, m + 1) for m in M_half]):
            result += 2j * k[_] * np.exp(2j * np.dot(theta_1 - theta_2, k))
        gradient_vector[_] = const * result.real

    # Return the gradient
    assert (np.all(abs(gradient_vector.imag) < 1e-12))
    return const * gradient_vector


# Expected energy of a system
def paper_equation_22(theta: np.ndarray, theta_ai: np.ndarray, energy: np.ndarray, energy_error: np.ndarray, M: list) -> float:

    # Find K_bar matrix
    k_bar_matrix = paper_equation_11(theta_ai, theta_ai, M=M)
    k_bar_matrix += np.diag(energy_error)**2
    k_bar_inverse = np.linalg.inv(k_bar_matrix)

    # Calculate the energy estimate
    energy_estimate = np.einsum("a,ab,b", paper_equation_11(theta, theta_ai, M=M), k_bar_inverse, energy, optimize=True)
    return energy_estimate


# Variance energy of a system
def paper_equation_23(theta: np.ndarray, theta_ai: np.ndarray, energy_error: np.ndarray, M: list) -> float:

    # Find K_bar matrix
    k_bar_matrix = paper_equation_11(theta_ai, theta_ai, M=M)
    k_bar_matrix += np.diag(energy_error)**2
    k_bar_inverse = np.linalg.inv(k_bar_matrix)

    # Calculate the variance estimate
    k_11 = paper_equation_11(theta, theta, M=M)
    k_1a = paper_equation_11(theta, theta_ai, M=M)
    k_a1 = paper_equation_11(theta_ai, theta, M=M)
    variance = abs(k_11 - np.einsum("a,ab,b", k_1a, k_bar_inverse, k_a1, optimize=True))
    return variance


