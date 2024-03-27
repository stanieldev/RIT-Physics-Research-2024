import itertools
import numpy
import heuristics


# Control Panel
KERNAL_DEBUG = False


# Agnostic kernel Eq.(21)
def agnostic_kernel_46(theta_ai, theta_bi, const=1./117):

    USE_OLD = False
    if USE_OLD:
        # The vector k sums are made in this list
        vector_k_sum_indices = [[h1, h2] for h1, h2 in itertools.product(range(-4, 5), range(-6, 7))]
        assert(len(theta_ai[0]) == len(theta_bi[0]) == 2)

        # Calculate the difference between the theta vectors
        theta_abi = numpy.asarray([[numpy.asarray(vi1) - numpy.asarray(vi2) for vi2 in theta_bi] for vi1 in theta_ai])

        # Calculate the exponent and sum over the k indices
        thetah_abk = numpy.einsum("abi,ki->abk", theta_abi, vector_k_sum_indices, optimize=True)*2j

        # Sum all the exponentials and multiply by the constant
        res = numpy.exp(thetah_abk).sum(axis=2)*const
        assert(numpy.all(abs(res.imag) < 1e-12))

        # Return the real part of the result
        return res.real
    else:
        heuristic_matrix = numpy.zeros((len(theta_ai), len(theta_bi)))
        for a, b in itertools.product(range(len(theta_ai)), range(len(theta_bi))):
            heuristic_matrix[a, b] = heuristics.paper_equation_11(theta_ai[a], theta_bi[b], [4, 6], const)
        return heuristic_matrix


def agnostic_kernel_46_gradient(theta_ai, theta_bi, const=None):

    # Print that this function was called
    if KERNAL_DEBUG:
        print(f"[KERNAL] Called agnostic_kernel_46_gradient(\n\t{theta_ai=}, \n\t{theta_bi=}, \n\t{const=}\n)")

    # The vector k sums are made in this list
    vector_k_sum_indices = [[h1, h2] for h1, h2 in itertools.product(range(-4, 5), range(-6, 7))]

    # Block 1 for Gradient 1
    res_1 = 0
    for i in range(len(theta_ai)):
        for j in range(len(theta_bi)):
            for k in range(len(vector_k_sum_indices)):
                factor = vector_k_sum_indices[k][0]
                res_1 += 2j * factor * numpy.exp(
                    2j * numpy.dot(numpy.array(theta_ai[i]) - numpy.array(theta_bi[j]), vector_k_sum_indices[k]))
    assert (numpy.all(abs(res_1.imag) < 1e-12))

    # Block 2 for Gradient 2
    res_2 = 0
    for i in range(len(theta_ai)):
        for j in range(len(theta_bi)):
            for k in range(len(vector_k_sum_indices)):
                factor = vector_k_sum_indices[k][1]
                res_2 += 2j * factor * numpy.exp(
                    2j * numpy.dot(numpy.array(theta_ai[i]) - numpy.array(theta_bi[j]), vector_k_sum_indices[k]))
    assert (numpy.all(abs(res_2.imag) < 1e-12))

    # Return gradient
    norm = numpy.sqrt(res_1.real ** 2 + res_2.real ** 2)
    if norm == 0:
        theta = numpy.random.rand() * 2 * numpy.pi
        gradient_vector = [numpy.sin(theta), numpy.cos(theta)]
        if KERNAL_DEBUG:
            print("[KERNAL] Warning: Gradient is zero, returning random vector")
        return numpy.array(gradient_vector)
    else:
        gradient_vector = [res_1.real / norm, res_2.real / norm]
        if KERNAL_DEBUG:
            print(f"[KERNAL] Returning gradient: {gradient_vector}")
        return numpy.array(gradient_vector)


def agnostic_kernel_n1n2(theta_ai, theta_bi, n1, n2, const):
    h_ki = [[h1, h2]
            for h1, h2 in itertools.product(range(-n1, n1+1), range(-n2, n2+1))]
    assert(len(theta_ai[0]) == len(theta_bi[0]) == 2)
    theta_abi = numpy.asarray([[numpy.asarray(vi1) - numpy.asarray(vi2)
            for vi2 in theta_bi]
            for vi1 in theta_ai])
    thetah_abk = numpy.einsum("abi,ki->abk", theta_abi, h_ki, optimize=True)*1j
    res = numpy.exp(thetah_abk).sum(axis=2)*const
    assert(numpy.all(abs(res.imag) < 1e-12))
    return res.real


def stochastic_kernel_direct(theta_ai, theta_bi, nlist, const=1):
    theta_abi = numpy.asarray([[numpy.asarray(vi1) - numpy.asarray(vi2)
            for vi2 in theta_bi]
            for vi1 in theta_ai])/2
    theta_abi_cos = numpy.cos(theta_abi)
    theta_abi_sin = numpy.sin(theta_abi)
    kab = numpy.ones(theta_abi_cos.shape[:2])
    for i, n in enumerate(nlist):
        kab *= (theta_abi_cos[:, :, i]**(2*n) + theta_abi_sin[:, :, i]**(2*n))
    kab *= const
    return kab


# Test
if __name__ == "__main__":
    print(agnostic_kernel_46([[0, 0]], [[0, 0]]))
    print(agnostic_kernel_n1n2([[0, 0]], [[0, 0]], 9, 12, 1.0))
    print(agnostic_kernel_n1n2([[0, 0]], [[0, 0]], 2, 2, 1.0))
    print(stochastic_kernel_direct([[0, 0]], [[0, 0]], [9, 12], 1.0))
