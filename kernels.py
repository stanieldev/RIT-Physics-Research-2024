import itertools, numpy



# Agnostic kernel Eq.(21)
def EQUATION_21(theta_ai, theta_bi, const=1. / 117):
    h_ki = [[h1, h2]
            for h1, h2 in itertools.product(range(-4, 5), range(-6, 7))]
    assert(len(theta_ai[0]) == len(theta_bi[0]) == 2)
    theta_abi = numpy.asarray([[numpy.asarray(vi1) - numpy.asarray(vi2)
            for vi2 in theta_bi]
            for vi1 in theta_ai])
    thetah_abk = numpy.einsum("abi,ki->abk", theta_abi, h_ki,
            optimize=True)*2j
    res = numpy.exp(thetah_abk).sum(axis=2)*const
    assert(numpy.all(abs(res.imag) < 1e-12))
    return res.real


def EQUATION_11_12(theta_ai, theta_bi, n1, n2, const):
    h_ki = [[h1, h2]
            for h1, h2 in itertools.product(range(-n1, n1+1), range(-n2, n2+1))]

    assert(len(theta_ai[0]) == len(theta_bi[0]) == 2)
    theta_abi = numpy.asarray([[numpy.asarray(vi1) - numpy.asarray(vi2)
            for vi2 in theta_bi]
            for vi1 in theta_ai])
    thetah_abk = numpy.einsum("abi,ki->abk", theta_abi, h_ki,
            optimize=True)*1j
    res = numpy.exp(thetah_abk).sum(axis=2)*const
    assert(numpy.all(abs(res.imag) < 1e-12))
    return res.real




def EQUATION_11_12_GRADIENT(theta_ai, theta_bi, n1, n2, const):
    x_list = range(-n1, n1 + 1)
    h_ki = [[h1, h2] for h1, h2 in itertools.product(x_list, range(-n2, n2 + 1))]

    assert (len(theta_ai[0]) == len(theta_bi[0]) == 2)
    theta_abi = numpy.asarray([[numpy.asarray(vi1) - numpy.asarray(vi2)
                                for vi2 in theta_bi]
                               for vi1 in theta_ai])
    thetah_abk = numpy.einsum("abi, a, ki->abk", theta_abi, x_list, h_ki,
                              optimize=True) * 1j
    res = numpy.exp(thetah_abk).sum(axis=2) * const
    assert (numpy.all(abs(res.imag) < 1e-12))
    theta_1_gradient = res.real

    assert (len(theta_ai[0]) == len(theta_bi[0]) == 2)
    theta_abi = numpy.asarray([[numpy.asarray(vi1) - numpy.asarray(vi2)
                                for vi2 in theta_bi]
                               for vi1 in theta_ai])
    thetah_abk = numpy.einsum("abi, b, ki->abk", theta_abi, x_list, h_ki,
                              optimize=True) * 1j
    res = numpy.exp(thetah_abk).sum(axis=2) * const
    assert (numpy.all(abs(res.imag) < 1e-12))
    theta_2_gradient = res.real

    return theta_1_gradient, theta_2_gradient




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


if __name__ == "__main__":
    print(EQUATION_21([[0, 0]], [[0, 0]]))
    print(EQUATION_11_12([[0, 0]], [[0, 0]], 9, 12, 1.0))
    print(EQUATION_11_12([[0, 0]], [[0, 0]], 2, 2, 1.0))
    print(stochastic_kernel_direct([[0, 0]], [[0, 0]], [9, 12], 1.0))
