# developed in c2qa (ykent@iastate.edu).
import numpy


class AML:
    def __init__(self, t=2, mode=1, nranges=None):
        self._t = t
        self._nranges = nranges
        self._mode = mode
        self.set_kernel(mode)
        self.init_lists()

    def init_lists(self):
        self._theta_ai = []
        self._energy_a = []
        self._err_a = []

    def set_kernel(self, mode):
        if mode == 1:
            from kernels import agnostic_kernel_46
            def kernel(theta_ai, theta_bi):
                return agnostic_kernel_46(theta_ai, theta_bi, self._t)
        elif mode == 2:
            from kernels import agnostic_kernel_n1n2
            def kernel(theta_ai, theta_bi):
                return agnostic_kernel_n1n2(
                        theta_ai, theta_bi,
                        self._nranges[0], self._nranges[1],
                        self._t
                        )
        elif mode == 3:
            from kernels import stochastic_kernel_direct
            def kernel(theta_ai, theta_bi):
                return stochastic_kernel_direct(
                        theta_ai, theta_bi,
                        self._nranges,
                        )
        else:
            raise ValueError(f"mode = {mode} not implemented.")
        self.kernel = kernel

    def add_training_data(self, theta_ai, energy_a, err=1e-6):
        '''initialize with known accurate points.
        '''
        if len(theta_ai) == 0:
            return

        self._theta_ai += theta_ai
        self._energy_a += energy_a
        self._err_a += [err]*len(energy_a)
        k_ab = self.kernel(self._theta_ai, self._theta_ai)
        # should use variance
        k_ab += numpy.diag(self._err_a)**2
        # assume one doesn't need regulaization due to the above term
        # which assumes self._errs > 1e-6.
        self._kbinv_ab = numpy.linalg.inv(k_ab)

        ### sanity check
        for theta_i, energy in zip(theta_ai, energy_a):
            ene_est, err_est = self.prediction(theta_i)
            print(f"prediction: {ene_est:.2e}, error: {ene_est-energy:.2e}, err_est: {err_est:.2e}")

    def prediction(self, theta_i):
        if len(self._theta_ai) == 0:
            ene_est = 1e2
            err_est = 1e2
        else:
            k_11 = self.kernel([theta_i], [theta_i])[0, 0]
            k_1a = self.kernel([theta_i], self._theta_ai)[0, :]
            k_a1 = self.kernel(self._theta_ai, [theta_i])[:, 0]
            ene_est = numpy.einsum("a,ab,b", k_1a, self._kbinv_ab, self._energy_a,
                    optimize=True)
            var = k_11 - numpy.einsum("a,ab,b", k_1a, self._kbinv_ab, k_a1,
                    optimize=True)
            err_est = numpy.sqrt(var)
        return ene_est, err_est



if __name__ == "__main__":
    aml = AML()
