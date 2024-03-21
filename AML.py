# developed in c2qa (ykent@iastate.edu).
import numpy
from kernels import *


# Kernal Enumeration
AGNOSTIC_KERNAL_46 = 1
AGNOSTIC_KERNAL_N1N2 = 2
STOCHASTIC_KERNAL_DIRECT = 3


class AML:
    '''Agnostic Machine Learning (AML) model.'''
    def __init__(self, t=2, mode=1, n_ranges=None):

        # Initialize the AML model.
        self._t_parameter = t
        self._mode = mode
        self._n_ranges = n_ranges

        # Select the kernel function.
        self.kernel, self.kernel_gradient = self._select_kernel(mode)

        # Initialize the training data lists.
        self._theta_ai = []
        self._energy_a = []
        self._err_a = []

    # Set the kernel function.
    def _select_kernel(self, mode):
        if mode == AGNOSTIC_KERNAL_46:
            def kernel(theta_ai, theta_bi):
                return agnostic_kernel_46(theta_ai, theta_bi, self._t_parameter)
            return kernel, None
        elif mode == AGNOSTIC_KERNAL_N1N2:
            def kernel(theta_ai, theta_bi):
                return agnostic_kernel_n1n2(theta_ai, theta_bi, self._n_ranges[0], self._n_ranges[1], self._t_parameter)
            return kernel, None
        elif mode == STOCHASTIC_KERNAL_DIRECT:
            def kernel(theta_ai, theta_bi):
                return stochastic_kernel_direct(theta_ai, theta_bi, self._n_ranges)
            return kernel, None
        else:
            raise ValueError("Unknown Mode")

    # Add training data function.
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
            print(f"[AML] Prediction: {ene_est:.2e}, error: {ene_est-energy:.2e}, err_est: {err_est:.2e}")

    # Make an educated prediction function.
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
            var = abs(k_11 - numpy.einsum("a,ab,b", k_1a, self._kbinv_ab, k_a1,
                    optimize=True))
            err_est = numpy.sqrt(var)
        return ene_est, err_est

    # Make an educated prediction function for the gradient.
    def predict_gradient(self, theta_i):
        if len(self._theta_ai) == 0:
            grad_est = numpy.zeros(len(theta_i))
        else:
            k_11 = self.kernel([theta_i], [theta_i])[0, 0]
            k_1a = self.kernel_gradient([theta_i], self._theta_ai)[0, :]
            k_a1 = self.kernel(self._theta_ai, [theta_i])[:, 0]
            grad_est = numpy.einsum("a,ab,b", k_1a, self._kbinv_ab, self._energy_a,
                    optimize=True)
        return grad_est


# Main guard
if __name__ == "__main__":
    aml = AML()
