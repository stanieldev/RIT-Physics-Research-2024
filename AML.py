# developed in c2qa (ykent@iastate.edu).
import numpy
from kernels import *


# Kernal Enumeration
AGNOSTIC_KERNAL_46 = 1
AGNOSTIC_KERNAL_N1N2 = 2
STOCHASTIC_KERNAL_DIRECT = 3

# Control Panel
DEBUG = False



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

            def kernel_gradient(theta_ai, theta_bi):
                return agnostic_kernel_46_gradient(theta_ai, theta_bi, self._t_parameter)
            return kernel, kernel_gradient
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
            if DEBUG:
                print(f"[AML] Added Training Data: {ene_est:.2e} ± {err_est:.2e} ({ene_est-energy:.2e})")

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
        k_gradient = self.kernel_gradient([theta_i], self._theta_ai)

        if len(self._theta_ai) == 0:
            ene_est_1 = 1e2
            ene_est_2 = 1e2
        else:
            k_1a_1 = [k_gradient[0]]
            k_1a_2 = [k_gradient[1]]
            ene_est_1 = numpy.einsum("a,ab,b", k_1a_1, self._kbinv_ab, self._energy_a,
                    optimize=True)
            ene_est_2 = numpy.einsum("a,ab,b", k_1a_2, self._kbinv_ab, self._energy_a,
                    optimize=True)

        # Return gradient
        norm = numpy.sqrt(ene_est_1.real ** 2 + ene_est_2.real ** 2)
        if norm == 0:
            theta = numpy.random.rand() * 2 * numpy.pi
            gradient_vector = [numpy.sin(theta)/1e6, numpy.cos(theta)/1e6]
            if DEBUG:
                print("[AML] Warning: Gradient is zero, returning random vector")
            return numpy.array(gradient_vector)
        else:
            gradient_vector = [ene_est_1.real, ene_est_2.real]
            if DEBUG:
                print(f"[AML] Returning gradient: {gradient_vector}")
            return numpy.array(gradient_vector)



# Main guard
if __name__ == "__main__":
    aml = AML()
