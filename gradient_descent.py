
# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

This implementation allows both, standard first-order as well as second-order SPSA.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from typing import Callable, Any, SupportsFloat
import logging
import warnings
from time import time

import scipy
import numpy as np

from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers.spsa import SPSA


from qiskit_algorithms.optimizers.optimizer import Optimizer, OptimizerSupportLevel, OptimizerResult, POINT

# number of function evaluations, parameters, loss, stepsize, accepted
CALLBACK = Callable[[int, np.ndarray, float, SupportsFloat, bool], None]
TERMINATIONCHECKER = Callable[[int, np.ndarray, float, SupportsFloat, bool], bool]

logger = logging.getLogger(__name__)




class NewGradientDescent():
    def __init__(self, maxiter=100, epsilon=0.01):
        self.maxiter = maxiter
        self.epsilon = epsilon

    def minimize(self,
                 x0: np.ndarray,
                 cost_function: Callable[[np.ndarray], float],
                 grad_cost_fun: Callable[[np.ndarray], np.ndarray],
                 bounds: list[tuple[float, float]] | None = None,
                 lower_epsilon: float = 0.01,
                 upper_epsilon: float = 0.1,
                 is_log = False
     ) -> OptimizerResult:

        if is_log:
            K =np.log(upper_epsilon/lower_epsilon)/self.maxiter
            EPSILON = upper_epsilon * np.array([np.exp(-K*i) for i in range(self.maxiter)])
        else:
            K = (upper_epsilon - lower_epsilon) / self.maxiter
            EPSILON = upper_epsilon - K * np.array([i for i in range(self.maxiter)])

        x = x0
        for i in range(self.maxiter):
            energy = cost_function(x)
            energy_grad = grad_cost_fun(x)
            energy_grad = energy_grad / np.linalg.norm(energy_grad)
            x = x - EPSILON[i] * energy_grad

            # make sure they stay within bounds
            if bounds:
                for k, (low, high) in enumerate(bounds):
                    if x[k] < low:
                        x[k] = high
                    if x[k] > high:
                        x[k] = low

            print(f'Iteration {i}, x = {x}, cost = {cost_function(x)}')
        final_parameters = x
        final_energy = cost_function(x)
        print(f'Final parameters: {final_parameters}, Final energy: {final_energy}')
        output = OptimizerResult()
        output.x = final_parameters
        return output




# normalized_cost_function_gradient_vector = normalized_cost_function_gradient_vector * next(eta)
#             x_next = x - normalized_cost_function_gradient_vector







class OldGradientDescent(SPSA):

    @staticmethod
    def calibrate(
            loss: Callable[[np.ndarray], float],
            initial_point: np.ndarray,
            c: float = 0.2,
            stability_constant: float = 0,
            target_magnitude: float | None = None,  # 2 pi / 10
            alpha: float = 0.602,
            gamma: float = 0.101,
            modelspace: bool = False,
            max_evals_grouped: int = 1,
    ) -> tuple[Callable, Callable]:
        r"""Calibrate SPSA parameters with a power series as learning rate and perturbation coeffs.

        The power series are:

        .. math::

            a_k = \frac{a}{(A + k + 1)^\alpha}, c_k = \frac{c}{(k + 1)^\gamma}

        Args:
            loss: The loss function.
            initial_point: The initial guess of the iteration.
            c: The initial perturbation magnitude.
            stability_constant: The value of `A`.
            target_magnitude: The target magnitude for the first update step, defaults to
                :math:`2\pi / 10`.
            alpha: The exponent of the learning rate power series.
            gamma: The exponent of the perturbation power series.
            modelspace: Whether the target magnitude is the difference of parameter values
                or function values (= model space).
            max_evals_grouped: The number of grouped evaluations supported by the loss function.
                Defaults to 1, i.e. no grouping.

        Returns:
            tuple(generator, generator): A tuple of power series generators, the first one for the
                learning rate and the second one for the perturbation.
        """
        logger.info("SPSA: Starting calibration of learning rate and perturbation.")
        if target_magnitude is None:
            target_magnitude = 2 * np.pi / 10

        dim = len(initial_point)

        # compute the average magnitude of the first step
        steps = 25
        points = []
        for _ in range(steps):
            # compute the random direction
            pert = bernoulli_perturbation(dim)
            points += [initial_point + c * pert, initial_point - c * pert]

        losses = _batch_evaluate(loss, points, max_evals_grouped)

        avg_magnitudes = 0.0
        for i in range(steps):
            delta = losses[2 * i] - losses[2 * i + 1]
            avg_magnitudes += np.abs(delta / (2 * c))

        avg_magnitudes /= steps

        if modelspace:
            a = target_magnitude / (avg_magnitudes ** 2)
        else:
            a = target_magnitude / avg_magnitudes

        # compute the rescaling factor for correct first learning rate
        if a < 1e-10:
            warnings.warn(f"Calibration failed, using {target_magnitude} for `a`")
            a = target_magnitude

        logger.info("Finished calibration:")
        logger.info(
            " -- Learning rate: a / ((A + n) ^ alpha) with a = %s, A = %s, alpha = %s",
            a,
            stability_constant,
            alpha,
        )
        logger.info(" -- Perturbation: c / (n ^ gamma) with c = %s, gamma = %s", c, gamma)

        # set up the power series
        def learning_rate():
            return powerseries(a, alpha, stability_constant)

        def perturbation():
            return powerseries(c, gamma)

        return learning_rate, perturbation

    @staticmethod
    def estimate_stddev(
            loss: Callable[[np.ndarray], float],
            initial_point: np.ndarray,
            avg: int = 25,
            max_evals_grouped: int = 1,
    ) -> float:
        """Estimate the standard deviation of the loss function."""
        losses = _batch_evaluate(loss, avg * [initial_point], max_evals_grouped)
        return np.std(losses)

    @property
    def settings(self) -> dict[str, Any]:
        # if learning rate or perturbation are custom iterators expand them
        if callable(self.learning_rate):
            iterator = self.learning_rate()
            learning_rate = np.array([next(iterator) for _ in range(self.maxiter)])
        else:
            learning_rate = self.learning_rate

        if callable(self.perturbation):
            iterator = self.perturbation()
            perturbation = np.array([next(iterator) for _ in range(self.maxiter)])
        else:
            perturbation = self.perturbation

        return {
            "maxiter": self.maxiter,
            "learning_rate": learning_rate,
            "perturbation": perturbation,
            "trust_region": self.trust_region,
            "blocking": self.blocking,
            "allowed_increase": self.allowed_increase,
            "resamplings": self.resamplings,
            "perturbation_dims": self.perturbation_dims,
            "second_order": self.second_order,
            "hessian_delay": self.hessian_delay,
            "regularization": self.regularization,
            "lse_solver": self.lse_solver,
            "initial_hessian": self.initial_hessian,
            "callback": self.callback,
            "termination_checker": self.termination_checker,
        }

    def _point_sample(self, loss, x, eps, delta1, delta2):
        """A single sample of the gradient at position ``x`` in direction ``delta``."""
        # points to evaluate
        points = [x + eps * delta1, x - eps * delta1]
        self._nfev += 2

        if self.second_order:
            points += [x + eps * (delta1 + delta2), x + eps * (-delta1 + delta2)]
            self._nfev += 2

        # batch evaluate the points (if possible)
        values = _batch_evaluate(loss, points, self._max_evals_grouped)

        plus = values[0]
        minus = values[1]
        gradient_sample = (plus - minus) / (2 * eps) * delta1

        hessian_sample = None
        if self.second_order:
            diff = (values[2] - plus) - (values[3] - minus)
            diff /= 2 * eps ** 2

            rank_one = np.outer(delta1, delta2)
            hessian_sample = diff * (rank_one + rank_one.T) / 2

        return np.mean(values), gradient_sample, hessian_sample

    def _point_estimate(self, loss, x, eps, num_samples):
        """The gradient estimate at point x."""
        # set up variables to store averages
        value_estimate = 0
        gradient_estimate = np.zeros(x.size)
        hessian_estimate = np.zeros((x.size, x.size))

        # iterate over the directions
        deltas1 = [
            bernoulli_perturbation(x.size, self.perturbation_dims) for _ in range(num_samples)
        ]

        if self.second_order:
            deltas2 = [
                bernoulli_perturbation(x.size, self.perturbation_dims) for _ in range(num_samples)
            ]
        else:
            deltas2 = None

        for i in range(num_samples):
            delta1 = deltas1[i]
            delta2 = deltas2[i] if self.second_order else None

            value_sample, gradient_sample, hessian_sample = self._point_sample(
                loss, x, eps, delta1, delta2
            )
            value_estimate += value_sample
            gradient_estimate += gradient_sample

            if self.second_order:
                hessian_estimate += hessian_sample

        return (
            value_estimate / num_samples,
            gradient_estimate / num_samples,
            hessian_estimate / num_samples,
        )

    def _compute_update(self, loss, x, k, eps, lse_solver):
        # compute the perturbations
        if isinstance(self.resamplings, dict):
            num_samples = self.resamplings.get(k, 1)
        else:
            num_samples = self.resamplings

        # accumulate the number of samples
        value, gradient, hessian = self._point_estimate(loss, x, eps, num_samples)

        return value, gradient

    def minimize(
            self,
            cost_function: Callable[[POINT], float],
            x0: POINT,
            jac: Callable[[POINT], POINT] | None = None,
            bounds: list[tuple[float, float]] | None = None,
            grad_cost_fun: Callable[[POINT], POINT] | None = None,
    ) -> OptimizerResult:
        # ensure learning rate and perturbation are correctly set: either none or both
        # this happens only here because for the calibration the loss function is required
        if self.learning_rate is None and self.perturbation is None:
            get_eta, get_eps = self.calibrate(cost_function, x0, max_evals_grouped=self._max_evals_grouped)
        else:
            get_eta, get_eps = _validate_pert_and_learningrate(
                self.perturbation, self.learning_rate
            )
        eta, eps = get_eta(), get_eps()

        if self.lse_solver is None:
            lse_solver = np.linalg.solve
        else:
            lse_solver = self.lse_solver

        # prepare some initials
        x = np.asarray(x0)
        if self.initial_hessian is None:
            self._smoothed_hessian = np.identity(x.size)
        else:
            self._smoothed_hessian = self.initial_hessian

        self._nfev = 0

        # if blocking is enabled we need to keep track of the function values
        if self.blocking:
            fx = cost_function(x)  # pylint: disable=invalid-name

            self._nfev += 1
            if self.allowed_increase is None:
                self.allowed_increase = 2 * self.estimate_stddev(
                    cost_function, x, max_evals_grouped=self._max_evals_grouped
                )

        logger.info("GradientDescent: Starting optimization.")
        start = time()

        # keep track of the last few steps to return their average
        last_steps = deque([x])

        # use a local variable and while loop to keep track of the number of iterations
        # if the termination checker terminates early
        k = 0
        while k < self.maxiter:
            k += 1
            iteration_start = time()






            # compute update
            cost_function_estimate, cost_function_gradient_vector = self._compute_update(cost_function, x, k, next(eps), lse_solver)







            # trust region
            if self.trust_region:
                norm = np.linalg.norm(cost_function_gradient_vector)
                if norm > 1:  # stop from dividing by 0
                    normalized_cost_function_gradient_vector = cost_function_gradient_vector / norm
                else:
                    normalized_cost_function_gradient_vector = cost_function_gradient_vector
            else:
                normalized_cost_function_gradient_vector = cost_function_gradient_vector


            # compute next parameter value
            normalized_cost_function_gradient_vector = normalized_cost_function_gradient_vector * next(eta)
            x_next = x - normalized_cost_function_gradient_vector
            fx_next = None









            # blocking
            if self.blocking:
                self._nfev += 1
                fx_next = cost_function(x_next)

                if fx + self.allowed_increase <= fx_next:  # accept only if loss improved
                    if self.callback is not None:
                        self.callback(
                            self._nfev,  # number of function evals
                            x_next,  # next parameters
                            fx_next,  # loss at next parameters
                            np.linalg.norm(normalized_cost_function_gradient_vector),  # size of the update step
                            False,
                        )  # not accepted

                    logger.info(
                        "Iteration %s/%s rejected in %s.",
                        k,
                        self.maxiter + 1,
                        time() - iteration_start,
                    )
                    continue
                fx = fx_next  # pylint: disable=invalid-name

            logger.info(
                "Iteration %s/%s done in %s.", k, self.maxiter + 1, time() - iteration_start
            )

            if self.callback is not None:
                # if we didn't evaluate the function yet, do it now
                if not self.blocking:
                    self._nfev += 1
                    fx_next = cost_function(x_next)

                self.callback(
                    self._nfev,  # number of function evals
                    x_next,  # next parameters
                    fx_next,  # loss at next parameters
                    np.linalg.norm(normalized_cost_function_gradient_vector),  # size of the update step
                    True,
                )  # accepted

            # update parameters
            x = x_next

            # update the list of the last ``last_avg`` parameters
            if self.last_avg > 1:
                last_steps.append(x_next)
                if len(last_steps) > self.last_avg:
                    last_steps.popleft()

            if self.termination_checker is not None:
                fx_check = cost_function_estimate if fx_next is None else fx_next
                if self.termination_checker(
                        self._nfev, x_next, fx_check, np.linalg.norm(normalized_cost_function_gradient_vector), True
                ):
                    logger.info("terminated optimization at {k}/{self.maxiter} iterations")
                    break

        logger.info("GradientDescent: Finished in %s", time() - start)

        if self.last_avg > 1:
            x = np.mean(last_steps, axis=0)

        result = OptimizerResult()
        result.x = x
        result.fun = cost_function(x)
        result.nfev = self._nfev
        result.nit = k

        return result

    def get_support_level(self):
        """Get the support level dictionary."""
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.required,
        }

def bernoulli_perturbation(dim, perturbation_dims=None):
    """Get a Bernoulli random perturbation."""
    if perturbation_dims is None:
        return 1 - 2 * algorithm_globals.random.binomial(1, 0.5, size=dim)

    pert = 1 - 2 * algorithm_globals.random.binomial(1, 0.5, size=perturbation_dims)
    indices = algorithm_globals.random.choice(
        list(range(dim)), size=perturbation_dims, replace=False
    )
    result = np.zeros(dim)
    result[indices] = pert

    return result

def powerseries(eta=0.01, power=2, offset=0):
    """Yield a series decreasing by a power law."""

    n = 1
    while True:
        yield eta / ((n + offset) ** power)
        n += 1

def constant(eta=0.01):
    """Yield a constant series."""

    while True:
        yield eta

def _batch_evaluate(function, points, max_evals_grouped, unpack_points=False):
    """Evaluate a function on all points with batches of max_evals_grouped.

    The points are a list of inputs, as ``[in1, in2, in3, ...]``. If the individual
    inputs are tuples (because the function takes multiple inputs), set ``unpack_points`` to ``True``.
    """

    # if the function cannot handle lists of points as input, cover this case immediately
    if max_evals_grouped is None or max_evals_grouped == 1:
        # support functions with multiple arguments where the points are given in a tuple
        return [
            function(*point) if isinstance(point, tuple) else function(point) for point in points
        ]

    num_points = len(points)

    # get the number of batches
    num_batches = num_points // max_evals_grouped
    if num_points % max_evals_grouped != 0:
        num_batches += 1

    # split the points
    batched_points = np.array_split(np.asarray(points), num_batches)

    results = []
    for batch in batched_points:
        if unpack_points:
            batch = _repack_points(batch)
            results += _as_list(function(*batch))
        else:
            results += _as_list(function(batch))

    return results

def _as_list(obj):
    """Convert a list or numpy array into a list."""
    return obj.tolist() if isinstance(obj, np.ndarray) else obj

def _repack_points(points):
    """Turn a list of tuples of points into a tuple of lists of points.
    E.g. turns
        [(a1, a2, a3), (b1, b2, b3)]
    into
        ([a1, b1], [a2, b2], [a3, b3])
    where all elements are np.ndarray.
    """
    num_sets = len(points[0])  # length of (a1, a2, a3)
    return ([x[i] for x in points] for i in range(num_sets))

def _make_spd(matrix, bias=0.01):
    identity = np.identity(matrix.shape[0])
    psd = scipy.linalg.sqrtm(matrix.dot(matrix))
    return psd + bias * identity

def _validate_pert_and_learningrate(perturbation, learning_rate):
    if learning_rate is None or perturbation is None:
        raise ValueError("If one of learning rate or perturbation is set, both must be set.")

    if isinstance(perturbation, float):

        def get_eps():
            return constant(perturbation)

    elif isinstance(perturbation, (list, np.ndarray)):

        def get_eps():
            return iter(perturbation)

    else:
        get_eps = perturbation

    if isinstance(learning_rate, float):

        def get_eta():
            return constant(learning_rate)

    elif isinstance(learning_rate, (list, np.ndarray)):

        def get_eta():
            return iter(learning_rate)

    else:
        get_eta = learning_rate

    return get_eta, get_eps
