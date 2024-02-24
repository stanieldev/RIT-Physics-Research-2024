from __future__ import annotations
from collections import deque
from collections.abc import Iterator
from typing import Callable
from time import time
import numpy as np
from qiskit_algorithms.optimizers.optimizer import OptimizerResult, POINT
from qiskit_algorithms.optimizers.spsa import SPSA, _validate_pert_and_learningrate, _make_spd, bernoulli_perturbation, _batch_evaluate
import logging

logger = logging.getLogger(__name__)




# TODO: LIST OF THINGS TO DO
# Compute Update Section 1: Has where the gradient is calculated using point estimate


# TODO: NOTES
# Checkpoint 1 is where the update is calculated using the gradient
# This doesn't need to be modified as far as I'm aware.




# Create a class called Gradient Descent that inherits from SPSA
class GradientDescent(SPSA):



    # This is the implementation in SPSAs
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
            diff /= 2 * eps**2

            rank_one = np.outer(delta1, delta2)
            hessian_sample = diff * (rank_one + rank_one.T) / 2

        return np.mean(values), gradient_sample, hessian_sample

    # This is the implementation in SPSA
    # This gives us an approximation of the point
    # 'loss' is the cost function of VQE
    # 'x' is the array of Theta Points of VQE
    # 'eps' is the perturbation from self.calibrate()
    # 'num_samples' is the number of samples to take and average to get an average slope at the point
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

# TODO WHERE SAMPLING IS DONE BEGINNING --------------------------------------------------------------------------------
            value_sample, gradient_sample, _ = self._point_sample(
                loss, x, eps, delta1, delta2
            )
# TODO WHERE SAMPLING IS DONE END --------------------------------------------------------------------------------------
            value_estimate += value_sample
            gradient_estimate += gradient_sample

            if self.second_order:
                hessian_estimate += _

        return (
            value_estimate / num_samples,
            gradient_estimate / num_samples,
            hessian_estimate / num_samples,
        )



    def _point_estimate_using_known_gradient(self, cost_fun, x, eps, num_samples, AI):

        # If AI, use point estimate, otherwise use the new algorithm
        if AI:
            print("AI")
            return self._point_estimate(cost_fun, x, eps, num_samples)
        if not AI:
            print("Not AI")
            return self._point_estimate(cost_fun, x, eps, num_samples)


        # # Initialize the variables to store the averages
        # value_estimate = 0
        # gradient_estimate = np.zeros(x.size)
        #
        # # Hessian is not used in VQE
        # hessian_estimate = None
        #
        # return (
        #     value_estimate,
        #     gradient_estimate,
        #     hessian_estimate,
        # )




    # Copied directly from SPSA class
    '''
    Loss :       The cost function of VQE
    x :          The array of Theta Points of VQE
    k :          Current Iteration Number
    eps:         Perturbation from self.calibrate()
    lse_solver:  np.linalg.solve (Linear Equation Solver)
    '''

    def _new_compute_update(self, loss, x, k, eps, lse_solver, AI):

        # compute the perturbations [ignore this doesn't change]
        if isinstance(self.resamplings, dict):
            num_samples = self.resamplings.get(k, 1)
        else:
            num_samples = self.resamplings

# TODO BEGIN SECTION 1 -------------------------------------------------------------------------------------------------
        # This is the thing that actually changes for this class
        # This gives us an approximation of the point,
        # 'loss' is the cost function of VQE
        # 'x' is the array of Theta Points of VQE
        # 'eps' is the perturbation from self.calibrate()
        # 'num_samples' is the number of samples to take and average to get an average slope at the point
        global USE_SPSA
        USE_SPSA = True
        if USE_SPSA:
            value, gradient, hessian = self._point_estimate(loss, x, eps, num_samples)
            USE_SPSA = False
        else:
            value, gradient, hessian = self._point_estimate_using_known_gradient(loss, x, eps, num_samples, AI)
# TODO END SECTION 1 ---------------------------------------------------------------------------------------------------

        # precondition gradient with inverse Hessian, if specified [ignore this isn't used in VQE]
        if self.second_order:
            smoothed = k / (k + 1) * self._smoothed_hessian + 1 / (k + 1) * hessian
            self._smoothed_hessian = smoothed

            if k > self.hessian_delay:
                spd_hessian = _make_spd(smoothed, self.regularization)

                # solve for the gradient update
                gradient = np.real(lse_solver(spd_hessian, gradient))

# TODO HERES WHERE IT RETURNS
        return value, gradient



    # Override the minimize method
    # Nothing except changing the logger name
    # The real changes are in the _compute_update method
    # This function is finished
    def minimize(
            self,
            fun: Callable[[POINT], float],
            x0: POINT,
            jac: Callable[[POINT], POINT] | None = None,
            bounds: list[tuple[float, float]] | None = None,
            AI: bool | None = None
    ) -> OptimizerResult:
        # ensure learning rate and perturbation are correctly set: either none or both
        # this happens only here because for the calibration the loss function is required
        if self.learning_rate is None and self.perturbation is None:
            get_eta, get_eps = self.calibrate(fun, x0, max_evals_grouped=self._max_evals_grouped)
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
            fx = fun(x)  # pylint: disable=invalid-name

            self._nfev += 1
            if self.allowed_increase is None:
                self.allowed_increase = 2 * self.estimate_stddev(
                    fun, x, max_evals_grouped=self._max_evals_grouped
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
# TODO: BEGINNING OF CHANGES -------------------------------------------------------------------------------------------
            # compute update
            # FX estimate is an estimate at the point
            # update is the gradient at the point
            fx_estimate, update = self._new_compute_update(fun, x, k, next(eps), lse_solver, AI)
# TODO: Checkpoint 1 ---------------------------------------------------------------------------------------------------
            # trust region
            if self.trust_region:
                norm = np.linalg.norm(update)
                if norm > 1:  # stop from dividing by 0
                    update = update / norm

            # compute next parameter value
            update = update * next(eta)
            x_next = x - update
            fx_next = None

            # blocking
            if self.blocking:
                self._nfev += 1
                fx_next = fun(x_next)

                if fx + self.allowed_increase <= fx_next:  # accept only if loss improved
                    if self.callback is not None:
                        self.callback(
                            self._nfev,  # number of function evals
                            x_next,  # next parameters
                            fx_next,  # loss at next parameters
                            np.linalg.norm(update),  # size of the update step
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

# TODO: END CHECKPOINT 1 -----------------------------------------------------------------------------------------------

            logger.info(
                "Iteration %s/%s done in %s.", k, self.maxiter + 1, time() - iteration_start
            )

            if self.callback is not None:
                # if we didn't evaluate the function yet, do it now
                if not self.blocking:
                    self._nfev += 1
                    fx_next = fun(x_next)

                self.callback(
                    self._nfev,  # number of function evals
                    x_next,  # next parameters
                    fx_next,  # loss at next parameters
                    np.linalg.norm(update),  # size of the update step
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
                fx_check = fx_estimate if fx_next is None else fx_next
                if self.termination_checker(
                        self._nfev, x_next, fx_check, np.linalg.norm(update), True
                ):
                    logger.info("terminated optimization at {k}/{self.maxiter} iterations")
                    break

        logger.info("GradientDescent: Finished in %s", time() - start)

        if self.last_avg > 1:
            x = np.mean(last_steps, axis=0)

        result = OptimizerResult()
        result.x = x
        result.fun = fun(x)
        result.nfev = self._nfev
        result.nit = k

        return result
