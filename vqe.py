""" VQE class for variational quantum eigensolver.
Author : XXX (ykent@iastate.edu), Stanley Goodwin (sfg5318@rit.edu)
Date : 02/20/2024
Purpose : This file is used to run the VQE algorithm on the IBM Quantum
          Experience backend. This file is included in the repository.
"""

# Manage imports
import json
import numpy
import pickle
from qiskit_aer.primitives import Estimator as Estimator_aer
from qiskit_ibm_runtime import Session, Estimator
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector


def _load_inp(inp_file):
    var_global, var_local = {}, {}
    exec(open(inp_file, "r").read(), var_global, var_local)
    return var_local["inp"]


def _load_incar(incar_file, layers=1):
    INCAR = json.load(open(incar_file, "r"))
    INCAR["generators"] = INCAR["generators"] * layers
    return INCAR


def set_hamiltonian(incar):
    return SparsePauliOp.from_list([(x.split('*')[1], float(x.split('*')[0])) for x in incar["h"]])


class VariationalQuantumEigensolver:
    def __init__(self, inp_file="inp.py", incar_file="incar"):

        # Load the input file
        self._inp = _load_inp(inp_file)

        # Load the INCAR file
        self._incar = _load_incar(incar_file, layers=self._inp.get("nlayers", 1))

        # Set the Hamiltonian
        self._hamiltonian = set_hamiltonian(incar=self._incar)

        # Set the parameter circuit
        self._nq = self._hamiltonian.num_qubits
        assert (self._inp["nq"] == self._nq)
        self.set_param_circ()
        mode_aml = self._inp.get("mode_aml", 0)
        if mode_aml > 0:
            from source.aml import AML
            self._aml = AML(
                t=self._inp["tpar"],
                nranges=self._inp.get("nranges", None),
                mode=mode_aml)
            self.set_exact_points(err_exact=self._inp["err_exact"])
        else:
            self._aml = None

        self.set_exact_gs()
        self.check_init_state(self._inp.get("mode_init_check", 1))
        self._records = {}

    def get_x0_list(self):
        return self._inp["x0_list"]

    def set_param_circ(self):
        self._params = [Parameter(str(i))
                        for i in range(len(self._incar["generators"]))]
        circ = self._inp["circ_init"].copy()
        for param, oplabels in zip(self._params, self._incar["generators"]):
            for oplabel in oplabels:
                coeff, label = oplabel.split('*')
                op = SparsePauliOp.from_list([(label, float(coeff))])
                circ.append(PauliEvolutionGate(op, time=param), range(self._nq))
        self._param_circ = circ

    def save_param_circ(self):
        with open("source/pcirc.pkl", "wb") as f:
            pickle.dump(self._param_circ,
                        f,
                        pickle.HIGHEST_PROTOCOL,
                        )

    def save_records(self):
        with open("source/records.pkl", "wb") as f:
            pickle.dump(self._records,
                        f,
                        pickle.HIGHEST_PROTOCOL,
                        )

    def get_statevector_with_params(self, xlist):
        state = Statevector.from_int(0, 2 ** self._nq)
        binds = {th: val for th, val in zip(self._params, xlist)}
        circ = self._param_circ.bind_parameters(binds)
        state = state.evolve(circ)
        return state

    def get_fidelity_with_params(self, xlist):
        state = self.get_statevector_with_params(xlist)
        fid = abs(self._gs[1].dot(state._data)) ** 2
        return fid

    def check_init_state(self, mode):
        # check initial state
        state = self.get_statevector_with_params([0] * len(self._params))
        if mode > 0:
            # get h1
            # see https://scipost.org/SciPostPhys.6.3.029
            h1op = SparsePauliOp.from_list(
                [(x.split('*')[1], float(x.split('*')[0]))
                 for x in self._incar["generators"][-1]])
            h1mat = h1op.to_matrix()
            w, v = numpy.linalg.eigh(h1mat)
            expval = state.expectation_value(h1op)
            assert (abs(w[0] - expval) < 1e-6)
            assert (abs(abs(v[:, 0].dot(state._data)) - 1) < 1e-6)
        res = state.expectation_value(self._hamiltonian)
        print(f"initial state energy exact = {res.real:.6f}")

    def set_exact_gs(self):
        hmat = self._hamiltonian.to_matrix()
        w, v = numpy.linalg.eigh(hmat)
        print(f"exact gs energy: {w[0].real:.6f}")
        self._gs = [w[0], v[:, 0]]

    def get_energy_sv(self, xlist):
        estimator = Estimator_aer(
            backend_options={
                "method": "statevector",
            },
            run_options={"shots": None},
            approximation=True,
        )
        job = estimator.run([self._param_circ],
                            [self._hamiltonian],
                            xlist,
                            )
        return job.result().values[0]

    def get_energy(self, xlist):
        if self._inp["estimator"] is not None:
            job = self._inp["estimator"].run([self._param_circ],
                                             [self._hamiltonian],
                                             xlist,
                                             )
            res = job.result().values[0]
        else:
            keep_try = True
            while keep_try:
                try:
                    with Session(service=self._inp["service"], backend=self._inp["backend"]) as session:
                        estimator = Estimator(session=session, options=self._inp["options"])
                        job = estimator.run([self._param_circ],
                                            [self._hamiltonian],
                                            xlist,
                                            )
                        session.close()
                    res = job.result().values[0]
                    keep_try = False
                except:
                    pass

        return res

    def _fun_evaluate_energy(self):
        eval_count = 0

        def evaluate_energy(xlist):
            nonlocal eval_count
            eval_count += 1
            res = self.get_energy(xlist)
            print(f"fun eval {eval_count}: {res:.6f} at x: {numpy.round(xlist, decimals=3)}")
            return res

        return evaluate_energy


    # param_circ comes from a list of parameters from INCAR generators.
    # hamiltonian comes from a sparse pauli matrix from INCAR H list.
    # x_list is the initial Theta Angles
    def _fun_evaluate_energy_al(self):
        '''active-learning method.
        '''
        eval_count = 0
        if "evals" not in self._records:
            self._records["evals"] = []
            self._records["is_prediction"] = []

        def evaluate_energy(xlist):
            nonlocal eval_count
            eval_count += 1

            e_est, err_est = self._aml.prediction(xlist)
            print(f"predition {eval_count}: {e_est:.6f}, err_est:{err_est:.1e} at xs: {numpy.round(xlist, decimals=4)}")
            if err_est > self._inp["tol_prediction"]:
                e_calc = self.get_energy(xlist)
                self._aml.add_training_data([xlist], [e_calc], err=self._inp["err"])
                e_est2, err_est2 = self._aml.prediction(xlist)
                print(f"e_est: {e_est:.6f} +/- {err_est:.2e}, e_calc: {e_calc:.6f}," +
                      f" e_est2: {e_est2:.6f} +/- {err_est2:.2e} at x: {numpy.round(xlist, decimals=3)}")
                # use new prediction
                e_est = e_est2
                self._records["is_prediction"].append(0)
            else:
                self._records["is_prediction"].append(1)
            self._records["evals"].append(e_est)
            return e_est

        return evaluate_energy

    def minimize_energy(self):
        IS_AI = False
        if self._aml is None:
            cost_fun = self._fun_evaluate_energy()
        else:
            cost_fun = self._fun_evaluate_energy_al()
            IS_AI = True
        res = self._inp["optimizer"].minimize(cost_fun,
                                              self._inp["x0_list"],
                                              bounds=self._inp["bounds"],
                                              AI=IS_AI
                                              )
        self._res_opt = res
        e = self.get_energy(res.x)
        print(f"minimal energy finally measured: {e:.6f}")
        print(f"records: {self._records}")
        if "is_prediction" in self._records:
            print(f"non-prediction times: {self._records['is_prediction'].count(0)}")

    def estimate_t(self, nsample=100):
        x_list = numpy.random.rand(nsample, len(self._inp["x0_list"]))
        e_list = []
        for x in x_list:
            e = self.get_energy(x)
            e_list.append(e ** 2)
            print(e)
        t_est, t_std = numpy.mean(e_list), numpy.std(e_list)
        print(f"estimated t: {t_est:.2e} with std: {t_std:.2e}")

    def set_exact_points(self, err_exact=1e-8):
        xlists = []
        ylist = []
        for xlist in self._inp["xlists_exact"]:
            e = self.get_energy_sv(xlist)
            xlists.append(xlist)
            ylist.append(e.real)
        self._aml.add_training_data(xlists, ylist, err=err_exact)


def chk_fs_orthonormal(fs):
    import scipy.integrate as integrate
    symbols = sorted(fs[0].free_symbols, key=lambda s: s.name)
    assert (len(symbols) == 1)
    symbol = symbols[0]
    for i, fi in enumerate(fs):
        for j, fj in enumerate(fs[:i + 1]):
            fij = fi * fj
            # print(fij)
            result = integrate.quad(
                lambda x: float(fij.subs([[symbol, x]]).evalf()),
                -numpy.pi, numpy.pi)[0]
            # print(f"ovlp ({i}, {j}): {result}")
            if i == j:
                assert (abs(result - 1) < 1e-7)
            else:
                assert (abs(result) < 1e-7)
