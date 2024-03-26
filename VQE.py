# developed in c2qa (ykent@iastate.edu).
from __future__ import annotations
import json
import numpy
import pickle
from qiskit_aer.primitives import Estimator as EstimatorAER
from qiskit_ibm_runtime import Estimator
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms.optimizers import GradientDescent as GDQiskit
from qiskit.circuit import QuantumCircuit
import os
from AML import AML

# Control Panel
DEBUG = True
OLD_OPTIMIZER = False

class VQECircuit:
    def __init__(self, circuit, parameters):
        self.circuit = circuit
        self.parameters = parameters


class Eigenstate:
    def __init__(self, energy, state):
        self.energy = energy
        self.state = state


class FakeResult:
    def __init__(self, x):
        self.x = x


# Function to load the input file (inp.py)
def _load_input_file(filename: str) -> dict:

    # Test input file
    if DEBUG:
        print(f"[VQE_INIT] Called _load_input_file({filename=})")
    assert filename.endswith(".py")
    assert os.path.isfile(filename)

    # Load the input file (.py) and execute it
    with open(filename, "r") as f:
        exec(f.read(), vglobal := {}, vlocal := {})
    simulation_settings = vlocal["inp"]

    # Do some checks
    assert isinstance(simulation_settings, dict)
    assert "hpath" in simulation_settings
    assert "nq" in simulation_settings
    assert "circ_init" in simulation_settings
    assert "x0_list" in simulation_settings
    assert "bounds" in simulation_settings
    assert "estimator" in simulation_settings
    assert "optimizer" in simulation_settings
    assert "mode_aml" in simulation_settings
    assert "err_exact" in simulation_settings
    assert "err" in simulation_settings
    assert "tol_prediction" in simulation_settings
    assert "mode_init_check" in simulation_settings
    assert "tpar" in simulation_settings
    assert "xlists_exact" in simulation_settings
    assert "nlayers" in simulation_settings

    # Return the simulation settings
    if DEBUG:
        print(f"[VQE_INIT] Finish _load_input_file({filename=})")
    return simulation_settings


# Function to load the INCAR file (incar.py)
def _load_incar_file(filename: str) -> dict:

    # Test incar file
    if DEBUG:
        print(f"[VQE_INIT] Called _load_incar_file({filename=})")
    assert os.path.isfile(filename)

    # Load the INCAR file (.py) and load it as a dictionary
    with open(filename, "r") as f:
        incar_settings = json.load(f)

    # Do some checks
    assert "h" in incar_settings
    assert "generators" in incar_settings

    # Return the INCAR dictionary
    if DEBUG:
        print(f"[VQE_INIT] Finish _load_incar_file({filename=})")
    return incar_settings


# Function to generate a hamiltonian from the INCAR file
def _generate_hamiltonian(hamiltonian: list) -> SparsePauliOp:

    # Check the input
    if DEBUG:
        print(f"[VQE_INIT] Called _generate_hamiltonian(hamiltonian)")
    assert isinstance(hamiltonian, list)

    # Generate the hamiltonian
    hamiltonian = SparsePauliOp.from_list([(x.split('*')[1], float(x.split('*')[0])) for x in hamiltonian])

    # Return the hamiltonian
    if DEBUG:
        print(f"[VQE_INIT] Finish _generate_hamiltonian(hamiltonian)")
    return hamiltonian


# Function to generate the circuit
def _generate_circuit(initial_circuit: QuantumCircuit, generators: list, num_qubits: int) -> VQECircuit:

    # Check the input
    if DEBUG:
        print(f"[VQE_INIT] Called _generate_circuit()")
    assert isinstance(initial_circuit, QuantumCircuit)
    assert isinstance(generators, list)
    assert isinstance(num_qubits, int)

    # Generate the parameters
    parameters = [Parameter(str(i)) for i in range(len(generators))]

    # Generate the circuit
    circuit = initial_circuit
    for parameter, generator in zip(parameters, generators):
        for label in generator:
            coefficient, label = label.split('*')
            op = SparsePauliOp.from_list([(label, float(coefficient))])
            circuit.append(PauliEvolutionGate(op, time=parameter), range(num_qubits))

    # Return the parameters
    if DEBUG:
        print(f"[VQE_INIT] Finish _generate_circuit()")
    return VQECircuit(circuit, parameters)


# Function to get the statevector of a circuit
def _generate_circuit_statevector(xlist: list, circuit: VQECircuit, num_qubits: int):

    # Check the input
    if DEBUG: print(f"_generate_circuit_statevector() was called.")
    assert isinstance(circuit, VQECircuit)
    assert isinstance(num_qubits, int)

    # Generate the statevector
    state = Statevector.from_int(0, 2 ** num_qubits)
    binds = {th: val for th, val in zip(circuit.parameters, xlist)}
    circ = circuit.circuit.bind_parameters(binds)
    state = state.evolve(circ)

    # Return the statevector
    if DEBUG: print(f"_generate_circuit_statevector() finished executing.")
    return state


# Function to find the exact ground state of a hamiltonian
def find_exact_groundstate(hamiltonian: SparsePauliOp) -> Eigenstate:

    # Check the input
    if DEBUG:
        print(f"[VQE_INIT] Called find_exact_groundstate()")
    assert isinstance(hamiltonian, SparsePauliOp)

    # Find the exact ground state
    hamiltonian_matrix = hamiltonian.to_matrix()
    eigenvalues, eigenvectors = numpy.linalg.eigh(hamiltonian_matrix)

    # Return the exact ground state
    if DEBUG:
        print(f"[VQE_INIT] Finish find_exact_groundstate()")
    return Eigenstate(eigenvalues[0], eigenvectors[:, 0])


class VQE:

    # Constructor
    def __init__(self, *, input_file: str, incar_file: str) -> None:

        # Print a message
        if DEBUG:
            print(f"[VQE_INIT] Initializing VQE object...")

        # Load the input file
        settings = _load_input_file(input_file)
        self._simulation_settings_unpacker(settings)  # Todo: Remove This Later
        self._x_list:   numpy.ndarray = settings["x0_list"]
        self._x_bounds: list = settings["bounds"]
        self._x_lists_exact: list = settings["xlists_exact"]
        self._estimator: EstimatorAER = settings["estimator"]
        self._optimizer:     GDQiskit = settings["optimizer"]
        _initial_circuit: QuantumCircuit = settings["circ_init"]
        _num_qubits: int = settings["nq"]
        _num_layers: int = settings["nlayers"]

        # Load the INCAR file
        incar = _load_incar_file(incar_file)
        _hamiltonian_list: list = incar["h"]
        self._generator:   list = incar["generators"] * _num_layers

        # Generate the Hamiltonian
        self._hamiltonian: SparsePauliOp = _generate_hamiltonian(_hamiltonian_list)
        self._num_qubits: int = self._hamiltonian.num_qubits
        assert (self._num_qubits == _num_qubits)

        # Generate the circuit
        self._circuit: VQECircuit = _generate_circuit(_initial_circuit, self._generator, self._num_qubits)

        # Load the AI model
        if self._mode_aml == 0:
            self._AML = None
        else:
            self._AML = AML(
                t=self._tpar,
                n_ranges=None,
                mode=self._mode_aml
            )

        # Initialize the exact ground state
        _ground_state: Eigenstate = find_exact_groundstate(self._hamiltonian)
        self._ground_state_energy = _ground_state.energy
        self._ground_state_vector = _ground_state.state

        # Check the initial state
        if self._mode_init_check > 0:
            self._check_init_state()

        # Initialize the records
        self._aml_records = {"evals": [], "is_prediction": []}
        self.optimal_result = None

        # Print a final message
        if DEBUG:
            print(f"[VQE_INIT] VQE object has been initialized.\n")

    @property
    def x_list(self) -> numpy.ndarray:
        return self._x_list

    @property
    def exact_ground_state_energy(self) -> float:
        return self._ground_state_energy.real

    # Check the initial state of the circuit
    def _check_init_state(self) -> None:

        # check initial state
        state = _generate_circuit_statevector(
            [0] * len(self._circuit.parameters),
            self._circuit,
            self._num_qubits
        )

        # Get H1 (https://scipost.org/SciPostPhys.6.3.029)
        H1 = SparsePauliOp.from_list([(x.split('*')[1], float(x.split('*')[0])) for x in self._generator[-1]])
        H1_matrix = H1.to_matrix()
        eigenvalues, eigenvectors = numpy.linalg.eigh(H1_matrix)
        expectation_value = state.expectation_value(H1)

        # Check the initial state
        assert (abs(eigenvalues[0] - expectation_value) < 1e-6)
        assert (abs(abs(eigenvectors[:, 0].dot(state._data)) - 1) < 1e-6)

        # Print the results
        res = state.expectation_value(self._hamiltonian)
        print(f"initial state energy exact = {res.real:.6f}")

    # Get the fidelity of a state
    def get_fidelity(self, xlist) -> float:
        state = _generate_circuit_statevector(xlist, self._circuit, self._num_qubits)
        fidelity = abs(self._ground_state_vector.dot(state._data)) ** 2
        return fidelity

    # Get the energy statevector
    def get_energy_sv(self, xlist) -> float:
        estimator = EstimatorAER(
            backend_options={
                "method": "statevector",
            },
            run_options={"shots": None},
            approximation=True,
        )
        job = estimator.run([self._circuit.circuit],[self._hamiltonian], xlist)
        return job.result().values[0]

    # Get the energy of a state
    def get_energy(self, xlist) -> float:
        assert (self._estimator is not None)
        job = self._estimator.run([self._circuit.circuit], [self._hamiltonian], xlist)
        return job.result().values[0]

    def _energy_function_algorithmic(self):
        eval_count = 0

        def evaluate_energy(xlist):
            nonlocal eval_count
            eval_count += 1
            result = self.get_energy(xlist)
            if DEBUG:
                print(f"[VQE] Evaluation {eval_count}: {result:.6f} at x: {numpy.round(xlist, decimals=3)}")
            return result

        return evaluate_energy

    def _energy_function_artificial(self):
        eval_count = 0

        def evaluate_energy(xlist):
            nonlocal eval_count
            eval_count += 1

            e_est, err_est = self._AML.prediction(xlist)

            if DEBUG:
                print(f"[VQE] Prediction {eval_count}: {e_est:.6f} +/- {err_est:.1e} @ X: {numpy.round(xlist, decimals=4)}")
            if err_est > self._tol_prediction:
                e_calc = self.get_energy(xlist)
                self._AML.add_training_data([xlist], [e_calc], err=self._err)
                e_est2, err_est2 = self._AML.prediction(xlist)
                if DEBUG:
                    print(f"[VQE]   Adjusted {eval_count}:  {e_est2:.6f} +/- {err_est2:.2e}, e_calc: {e_calc:.6f} @ X: {numpy.round(xlist, decimals=4)}")

                # use new prediction
                e_est = e_est2
                self._aml_records["is_prediction"].append(0)
            else:
                self._aml_records["is_prediction"].append(1)
            self._aml_records["evals"].append(e_est)

            # setting this to float(x) works for some reason
            return float(e_est)

        return evaluate_energy

    def _energy_function_artificial_with_gradient(self):
        eval_count = 0

        def evaluate_energy(xlist):
            nonlocal eval_count
            eval_count += 1

            e_est, err_est = self._AML.prediction(xlist)
            e_grad = self._AML.predict_gradient(xlist)

            if DEBUG:
                print(f"[VQE] G Prediction {eval_count}: {e_est:.6f} +/- {err_est:.1e} @ X: {numpy.round(xlist, decimals=4)}")
            if err_est > self._tol_prediction:
                e_calc = self.get_energy(xlist)
                self._AML.add_training_data([xlist], [e_calc], err=self._err)
                e_est2, err_est2 = self._AML.prediction(xlist)
                e_grad_2 = self._AML.predict_gradient(xlist)
                if DEBUG:
                    print(f"[VQE]   G Adjusted {eval_count}:  {e_est2:.6f} +/- {err_est2:.2e}, e_calc: {e_calc:.6f} @ X: {numpy.round(xlist, decimals=4)}")

                # use new prediction
                e_est = e_est2
                e_grad = e_grad_2
                self._aml_records["is_prediction"].append(0)
            else:
                self._aml_records["is_prediction"].append(1)
            self._aml_records["evals"].append(e_est)

            # setting this to float(x) works for some reason
            return float(e_est), e_grad

        return evaluate_energy

    # Energy minimization function
    def minimize_energy(self) -> None:

        # If the AML is disabled, use the algorithmic energy function
        if self._AML is None:
            cost_fun = self._energy_function_algorithmic()
        elif OLD_OPTIMIZER:
            cost_fun = self._energy_function_artificial()
        else:
            cost_fun = self._energy_function_artificial_with_gradient()

        # Minimize the energy
        if OLD_OPTIMIZER:
            result = self._optimizer.minimize(cost_fun, self._x_list, bounds=self._x_bounds)
        else:
            result = self._gradient_descent(cost_fun, self._x_list, bounds=self._x_bounds)

        # Print the results
        self.optimal_result = result
        energy = self.get_energy(result.x)
        print(f"Minimal Energy Measured: {energy:.6f}")
        # if the records has data, print it
        if not (self._aml_records["evals"] and self._aml_records["is_prediction"]):
            print(f"Records: {self._aml_records}")
        if not self._aml_records["is_prediction"]:
            print(f"non-prediction times: {self._aml_records['is_prediction'].count(0)}")


    # Gradient Descent
    def _gradient_descent(self, cost_fun, x0, bounds) -> FakeResult:
        min_step = 1e-3
        max_step = 1
        maxiter = 300

        x0 = numpy.array(x0)
        for _ in range(maxiter):
            energy, gradient = cost_fun(x0)
            step = max_step * (max_step/min_step) ** (_ / maxiter)
            x0 -= step * gradient

            # if any of the points are out of bounds, loop them back
            for i in range(len(x0)):
                lower = bounds[i][0]
                upper = bounds[i][1]
                if x0[i] < lower:
                    x0[i] = upper - (lower - x0[i])
                if x0[i] > upper:
                    x0[i] = lower + (x0[i] - upper)

            if DEBUG:
                print(f"[VQE] Gradient Iteration {_}: Energy: {energy:.6f}, Gradient: {gradient}")

        return FakeResult(x0)




    # Estimate t parameter (for AML)
    def estimate_t(self, num_sample: int = 100) -> None:
        x_list = numpy.random.rand(num_sample, len(self._x_list))
        energy_list = [self.get_energy(x) ** 2 for x in x_list]
        t_mean, t_std = numpy.mean(energy_list), numpy.std(energy_list)
        # energy_list = []
        # for x in x_list:
        #     energy = self.get_energy(x)
        #     energy_list.append(energy ** 2)
        #     print(energy)
        # t_est, t_std = numpy.mean(energy_list), numpy.std(energy_list)
        print(f"Estimated t: {t_mean:.2e} with std: {t_std:.2e}")

    # Set exact points
    # def set_exact_points(self, err_exact=1e-8):
    #     x_lists = [i for i in self._x_lists_exact]
    #     energy_list = [self.get_energy_sv(xlist).real for xlist in x_lists]
    #     self._AML.add_training_data(x_lists, energy_list, err=err_exact)

    # Simulation Settings Unpacker
    def _simulation_settings_unpacker(self, simulation_settings: dict):

        self._mode_aml: int = simulation_settings["mode_aml"]
        self._err: float = simulation_settings["err"]
        self._tol_prediction: float = simulation_settings["tol_prediction"]
        self._mode_init_check: int = simulation_settings["mode_init_check"]
        self._tpar: int = simulation_settings["tpar"]

    # Save circuit to file
    def save_circuit(self) -> None:
        with open("output/parameter_circuit.pkl", "wb") as f:
            pickle.dump(self._circuit, f, pickle.HIGHEST_PROTOCOL)

    # Save records to file
    def save_records(self) -> None:
        with open("output/records.pkl", "wb") as f:
            pickle.dump(self._aml_records, f, pickle.HIGHEST_PROTOCOL)
