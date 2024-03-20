import numpy
import os
import pickle
from qiskit.circuit import QuantumCircuit
from qiskit_aer.primitives import Estimator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_provider import IBMProvider
from qiskit_algorithms.optimizers import GradientDescent as GDQiskit



# Initial State Preparation: Product of Bell Pairs.
NUM_QUBITS = 8
circuit = QuantumCircuit(NUM_QUBITS)
for i in range(NUM_QUBITS // 2):
    circuit.x(2 * i + 1)
    circuit.h(2 * i)
    circuit.z(2 * i)
    circuit.z(2 * i + 1)
    circuit.cx(2 * i, 2 * i + 1)
# print(circuit.draw())





SIMULATION_DEVICE = "settings/ibmq_mumbai"
NUM_PARAMETERS = 2


### estimator
seed = None
shots = 2 ** 10
nm_fname = f"{SIMULATION_DEVICE}_nm.pkl"
if os.path.isfile(nm_fname):
    with open(nm_fname, "rb") as f:
        coupling_map, noise_model = pickle.load(f)
else:
    provider = IBMProvider()
    backend = provider.get_backend(SIMULATION_DEVICE)
    noise_model = NoiseModel.from_backend(backend)
    coupling_map = backend.configuration().coupling_map
    with open(nm_fname, "wb") as f:
        pickle.dump([coupling_map, noise_model],
                    f,
                    pickle.HIGHEST_PROTOCOL,
                    )
estimator = Estimator(
    backend_options={
        "method": "density_matrix",
        "coupling_map": coupling_map,
        "noise_model": noise_model,
    },
    run_options={"seed": seed, "shots": shots},
    transpile_options={"seed_transpiler": seed,
                       "initial_layout": [0, 1, 4, 7, 10, 12, 13, 14],
                       # "initial_layout": [12, 15, 18, 21, 23, 24, 25, 26],  # worst connection
                       },
    approximation=False,
)
### optimizer
optimizer = GDQiskit(maxiter=100,
                    learning_rate=0.005,
                    perturbation=0.005,
                    )

xlists_exact = [[0, x] for x in numpy.linspace(-numpy.pi/2, numpy.pi/2, 13, endpoint=False)]
# xlists_exact = []







inp = {
    "hpath": "../../../",
    "nq": NUM_QUBITS,
    "circ_init": circuit,
    "x0_list": [numpy.pi / 5] * NUM_PARAMETERS,
    "bounds": [(-numpy.pi, numpy.pi)] * NUM_PARAMETERS,
    "estimator": estimator,
    "optimizer": optimizer,
    "mode_aml": 1,
    "err_exact": 1e-6,
    "err": 0.001,
    "tol_prediction": 0.001,
    "mode_init_check": 0,
    "tpar": 2,
    "xlists_exact": xlists_exact,
    "nlayers": 1,
}
