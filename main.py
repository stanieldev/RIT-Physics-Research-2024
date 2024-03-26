# developed in c2qa.
import argparse
from VQE import VQE

# Control Panel
DEBUG = True


# Implement an argument parser
parser = argparse.ArgumentParser(description='VQE')
parser.add_argument("-f", "--input", type=str, help="Simulation input file directory.")
parser.add_argument("-i", "--incar", type=str, help="Choose the INCAR file to use.")

# Parse the arguments
args = parser.parse_args()


# Execution
vqe = VQE(input_file=args.input, incar_file=args.incar)

# Minimize energy
if DEBUG:
    print("[MAIN] Minimizing the energy...\n")
vqe.minimize_energy()
if DEBUG:
    print("[MAIN] Finished minimizing energy.\n")

# Find the fidelity of the initial state
fid = vqe.get_fidelity(vqe.x_list)
print(f"initial fidelity: {fid:.6f}")

# Find the fidelity of the final state
fid = vqe.get_fidelity(vqe.optimal_result.x)
print(f"final fidelity: {fid:.6f}")

# Find the final energy
e_sc = vqe.get_energy_sv(vqe.optimal_result.x)

# Print the results
print(f"exact ground state energy: {vqe.exact_ground_state_energy:.6f}")
print(f"exact final ansatz energy: {e_sc:.6f}")
print(f"final solution: {vqe.optimal_result.x}")

# Save the results
vqe.save_circuit()
vqe.save_records()



ESTIMATE_T = False
if ESTIMATE_T:
    vqe.estimate_t(num_sample=100)