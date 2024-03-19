# developed in c2qa.
import argparse
from old.vqe import VQE


# Implement an argument parser
parser = argparse.ArgumentParser(description='VQE')
parser.add_argument("-f", "--input", type=str, help='Simulation input file directory.')

# Parse the arguments
args = parser.parse_args()
input_dir = args.input
print(f"input_dir: {input_dir}")


# Execution
vqe = VQE()
vqe.minimize_energy()
fid = vqe.get_fidelity_with_params(vqe._inp["x0_list"])
print(f"initial fidelity: {fid:.6f}")
fid = vqe.get_fidelity_with_params(vqe._res_opt.x)
print(f"final fidelity: {fid:.6f}")
e_sc = vqe.get_energy_sv(vqe._res_opt.x)
print(f"exact final ansatz energy: {e_sc:.6f}")
print(f"final solution: {vqe._res_opt.x}")
vqe.save_param_circ()
vqe.save_records()



ESTIMATE_T = False
if ESTIMATE_T:
    vqe.estimate_t(nsample=100)