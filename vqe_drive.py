# developed in c2qa.
import argparse


from vqe import VQE

KEY = "18846f5795a2258e16d93944a2a7ed1610de539e6981ee9244f3721066c2b5999f12d2aabfb5245e04df97e9e13d948c4eff81d5ed9cbc3c6b5552d78383f59f"
from qiskit_ibm_provider import IBMProvider
IBMProvider.save_account(KEY, overwrite=True)




parser = argparse.ArgumentParser(description=
        "Driver for vqe_active learning job.")
parser.add_argument("-m", "--mode", type=int, default=0,
        help="job mode. 0: vqe calculation (default); otherwise: estimate t.")
parser.add_argument("-n", "--nsample", type=int, default=100,
        help="sample size. 100(default)")


vqe = VQE()
args = parser.parse_args()

if args.mode == 0:
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
else:
    vqe.estimate_t(nsample=args.nsample)
