# Manage imports
import argparse


from source.vqe import VQE



# Parse command line arguments
parser = argparse.ArgumentParser(description=
                                 "Driver for vqe_active learning job.")
parser.add_argument("-m", "--mode", type=int, default=0,
                    help="job mode. 0: vqe calculation (default); otherwise: estimate t.")
parser.add_argument("-n", "--nsample", type=int, default=100,
                    help="sample size. 100(default)")

# Run VQE
vqe = VQE(imp_file="source/inp.py")
args = parser.parse_args()

if args.mode == 0:
    vqe.minimize_energy()
    fid = vqe.get_fidelity_with_params(vqe.get_x0_list())
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
