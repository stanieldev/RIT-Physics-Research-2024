# VQE Simulation Project
This project attempts to use the Variational Quantum Eigensolver (VQE) algorithm, in conjunction with AI, to find the minimum energy of a quantum system by reducing the number of calls to a quantum computer. The project is based on the paper "Error mitigation in variational quantum eigensolvers using tailored probabilistic
machine learning" by T. Jiang, J. Rogers, M. S. Frank, O. Christiansen, Y. Yao, N. Lanata. The paper can be found [here](https://arxiv.org/pdf/2111.08814.pdf).

## Installation
The project requires Python 3.9. I have been unsuccessful with porting to newer versions as of right now.
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

## Usage
To run the project, use the following command:
```bash
python vqe_drive.py
```
As of right now, the command-line arguments are not used in the project. The user can change the parameters in the `vqe_drive.py` file.

## Contributing
None. This is a personal project.

## License
Not my code. I do not own the rights to the code. The code is based on the paper "Error mitigation in variational quantum eigensolvers using tailored probabilistic machine learning" by T. Jiang, J. Rogers, M. S. Frank, O. Christiansen, Y. Yao, N. Lanata. The paper can be found [here](https://arxiv.org/pdf/2111.08814.pdf). The source code was provided by the authors of the paper.

## Project Status
The project is still in development. The code is functional but not optimized. The code is not yet ready for use in a production environment.

## Acknowledgements
The authors of the paper "Error mitigation in variational quantum eigensolvers using tailored probabilistic machine learning" by T. Jiang, J. Rogers, M. S. Frank, O. Christiansen, Y. Yao, N. Lanata. The paper can be found [here](https://arxiv.org/pdf/2111.08814.pdf). The source code was provided by the authors of the paper.
