"""
Author : Stanley Goodwin
Date : 02/20/2024
Purpose : This file is used to save the IBM API Key to the local system.
          This is necessary for running the VQE algorithm on the IBM Quantum
          Experience backend. This file is included in the repository.
"""


# Universal import for IBM API Key
from qiskit_ibm_provider import IBMProvider

# If reading from an environment variable:
import os
IBM_API_KEY = os.environ.get("IBM_API_KEY")

# # Otherwise, hardcode the API key here (not recommended for production code):
# IBM_API_KEY = "your_api_key_here"

# Save the account
IBMProvider.save_account(IBM_API_KEY, overwrite=True)
