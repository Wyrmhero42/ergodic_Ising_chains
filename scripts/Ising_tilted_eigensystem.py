import sys
import os
import h5py
import numpy as np
import scipy
from datetime import datetime
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import functions.Hamiltonians as ham
import module


if __name__ == "__main__":

    # L = 16, dim=L^2 = 65536
    # L = 17, dim=L^2 = 131072

    L = int(sys.argv[1])

    #ergodic chain params (Jz, hx, hz) = (−2, 3.375, 2)
    J_z, h_x, h_z = -2.0, 3.375, 2.0
    
    param_str = f"L{L}_Jz{J_z}_hx{h_x}_hz{h_z}"

    print(f"Building Hamiltonian...", flush=True)
    H = ham.tilted_field_Ising_Hamiltonian_sparse(L, J_z, h_x, h_z)

    print("Diagonalizing Hamiltonian...", flush=True)
    start_time = datetime.now()
    eigvals, eigvecs = np.linalg.eigh(H.toarray())
    print(f"Diagonalization time: {datetime.now() - start_time}", flush=True)

    start_time = datetime.now()
    module.save_Hamiltonian_custom("Ising_tilted", param_str, H, sparse_csr=True)
    module.save_eigensystem_custom("Ising_tilted", param_str, eigvals, eigvecs)
    print(f"saving finished: {datetime.now() - start_time}", flush=True)
