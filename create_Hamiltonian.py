# function for sparse transverse Ising Hamiltonian, ground state and sparese time evolution

import numpy as np
import scipy.sparse as sp
from datetime import datetime
import scipy.sparse.linalg as spla
import scipy.linalg as la
import matplotlib.pyplot as plt


def _kron_list_sparse(ops):
    """Kronecker product of a list of sparse operators -> CSR."""
    mat = ops[0]
    for op in ops[1:]:
        mat = sp.kron(mat, op, format='csr')
    return mat

def one_site_op_sparse(op, site, N, dtype=np.float64):
    Id_left = sp.eye(2**site, format='csr', dtype=dtype)
    Id_right = sp.eye(2**(N-site-1), format='csr', dtype=dtype)

    # edge cases
    if site == 0:
        return _kron_list_sparse([op, Id_right])
    if site == N - 1:
        return _kron_list_sparse([Id_left, op])
    else:
        return _kron_list_sparse([Id_left, op, Id_right])

def two_site_op_sparse(op1, op2, site_of_op1: int, N: int, dtype=np.float64):
    """Construct a nearest neighbor two-site operator as a sparse matrix."""

    Id_left = sp.eye(2**site_of_op1, format='csr', dtype=dtype)
    Id_right = sp.eye(2**(N-site_of_op1-2), format='csr', dtype=dtype)

    # edge cases
    if site_of_op1 == 0:
        return _kron_list_sparse([op1, op2, Id_right])
    if site_of_op1 == N - 2:
        return _kron_list_sparse([Id_left, op1, op2])
    else:
        return _kron_list_sparse([Id_left, op1, op2, Id_right])



# Pauli X and Z (full sigma matrices)
sigma_x = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64))
sigma_z = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64))


def tilted_field_Ising_Hamiltonian_sparse(L, J_z, h_x, h_z, print_info=True, dtype=np.float64):
    """
    Build the tilted field Ising Hamiltonian as a scipy.sparse.csr_matrix.
    
    H1 original (tilted-field Ising, open chain):
    H1 = sum_{i=0..L-2} Jz * sigma_i^z sigma_{i+1}^z
                        + (hx/2) * (sigma_i^x + sigma_{i+1}^x)
                        + (hz/2) * (sigma_i^z + sigma_{i+1}^z)
                        
    Simplified for open boundary conditions (collect on-site terms):
    - The bulk sites (i = 1..L-2) see full fields hx, hz.
    - The edge sites (i = 0 and i = L-1) see half-strength fields hx/2, hz/2.
    
    H1 = sum_{i=0..L-2} Jz * sigma_i^z sigma_{i+1}^z
       + hx * sum_{i=1..L-2} sigma_i^x
       + (hx/2) * (sigma_0^x + sigma_{L-1}^x)
       + hz * sum_{i=1..L-1} sigma_i^z
       + (hz/2) * (sigma_0^z + sigma_{L-1}^z)

    Returns:
     - H: real csr_matrix of shape (2**L, 2**L)
    
    Also prints: shape, density and total CSR memory (bytes and MB) and build time.
    """
    if L < 2:
        raise ValueError("L must be >= 2")

    # start timing the construction
    start_t = datetime.now()

    # Pauli X and Z (full sigma matrices)
    sigma_x = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64))
    sigma_z = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64))

    dim = 2**L

    Hamiltonian = sp.csr_matrix((dim, dim), dtype=np.float64)

    #sum_{i=0..L-2} Jz * sigma_i^z sigma_{i+1}^z # dont forget array index shift
    for i in range(L - 1):
        Hamiltonian += two_site_op_sparse(J_z * sigma_z, sigma_z, i, L)
    # + hx * sum_{i=1..L-2} sigma_i^x
    # + hz * sum_{i=1..L-2} sigma_i^z
    for i in range(1, L - 1):
        Hamiltonian += one_site_op_sparse(h_x * sigma_x, i, L)
        Hamiltonian += one_site_op_sparse(h_z * sigma_z, i, L)
    # + (hx/2) * (sigma_0^x + sigma_{L-1}^x)
    # + (hz/2) * (sigma_0^z + sigma_{L-1}^z)
    for i in [0, L - 1]:
        Hamiltonian += one_site_op_sparse((h_x / 2.0) * sigma_x, i, L)
        Hamiltonian += one_site_op_sparse((h_z / 2.0) * sigma_z, i, L)
    
    # finish timing
    elapsed = datetime.now() - start_t

    # Print diagnostics: size, density, CSR memory, build time
    if print_info:
        num_nonzero_elements = Hamiltonian.nnz
        total_entries = dim**2
        density = num_nonzero_elements / float(total_entries)
        mem_bytes = Hamiltonian.data.nbytes + Hamiltonian.indices.nbytes + Hamiltonian.indptr.nbytes
        
        print(f"H shape: {Hamiltonian.shape}")
        print(f"non-zeros: {num_nonzero_elements} / {total_entries} (density = {density:.3e})")
        print(f"CSR memory: {mem_bytes / 1024**2  :.2f} MB")
        print(f"H build time:", elapsed)
        
    return Hamiltonian.astype(dtype=dtype)


def XZ_transverse_field_Ising_Hamiltonian_sparse(L, J_x, J_z, h_x, print_info=True, dtype=np.float64):
    """
    Build the XZ transverse Hamiltonian as a scipy.sparse.csr_matrix.
    
    H2 original (XZ chain in transverse field, open chain):
    H2 = sum_{i=0..L-2} Jx * sigma_i^x sigma_{i+1}^x
                        + Jz * sigma_i^z sigma_{i+1}^z
                        + (hx/2) * (sigma_i^x + sigma_{i+1}^x)
                        
    Simplified for open boundary conditions:
    - The bulk sites (i = 1..L-2) see full field hx.
    - The edge sites (i = 0 and i = L-1) see half-strength field hx/2.

    H2 = sum_{i=0..L-2} ( Jx * sigma_i^x sigma_{i+1}^x
                        + Jz * sigma_i^z sigma_{i+1}^z )
       + hx * sum_{i=1..L-2} sigma_i^x
       + (hx/2) * (sigma_0^x + sigma_{L-1}^x)
            Returns:
     - H: real csr_matrix of shape (2**L, 2**L)
    
    Also prints: shape, density and total CSR memory (bytes and MB) and build time.
    """  
     
    if L < 2:
        raise ValueError("L must be >= 2")

    # start timing the construction
    start_t = datetime.now()

    # Pauli X and Z (full sigma matrices)
    sigma_x = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64))
    sigma_z = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64))

    dim = 2**L

    Hamiltonian = sp.csr_matrix((dim, dim), dtype=np.float64)

    # sum_{i=0..L-2} Jx * sigma_i^x sigma_{i+1}^x + Jz * sigma_i^z sigma_{i+1}^z
    for i in range(L - 1):
        Hamiltonian += two_site_op_sparse(J_x * sigma_x, sigma_x, i, L)
        Hamiltonian += two_site_op_sparse(J_z * sigma_z, sigma_z, i, L)
    # + hx * sum_{i=1..L-2} sigma_i^x
    for i in range(1, L - 1):
        Hamiltonian += one_site_op_sparse(h_x * sigma_x, i, L)
    # + (hx/2) * (sigma_0^x + sigma_{L-1}^x)
    for i in [0, L - 1]:
        Hamiltonian += one_site_op_sparse((h_x / 2.0) * sigma_x, i, L)

    # finish timing
    elapsed = datetime.now() - start_t

    # Print diagnostics: size, density, CSR memory, build time
    if print_info:
        num_nonzero_elements = Hamiltonian.nnz
        total_entries = dim**2
        density = num_nonzero_elements / float(total_entries)
        mem_bytes = Hamiltonian.data.nbytes + Hamiltonian.indices.nbytes + Hamiltonian.indptr.nbytes
        
        print(f"H shape: {Hamiltonian.shape}")
        print(f"non-zeros: {num_nonzero_elements} / {total_entries} (density = {density:.3e})")
        print(f"CSR memory: {mem_bytes / 1024**2  :.2f} MB")
        print(f"H build time:", elapsed)
        
    return Hamiltonian.astype(dtype=dtype)
    
    

H = tilted_field_Ising_Hamiltonian_sparse(20, J_z=1.0, h_x=0.5, h_z=0.3)
# for XZ Hamiltonian later
H_xz = XZ_transverse_field_Ising_Hamiltonian_sparse(20, J_x=1.0, J_z=1.0, h_x=0.5)