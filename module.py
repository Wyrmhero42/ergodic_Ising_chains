import os
import h5py
import numpy as np
import multiprocessing

from scipy.optimize import fsolve
from scipy.linalg import eigvals
from numpy.linalg import matrix_power


###########################################
# FILE SYSTEM
#
# Add a file 'storage_path.txt' in the same directory as module.py which contains the path of the file
# which contains the code output, i.e., basis states, Hamiltonian, eigensystem etc.
# Example content of 'storage_path.txt': my_secret_path.h5
#
###########################################

def get_path():  # choose path of ETH file
    with open('storage_path.txt', 'r') as fh:
        path = fh.read().strip()
        fh.close()
    return path


def save_momentum_states_and_symmetries(k, P, jmax, basis, sym):
    print('saving states...', flush=True)
    path = get_path()
    hdf = h5py.File(path, 'a')
    try:
        hdf.create_dataset(
            '/Periodic/States/States_Periodic_pa_tb_k=%1.0f' % k + '_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '.h5',
            data=basis)
        hdf.create_dataset(
            '/Periodic/States/Sym_Periodic_pa_tb_k=%1.0f' % k + '_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '.h5',
            data=sym)
        hdf.close()
    except Exception as e:
        print(e)
        hdf.close()


def import_momentum_states(k, P, jmax):
    print('importing states...', flush=True)
    path = get_path()
    hdf = h5py.File(path, 'r')
    if 'Periodic/States' in hdf:
        G1 = hdf.get('Periodic/States')
        B = np.array(G1.get('States_Periodic_pa_tb_k=%1.0f' % k + '_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '.h5'))
        hdf.close()
        return B
    else:
        hdf.close()
        print('**ERROR**')
        exit()


def import_symmetries(k, P, jmax):
    print('importing symmetries...', flush=True)
    path = get_path()
    hdf = h5py.File(path, 'r')
    if 'Periodic/States' in hdf:
        G1 = hdf.get('Periodic/States')
        Sym = np.array(G1.get('Sym_Periodic_pa_tb_k=%1.0f' % k + '_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '.h5'))
        hdf.close()
        return Sym
    else:
        hdf.close()
        print('**ERROR**')
        exit()


def save_plaquette_term(P, jmax, HM):
    print('saving plaquette term...', flush=True)
    path = get_path()
    hdf = h5py.File(path, 'a')
    try:
        hdf.create_dataset(
            '/Periodic/Plaquette_term/Plaquette_pa=1_tb=1_k=0_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '.h5', data=HM,
            compression="gzip")
        hdf.close()
    except Exception as e:
        print(e)
        hdf.close()


def import_plaquette_term(P, jmax):
    print('importing plaquette term...', flush=True)
    path = get_path()
    hdf = h5py.File(path, 'r')
    if 'Periodic/Plaquette_term' in hdf:
        G2 = hdf.get('Periodic/Plaquette_term')
        HM = np.array(G2.get('Plaquette_pa=1_tb=1_k=0_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '.h5'))
        hdf.close()
        return HM
    else:
        hdf.close()
        print('**ERROR**')
        exit()


def save_eigensystem(P, jmax, g2, w, v):
    print('saving eigensystem...', flush=True)
    path = get_path()
    hdf = h5py.File(path, 'a')
    try:
        hdf.create_dataset('/Periodic/Eigenvalues/Eigenvalues_Periodic_pa=1_tb=1_k=0_P=%1.0f'%P+'_jmax=%1.1f'%jmax+'_g2=%1.2f'%g2+'.h5',data=w)
        hdf.create_dataset('/Periodic/Eigenvectors/Eigenvectors_Periodic_pa=1_tb=1_k=0_P=%1.0f'%P+'_jmax=%1.1f'%jmax+'_g2=%1.2f'%g2+'.h5',data=v, compression="gzip")
        hdf.close()
    except Exception as e:
        print(e)
        hdf.close()


def import_eigenvalues(P, jmax, g2):
    print('importing eigenvalues...', flush=True)
    path = get_path()
    hdf = h5py.File(path, 'r')
    filename = 'Periodic/Eigenvalues/Eigenvalues_Periodic_pa=1_tb=1_k=0_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '_g2=%1.2f' % g2 + '.h5'
    if filename in hdf:
        w = np.array(hdf.get(filename))
        hdf.close()
        return w
    hdf.close()
    filename = './per_output/w_spectrum_pa=1_tb=1_k=0_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '_g2=%1.2f' % g2 + '.npy'
    if os.path.exists(filename):
        w = np.load(filename)
        return w
    print('**ERROR**')
    exit()


def import_eigenvectors(P, jmax, g2):
    print('importing eigenvectors...', flush=True)
    path = get_path()
    hdf = h5py.File(path, 'r')
    filename = 'Periodic/Eigenvectors/Eigenvectors_Periodic_pa=1_tb=1_k=0_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '_g2=%1.2f' % g2 + '.h5'
    if filename in hdf:
        v = np.array(hdf.get(filename))
        hdf.close()
        return v
    else:
        hdf.close()
        print('**ERROR**')
        exit()


def save_rdm_indices(k, P, jmax, C1, C2, all_indices, iu1_0, iu1_1, dim):
    print('saving RDM indices...', flush=True)
    path = get_path()
    hdf = h5py.File(path, 'a')
    EEpath = 'Periodic/EEstuff_k=%1.0f' % k + '_P=%1.0f' % P + '_jmax=%1.1f' % jmax
    try:
        hdf.create_group(EEpath)
    except Exception as e:
        print(e)
    try:
        # dt = h5py.vlen_dtype(np.dtype('i,i'))
        # dset = hdf.create_dataset(
        #     EEpath+'/RDM_indices_Periodic_pa=1_tb=1_k=0_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '_C1=%1.0f' % C1 + '_C2=%1.0f' % C2 + '.h5', (len(all_indices),), dtype=dt, compression="gzip")
        # for i, sublist in enumerate(all_indices):
        #     dset[i] = np.array(sublist, dtype='i,i')
        ###
        # Flatten the data
        flat_data = [item for sublist in all_indices for item in sublist]
        offsets = np.zeros(len(all_indices) + 1, dtype='int32')
        # Compute offsets
        count = 0
        for i, sublist in enumerate(all_indices):
            count += len(sublist)
            offsets[i + 1] = count
        flat_data = np.array(flat_data, dtype='int32')
        ###
        hdf.create_dataset(
            EEpath + '/flat_data_Periodic_pa=1_tb=1_k=0_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '_C1=%1.0f' % C1 + '_C2=%1.0f' % C2 + '.h5',
            data=flat_data,
            compression="gzip")
        hdf.create_dataset(
            EEpath + '/offsets_Periodic_pa=1_tb=1_k=0_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '_C1=%1.0f' % C1 + '_C2=%1.0f' % C2 + '.h5',
            data=offsets)
        hdf.create_dataset(
            EEpath+'/iu1_0_Periodic_pa=1_tb=1_k=0_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '_C1=%1.0f' % C1 + '_C2=%1.0f' % C2 + '.h5', data=iu1_0,
            compression="gzip")
        hdf.create_dataset(
            EEpath+'/iu1_1_Periodic_pa=1_tb=1_k=0_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '_C1=%1.0f' % C1 + '_C2=%1.0f' % C2 + '.h5', data=iu1_1,
            compression="gzip")
        hdf.create_dataset(
            EEpath+'/dimRDM_Periodic_pa=1_tb=1_k=0_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '_C1=%1.0f' % C1 + '_C2=%1.0f' % C2 + '.h5', data=dim)
    except Exception as e:
        print(e)
    hdf.close()


def import_rdm_indices(k, P, jmax, C1, C2):  # returns 5 objects
    print('importing RDM indices...', flush=True)
    path = get_path()
    f = h5py.File(path, 'r')
    EEpath = 'Periodic/EEstuff_k=%1.0f' % k + '_P=%1.0f' % P + '_jmax=%1.1f' % jmax
    try:
        # dset = f[
        #     EEpath + '/RDM_indices_Periodic_pa=1_tb=1_k=0_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '_C1=%1.0f' % C1 + '_C2=%1.0f' % C2 + '.h5']
        # # Iterate through the dataset
        # all_indices = [0] * len(dset)
        # for i in range(len(dset)):
        #     item = dset[i]
        #     # Convert to list of tuples
        #     tuples_list = [tuple(t) for t in item]
        #     all_indices[i] = tuples_list
        #
        data = np.array(
            f.get(
                EEpath + '/flat_data_Periodic_pa=1_tb=1_k=0_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '_C1=%1.0f' % C1 + '_C2=%1.0f' % C2 + '.h5'))
        offsets = np.array(
            f.get(
                EEpath + '/offsets_Periodic_pa=1_tb=1_k=0_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '_C1=%1.0f' % C1 + '_C2=%1.0f' % C2 + '.h5'))
        # lists = [data[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
        # all_indices = [[tuple(x) for x in sublist] for sublist in lists]

        iu1_0 = np.array(
            f.get(
                EEpath + '/iu1_0_Periodic_pa=1_tb=1_k=0_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '_C1=%1.0f' % C1 + '_C2=%1.0f' % C2 + '.h5'))
        iu1_1 = np.array(
            f.get(
                EEpath + '/iu1_1_Periodic_pa=1_tb=1_k=0_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '_C1=%1.0f' % C1 + '_C2=%1.0f' % C2 + '.h5'))
        dim = np.array(f.get(
            EEpath + '/dimRDM_Periodic_pa=1_tb=1_k=0_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '_C1=%1.0f' % C1 + '_C2=%1.0f' % C2 + '.h5'))
        f.close()
        return data, offsets, iu1_0, iu1_1, dim
    except Exception as e:
        print(e)
        f.close()


def save_rdm_spectrum(P, jmax, g2, C1, C2, spectrum):  # save the whole spectrum of reduced density matrices (compressed)
    print('saving rdm spectrum...', flush=True)
    path = get_path()
    hdf = h5py.File(path, 'a')
    try:
        hdf.create_dataset('/Periodic/RDMspec/rdm_spectrum_pa=1_tb=1_k=0_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '_C1=%1.0f' % C1 + '_C2=%1.0f' % C2 + '_g2=%1.2f' % g2+'.h5', data=spectrum, compression="gzip")
        hdf.close()
    except Exception as e:
        print(e)
        hdf.close()


def import_rdm_spectrum(P, jmax, g2, C1, C2):
    print('importing rdm spectrum...', flush=True)
    path = get_path()
    hdf = h5py.File(path, 'r')
    filename = '/Periodic/RDMspec/rdm_spectrum_pa=1_tb=1_k=0_P=%1.0f' % P + '_jmax=%1.1f' % jmax + '_C1=%1.0f' % C1 + '_C2=%1.0f' % C2 + '_g2=%1.2f' % g2+'.h5'
    if filename in hdf:
        spec = np.array(hdf.get(filename))
        hdf.close()
        return spec
    else:
        hdf.close()
        print('**ERROR**')
        exit()


###########################################
# STATE VISUALIZATION
###########################################

def draw_chain(state):
    spaces = 8
    lower = state[0::3]
    lower = (" "*spaces).join(f"{num:.1f}" for num in lower)
    upper = state[1::3]
    upper = (" "*spaces).join(f"{num:.1f}" for num in upper)
    mid = state[2::3]
    mid = " "*(round(spaces/2)+1)+"|"+(" "*(spaces-1)+"|").join(f"{num:.1f}" for num in mid)

    print(upper, flush=True)
    print('-'*len(lower)+"-"*round(spaces/2), flush=True)
    print(mid, flush=True)
    print('-' * len(upper)+"-"*round(spaces/2), flush=True)
    print(lower, flush=True)


###########################################
# COMPUTATION SETTINGS
###########################################

def computation_settings(threads):
    os.environ['OMP_NUM_THREADS'] = '%1.0f' % threads
    os.environ['MKL_NUM_THREADS'] = '%1.0f' % threads
    os.environ['NUMEXPR_NUM_THREADS'] = '%1.0f' % threads
    processes = int(multiprocessing.cpu_count() / threads)
    print('new #threads: ', threads, flush=True)
    print('new #processes: ', processes, flush=True)
    return threads, processes


###########################################
# COMPUTATIONS
###########################################

def Hel_periodic(B):  # B is basis
    HE = np.zeros(len(B))
    chainlen = len(B[0])
    for i in range(len(B)):
        m = 0
        for v in range(chainlen):
            m = m + (B[i][v] * (B[i][v] + 1))
        HE[i] = m
    HE = np.matrix(np.diag(HE))
    return HE


def calculate_Z(w, beta):
    return np.sum(np.exp(-beta * w))


# Define derivative of Log[Z] with respect to beta
def derivative_log_Z(w, beta):
    Z = calculate_Z(w, beta)
    return np.sum(w * np.exp(-beta * w)) / Z


# Define the equation to solve
def equation_to_solve(beta, val, w):
    return derivative_log_Z(w, beta) - val


def solve_beta(energy, w):
    initial_guess = 0
    beta = fsolve(equation_to_solve, initial_guess, args=(energy, w))[0]
    return beta


def thermal_rdm(beta, w, rdm_spec):
    # weights = np.exp(-beta * w)[:, np.newaxis, np.newaxis]
    # rdm = np.sum(weights * rdm_spec, axis=0)
    # rdm = rdm / np.trace(rdm)
    rdm = np.zeros(rdm_spec[0].shape)
    for i in range(rdm_spec.shape[0]):
        rdm += rdm_spec[i] * np.exp(- beta * w[i])
    rdm /= np.trace(rdm)
    return rdm


def EE(rho):
    val = eigvals(rho)
    del rho
    val = val[np.round(val, 15) > 0]
    return np.abs(-np.sum(val * np.log(val)))


def nth_moment(rdm, n):
    return np.trace(matrix_power(rdm, n))


def anti_flatness(rdm):
    return nth_moment(rdm, 3) - nth_moment(rdm, 2)**2


def anti_flatness_normalized(rdm):
    return nth_moment(rdm, 3) / nth_moment(rdm, 2)**2 - 1


def PE(state, k):  # where the computational basis is the electric basis
    return 1 / (1 - k) * np.log2(np.sum(np.abs(state) ** (2 * k)))


###########################################
# ENSEMBLES
###########################################


def ensemble_energy_window(Hamiltonian, energy, energy_width, max_ensemble_size):
    diag = np.diag(Hamiltonian)
    win1, win2 = energy - 0.5 * energy_width, energy + 0.5 * energy_width
    print(f'ensemble energy window: [{win1}, {win2}]', flush=True)
    indices = [i for i, value in enumerate(diag) if win1 < value < win2]
    indices = indices[0:]
    if not indices:
        print('ERROR: NO STATES IN ENERGY WINDOW')
        exit()
    print('states in window:', len(indices), flush=True)
    ensemble_size = min(len(indices), max_ensemble_size)
    print(f'ensemble_size: {ensemble_size}', flush=True)
    ensemble = np.zeros((Hamiltonian.shape[0], ensemble_size), dtype=np.complex128)
    for i in range(ensemble_size):
        ensemble[indices[i], i] = 1

    return ensemble


def ensemble_comp_basis_states(dim):
    return np.eye(dim, dim, dtype=np.complex128)


###########################################
# STUFF FOR M QUANTUM NUMBERS IN EE
###########################################

def Translation(S,r):
    return S[-(3*r):] + S[:-(3*r)]


def Parity(S,r,P):
    if r==1:
        Snew=S[::-1]
        for i in range(0, 3*P, 3):
            Snew[i+1], Snew[i+2] = Snew[i+2], Snew[i+1]
        return Snew[-2:]+Snew[:-2]
    else:
        return S


def TBS(S,r,P):
    if r == 1:
        Snew = S[0:]
        for i in range(0, 3*P, 3):
            Snew[i], Snew[i + 1] = Snew[i + 1], Snew[i]
        return Snew
    else:
        return S


def append_first_two_elements(l):
    return l + l[:2]


def get_blocks_multiplicities_indices_offsets(B, Sym, P, C1, C2):
    pos1 = ((3 * C1) % (3 * P + 1))
    pos2 = ((3 * C2) % (3 * P + 1)) + 2

    B = B.tolist()
    B_total = []
    for i, state in enumerate(B):
        for r in range(Sym[i][0]):
            State1 = Translation(state, r)
            for e in range(Sym[i][1]):
                State2 = Parity(State1, e, P)
                for u in range(Sym[i][2]):
                    B_total.append(append_first_two_elements(TBS(State2, u, P)))
    B_total = np.asarray(B_total)
    A_basis = np.unique(B_total[:, pos1:pos2], axis=0).tolist()
    del B_total
    dim = len(A_basis)
    print('A_basis generated', flush=True)
    print('dimension of RDM: ', dim, flush=True)

    A_basis_dangling_links = np.zeros((len(A_basis), 4))
    for i, substate in enumerate(A_basis):
        A_basis_dangling_links[i, :] = [substate[0], substate[1], substate[-2], substate[-1]]
    A_basis_dangling_links_unique = np.unique(A_basis_dangling_links, axis=0)
    A_basis_block_multiplicity = np.zeros(A_basis_dangling_links_unique.shape[0], dtype='int32')
    for i, substate in enumerate(A_basis_dangling_links_unique):
        A_basis_block_multiplicity[i] = int(
            (2 * substate[0] + 1) * (2 * substate[1] + 1) * (2 * substate[2] + 1) * (2 * substate[3] + 1))

    A_basis_block_indices = []
    A_basis_block_offsets = np.zeros(A_basis_block_multiplicity.shape[0] + 1, dtype='int32')
    counter = 0
    for i in range(A_basis_block_multiplicity.shape[0]):
        dangling_links = A_basis_dangling_links_unique[i]
        for j, substate in enumerate(A_basis):
            if substate[0] == dangling_links[0] and substate[1] == dangling_links[1] and substate[-2] == dangling_links[-2] and substate[-1] == dangling_links[-1]:
                A_basis_block_indices.append(j)
                counter += 1
        A_basis_block_offsets[i + 1] = counter
    A_basis_block_indices = np.asarray(A_basis_block_indices)

    return A_basis_block_multiplicity, A_basis_block_indices, A_basis_block_offsets


def EE_mqn(rho, block_multiplicity, block_indices, block_offsets):
    sum = 0
    for i in range(block_multiplicity.shape[0]):
        idx = block_indices[block_offsets[i]:block_offsets[i + 1]]
        block = rho[np.ix_(idx, idx)] / block_multiplicity[i]
        val = eigvals(block)
        val = val[np.round(val, 15) > 0]
        sum += block_multiplicity[i] * np.abs(-np.sum(val * np.log(val)))
    return sum


def nth_moment_mqn(rho, power, block_multiplicity, block_indices, block_offsets):
    sum = 0
    for i in range(block_multiplicity.shape[0]):
        idx = block_indices[block_offsets[i]:block_offsets[i + 1]]
        block = rho[np.ix_(idx, idx)] / block_multiplicity[i]
        val = eigvals(block)
        val = val[np.round(val, 15) > 0]
        sum += block_multiplicity[i] * np.abs(np.sum(val ** power))
    return sum
