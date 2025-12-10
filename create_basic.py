import module
import os
import h5py

path = module.get_path()
if not os.path.isfile(path):
    with h5py.File(path, 'a') as hdf:
        hdf.create_group('Periodic/States')
        hdf.create_group('Periodic/Eigenvalues')
        hdf.create_group('Periodic/Eigenvectors')
        hdf.create_group('Periodic/Plaquette_term')
        
        hdf.create_group('Open/States')
        hdf.create_group('Open/Eigenvalues')
        hdf.create_group('Open/Eigenvectors')
        hdf.create_group('Open/Plaquette_term')
        
        hdf.create_group('Ising_XZ_trans/Hamiltonians')
        hdf.create_group('Ising_XZ_trans/Eigenvalues')
        hdf.create_group('Ising_XZ_trans/Eigenvectors')
        
        hdf.create_group('Ising_tilted/Hamiltonians')
        hdf.create_group('Ising_tilted/Eigenvalues')
        hdf.create_group('Ising_tilted/Eigenvectors')
        hdf.close()

base_dir = "output"
subdirs = ["periodic", "aperiodic"]
subsubdirs = ["plots", "rdm_spectrum_analysis", "ensemble_evolution"]
for subdir in subdirs:
    for subsubdir in subsubdirs:
        path = os.path.join(base_dir, subdir, subsubdir)
        os.makedirs(path, exist_ok=True)
        
subdirs = ["Ising_XZ_trans", "Ising_tilted"]
subsubdirs = ["plots", "eigenvalues"]
for subdir in subdirs:
    for subsubdir in subsubdirs:
        path = os.path.join(base_dir, subdir, subsubdir)
        os.makedirs(path, exist_ok=True)
print("Directory structure created (if it didn't already exist).")

