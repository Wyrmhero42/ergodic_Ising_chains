import module
import os
import h5py

path = module.get_path()
if not os.path.isfile(path):
    with h5py.File(path, 'a') as hdf:
        hdf.create_group('periodic/states')
        hdf.create_group('periodic/eigenvalues')
        hdf.create_group('periodic/eigenvectors')
        hdf.create_group('periodic/plaquette_term')

        hdf.create_group('open/states')
        hdf.create_group('open/eigenvalues')
        hdf.create_group('open/eigenvectors')
        hdf.create_group('open/plaquette_term')
        
        hdf.create_group('Ising_XZ_trans/Hamiltonians')
        hdf.create_group('Ising_XZ_trans/eigenvalues')
        hdf.create_group('Ising_XZ_trans/eigenvectors')

        hdf.create_group('Ising_tilted/Hamiltonians')
        hdf.create_group('Ising_tilted/eigenvalues')
        hdf.create_group('Ising_tilted/eigenvectors')
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

