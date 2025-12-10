import module
import os
import h5py

path = module.get_path()


def printname(name):
    print(name)


if __name__ == "__main__":
    print("path:",path)
    with h5py.File(path, "r") as f:
        f.visit(printname)
        f.close()
