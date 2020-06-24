# Import native packages
import os

# Import pypi packages
import numpy as np

# Import custom packages
from settings import settings

def make_response_files(folder,dim_in,n_obj,n_constr):
    files = []
    for name in ["database","verification"]:
        files.append(os.path.join(folder,f"{name}.csv"))
        make_data_file(files[-1],dim_in,n_obj,n_constr)

    return files

def make_data_file(file,dim_in,n_obj,n_constr):
    """
    Initialize the results file.

    Arguments:
        target: target file path
        dim_in: number of input dimensions

    Todo:
        * column names
    """

    # Set results file
    
    with open(file, "w") as f:
        headers = header_names(dim_in,n_obj,n_constr)
        f.write(str(dim_in))
        f.write("\n")
        f.write(",".join(headers))
        f.write("\n")

def write_results(file,inputs,outputs):
    all_data = np.concatenate((inputs,outputs),1)

    with open(file, 'a') as f:
        np.savetxt(f,all_data,delimiter=",")
    
def load_results(file):
    dim_in = int(np.loadtxt(file,max_rows=1))
    col_names = np.loadtxt(file,skiprows=1,max_rows=1,dtype=str,delimiter=",")
    data = np.loadtxt(file,skiprows=2,delimiter=",")

    return dim_in, col_names, data

def header_names(dim_in,n_obj,n_const):
    headers = []
    for i in range(dim_in):
        headers.append(f"input_{i+1}")
    for i in range(n_obj):
        headers.append(f"objective_{i+1}")
    for i in range(n_const):
        headers.append(f"constraint_{i+1}")

    return headers

def append_verification_to_database(file,verification_file):
    with open(file, 'a') as datafile:
        with open(verification_file, 'r') as verifile:
            read = "\n".join(verifile.read().split("\n")[2:])
            datafile.write(read)
