"""
This module contains functions to handle the results database.
"""

# Import native packages
import os

# Import pypi packages
import numpy as np

# Import custom packages
from core.settings import settings

def make_response_files(folder,dim_in,n_obj,n_constr):
    """
    Sets up the training and verification database files.

    Args:
        folder (str): Path to the current results data folder.
        dim_in (int): Number of input dimensions.
        n_const (int): Number of constraints.
        n_obj (int): Number of objectives.

    Returns:
        files (list): List of the database files paths.
    """
    files = []
    for name in ["database","verification"]:
        files.append(os.path.join(folder,f"{name}.csv"))
        make_data_file(files[-1],dim_in,n_obj,n_constr)

    return files

def make_data_file(file,dim_in,n_obj,n_constr):
    """
    Initialize the results file header.

    Args:
        file (str): Path and name of the database file.
        dim_in (int): Number of input dimensions.
        n_const (int): Number of constraints.
        n_obj (int): Number of objectives.

    Todo:
        Write correct column names.
    """

    # Set results file
    
    with open(file, "w") as f:
        headers = header_names(dim_in,n_obj,n_constr)
        f.write(str(dim_in))
        f.write("\n")
        f.write(",".join(headers))
        f.write("\n")

def write_results(file,inputs,outputs):
    """
    Write the samples to the database.
    
    Args:
        inputs (np.array): Input coordinates.
        outputs (np.array): Output coordinates.
    """
    all_data = np.concatenate((inputs,outputs),1)

    with open(file, 'a') as f:
        np.savetxt(f,all_data,delimiter=",")
    
def load_results(file):
    """
    Loads the data from the result database.
    
    Args:
        file (str): Path and name of the database file.

    Returns:
        dim_in (int): Number of input dimensions.
        col_names (np.array): Name if stored variables.        
        data (np.array): Samples.        
    """
    dim_in = int(np.loadtxt(file,max_rows=1))
    col_names = np.loadtxt(file,skiprows=1,max_rows=1,dtype=str,delimiter=",")
    data = np.loadtxt(file,skiprows=2,delimiter=",")

    return dim_in, col_names, data

def header_names(dim_in,n_obj,n_const):
    """
    Determines the header names.
    
    Args:
        dim_in (int): Number of input dimensions.
        n_const (int): Number of constraints.
        n_obj (int): Number of objectives.
    
    Returns:
        headers (list): Header names.
    """
    headers = []
    for i in range(dim_in):
        headers.append(f"input_{i+1}")
    for i in range(n_obj):
        headers.append(f"objective_{i+1}")
    for i in range(n_const):
        headers.append(f"constraint_{i+1}")

    return headers

def append_verification_to_database(file,verification_file):
    """
    Appends the results from the verification to the training database.

    Args:
        file (str): Path and name of the training database file.
        verification_file (str): Path and name of the verification database file.
    """
    with open(file, 'a') as datafile:
        with open(verification_file, 'r') as verifile:
            read = "\n".join(verifile.read().split("\n")[2:])
            datafile.write(read)