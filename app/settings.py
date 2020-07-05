# Import native packages
import json
import os
from pathlib import Path
import pickle
from shutil import rmtree, copyfile

def load_json(file):
    """
    Read a JSON file.
    """
    with open(file+".json") as f:
         data = json.load(f)

    return data

def get_input_from_id(problem_id,problem_folder):
    """
    Get filename from problem ID.
    """
    matching_ids = [name for name in os.listdir(problem_folder) if name.startswith(str(problem_id).zfill(2))]

    if len(matching_ids) == 1:
        file_name = matching_ids[0].replace(".json","")
    elif len(matching_ids) == 0:
        raise Exception(f"ID {problem_id} input undefined")
    else:
        raise Exception(f"ID {problem_id} input multiple defined")

    if not (file_name[:2].isdigit() and file_name[2] is "-"):
        raise Exception('Invalid input file name, should start with "XX-" where XX is the problem ID')

    file = os.path.join(problem_folder,file_name)

    return file

def check_valid_settings():
    """
    Check if valid setting are used.
    """
    path = os.path.join(settings["root"],"app","config","inputs","valid_settings")
    valid = load_json(path)
    for i in valid.keys():
        for j in valid[i].keys():
            if not (settings[i][j] in valid[i][j]):
                breakpoint()
                raise Exception(f"Invalid setting for {j}, valid keys are: [{', '.join(valid[i][j])}]")

def update_settings(problem_id):
    if not len(settings.keys()) == 1:
        raise Exception("Should only apply method on empty settings")

    # Get filename
    problem_folder = os.path.join(settings["root"],"app","inputs")
    file = get_input_from_id(problem_id,problem_folder)

    # Update
    update_from_json = load_json(file)
    settings.update(update_from_json)

    # Check valid inputs
    check_valid_settings()

    # Add ID
    settings["id"] = problem_id
    check_already_present(settings["id"])

    copyfile(file+".json",os.path.join(settings["folder"],"input.json"))

def check_already_present(id_current):
    # List defined problems
    data_folder, all_result_folders = get_results_folders()

    matching_folders = [folder for folder in all_result_folders if folder.startswith(str(id_current).zfill(2))]

    # If ID has already results, decide whether to overwrite
    if len(matching_folders) > 0:
        while True:
##            overwrite = input(f"ID {id_current} has already results. Do you want to overwrite results? [y/n]")
            overwrite = "y"
            if overwrite in ["y","n"]:
                if overwrite == "y":
                    for folder in matching_folders:
                        rmtree(os.path.join(data_folder,folder))
                    break
                else:
                    raise Exception(f"ID {id_current} already defined")
            else:
                print("Invalid input")

    # Make workfolder
    settings["folder"] = make_workfolder()

def get_results_folders():
    """
    Docstring
    """
    data_folder = os.path.join(settings["root"],"data")
    all_result_folders = [name for name in next(os.walk(os.path.join(settings["root"],"data")))[1]]
    
    return data_folder, all_result_folders

def make_workfolder():
    """
    Initialize the workdirectory.

    """
    # Setup the folder path
    folder_name = str(settings["id"]).zfill(2) + "-" +  settings["data"]["problem"]
    folder_path = os.path.join(settings["root"],"data",folder_name)
    figures_path = os.path.join(folder_path,"figures")
    logs_path = os.path.join(folder_path,"logs")

    # Create folder, if not done yet
    Path(folder_path).mkdir(parents=True,exist_ok=True) # parents in fact not needed   
    Path(figures_path).mkdir(parents=True,exist_ok=True) # parents in fact not needed  
    Path(logs_path).mkdir(parents=True,exist_ok=True) # parents in fact not needed    

    return folder_path

def dump_object(name,*args):
    file = os.path.join(settings["folder"],"logs",f"{name}_dump.pkl")
    # Saving the objects:
    with open(file, "wb") as f: 
        pickle.dump(args, f)

def load_object(name):
    file = os.path.join(settings["folder"],"logs",f"{name}_dump.pkl")
    # Getting back the objects:
    with open(file, "rb") as f:
        obj = pickle.load(f)

    return obj

# Initialize setting with the path to the root folder
settings = {"root": os.path.split(os.path.split(__file__)[0])[0]}

