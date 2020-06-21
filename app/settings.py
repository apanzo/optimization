# Import native packages
import os
import json

def load_json(file):
    with open(file+".json") as f:
         data = json.load(f)

    return data

def load_settings(folder,problem_id,valid_check=0):
    problem_folder = os.path.join(settings["root"],folder,"inputs")

    file = get_input_from_id(problem_id,problem_folder)
            
    if not len(settings.keys()) == 1 and not valid_check:
        raise Exception("Should only apply method on empty list")
        
    update = load_json(os.path.join(problem_folder,file))

    if not valid_check:
        update["data"]["id"] = int(file[:2])
    
    return update


def get_input_from_id(problem_id,problem_folder):
    matching_ids = [name for name in os.listdir(problem_folder) if name.startswith(str(problem_id).zfill(2))]
    if len(matching_ids) == 1:
        file = matching_ids[0].replace(".json","")
    elif len(matching_ids) == 0:
        raise Exception(f"ID {problem_id} input undefined")
    else:
        raise Exception(f"ID {problem_id} input multiple defined")

    if not (file[:2].isdigit() and file[2] is "-"):
        raise Exception('Invalid input file name, should start with "XX-" where XX is the problem ID')

    return file

def check_valid_settings():
    valid = load_settings(os.path.join("app","config"),0,1)
    for i in valid.keys():
        for j in valid[i].keys():
            if not (settings[i][j] in valid[i][j]):
                raise Exception("Invalid setting for " + "-".join([i,j]) + ", valid keys are: [" + ", ".join(valid[i][j])+"]")
            
settings = {"root": os.path.split(os.path.split(__file__)[0])[0]}

