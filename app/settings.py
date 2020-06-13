# Import native packages
import os
import json

def load_json(file):
    with open(file+".json") as f:
         data = json.load(f)

    return data

def load_settings(folder,file,valid_check=0):
    if not len(settings.keys()) == 1 and not valid_check:
        raise Exception("Should only apply on empty list")
        
    update = load_json(os.path.join(settings["root"],folder,"inputs",file))

    return update

def check_valid_settings():
    valid = load_settings(os.path.join("app","config"),"valid_settings",1)
    for i in valid.keys():
        for j in valid[i].keys():
            if not (settings[i][j] in valid[i][j]):
                raise Exception("Invalid setting for " + "-".join([i,j]) + ", valid keys are: [" + ", ".join(valid[i][j])+"]")
            
settings = {"root": os.path.split(os.path.split(__file__)[0])[0]}

