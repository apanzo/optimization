# Import native packages
import os
import json

def load_json(file):
    with open(file+".json") as f:
         data = json.load(f)

    return data

def load_settings(folder,file):
    if not len(settings.keys()) == 1:
        raise Exception("Should only apply on empty list")
        
    update = load_json(os.path.join(settings["root"],folder,"inputs",file))

    return update

settings = {"root": os.path.split(os.path.split(__file__)[0])[0]}
