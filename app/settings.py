# Import native packages
import os
import json

def load_json(file):
    with open(file+".json") as f:
         data = json.load(f)

    return data

root_path = os.path.split(os.path.split(__file__)[0])[0]
settings = load_json(os.path.join(root_path,"app","settings"))
settings["root"] = root_path
