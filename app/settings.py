# Import native packages
import os
import json

def load_json(file):
    with open(file+".json") as f:
         data = json.load(f)

    return data

settings = load_json("settings")
settings["root"] = root_path = os.path.split(os.getcwd())[0]
