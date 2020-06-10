import json

def check_json(file):
    with open(file+".json") as f:
         data = json.load(f)

    return data

data = check_json("settings")

##class Test:
##    
##    def __init__(self):
##        self.options = [1,2,3]
##
##    def __getitem__(self,a):
##        return self.options[a]

