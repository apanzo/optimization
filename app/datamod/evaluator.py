# Import native packages
import datetime
import os
import subprocess

# Import pypi packages
import numpy as np

# Import custom packages
from datamod.results import write_results
from settings import load_json, settings

class Evaluator:
    """
    Docstring
    """

    def __init__(self):
        self.save_results = write_results

    def generate_results(self,samples,file):
        response = self.evaluate(samples)
        self.save_results(file,samples,response)

class EvaluatorBenchmark(Evaluator):
    """
    Evaluate a benchmark problem on the given sample and write the results in a results finle.

    Arguments:
        problem: problem object
        samples: coordinates of new samples
        file: target results file
        n_constr: number of constrains

    Todo:
        * write performance metrics log

    Notes:
        * return values is wrongly implemented in pymoo
    """

    def __init__(self,problem,n_constr):
        super().__init__()
        self.results = ["F","G"] if n_constr else ["F"]
        self.problem = problem

    def evaluate(self,samples):
        # Evaluate
        response_all = self.problem.evaluate(samples,return_values_of=self.results,return_as_dictionary=True) # return values doesn't work - pymoo implementatino problem
        response = np.concatenate([response_all[column] for column in response_all if column in self.results], axis=1)

        return response

class EvaluatorANSYS(Evaluator):
    """
    Docstring
    """

    def __init__(self):
        super().__init__()
        
        self.ansys_project_folder = settings["data"]["project_folder"]
        self.workbench_project = settings["data"]["problem"]+".wbpj"
        self.template = os.path.join(settings["root"],"app","config","dataconf","ansys_journal_template.wbjn")
        self.output = os.path.join(settings["folder"],"journal.wbjn")
        self.input_param_name = settings["data"]["input_names"]

        self.setup = load_json(os.path.join(settings["root"],"app","config","dataconf","ansys"))
        self.valid_licences = [name for name in self.setup["licenses"].values()]

    def get_info(self):
        dim_in, dim_out, n_constr = settings["data"]["dims"]
        range_in = np.array((settings["data"]["lower_limits"],settings["data"]["upper_limits"])).T

        if not all((range_in[:,0] - range_in[:,1]) < 0):
            raise Exception("Invalid bounds specified, upper bound must be higher than lower")

        return range_in, dim_in, dim_out, n_constr

    def evaluate(self,samples):
        self.update_journal(samples)
        self.can_run_ansys()
        self.call_ansys()
        
        return self.get_results()

    def update_journal(self,samples):
        # Make journal file
        points = str([list(point) for point in samples])
        self.iteration = 8 ###########################################

        # Open template
        with open(self.template, "r") as file:
            text = file.readlines()

        # Modify template
        temp_str = "project_folder = '" + self.ansys_project_folder + "\\" + "'\n"
        text[8] = temp_str.replace("\\","\\\\")
        text[9] = "workbench_project = '" + self.workbench_project + "'\n"
        text[10] = "points = " + str(points) + "\n"
        text[11] = "input_param_name = " + str(self.input_param_name) + "\n"
        text[12] = "iteration = " + str(self.iteration) + "\n"

        # Write modified template
        with open(self.output, "w") as file:
            file.write("".join(text))
            
    def scrape_license_info(self,line):
        license_name = line.split()[2][:-1]
        issued, used = [int(s) for s in line.split() if s.isdigit()]
        available = issued-used
        
        return license_name, available

    def check_licenses(self):
        directory = self.setup["license_checker"]
        file = "lmutil"
        arguments = ["lmstat","-c",self.setup["license_server"],"-f"]

        call = subprocess.run([file]+arguments,cwd=directory,shell=True,capture_output=True,text=True)

        output = call.stdout.split("\n")

        license_status = dict([self.scrape_license_info(line) for line in output if  line.startswith("Users of")])

        return license_status

    def call_ansys(self):
        directory = self.setup["ansys_directory"]
        file = "runwb2"
        arguments = ["-B","-R"]

        call = subprocess.run([file]+arguments+[self.output],cwd=directory,shell=True,capture_output=True,text=True)

        if call.returncode:
            raise Exception("Analysis failed")

    def can_run_ansys(self,minimal_amount = 2):
        """
        Docstring

        Notes:
            * arg not used
        """
        license_status = self.check_licenses()

        free_licenses = [sum([license_status[i] for i in group]) for group in self.valid_licences]

        can_run = [amount >= 1 for amount in free_licenses]
        can_run_peak = [amount >= minimal_amount for amount in free_licenses]

        now = datetime.datetime.now()

        if all(can_run) and not (now.hour in range(8,15)):
            print("Can run")
        elif all(can_run_peak):
            print("Can run")
        else:
            print("Not enough licences available: " + str(free_licenses))

    def get_results(self):
        # read  results
        file = os.path.join(self.ansys_project_folder,"results",f"iteration_{self.iteration}.csv")

        names = list(np.loadtxt(file,skiprows=1,max_rows=1,dtype=str,delimiter=",")[1:])
        nodim = len(names)
        columns = [names.index(name) for name in names if not name in settings["data"]["input_names"]]
        data = np.loadtxt(file,skiprows=7,delimiter=",",usecols=range(1,nodim+1))

        response = data[:,columns]

        return response
