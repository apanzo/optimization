# Import native packages
import datetime
import os
import subprocess
from time import sleep

# Import pypi packages
import numpy as np

# Import custom packages
from core.settings import load_json, settings
from datamod import load_problem
from datamod.results import load_results, write_results
from metamod import reload_info

class Evaluator:
    """
    Docstring
    """

    def __init__(self):
        self.save_results = write_results

    def generate_results(self,samples,file,iteration,verify):
        self.iteration = iteration
        response = self.evaluate(samples,verify)
        self.save_results(file,samples,response)

class EvaluatorBenchmark(Evaluator):
    """
    Evaluate a benchmark problem on the given sample and write the results in a results file.

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

    def __init__(self):
        super().__init__()

    def get_info(self):
        self.problem, range_in, dim_in, dim_out, n_constr = load_problem(settings["data"]["problem"])
        self.results = ["F","G"] if n_constr else ["F"]
        
        return range_in, dim_in, dim_out, n_constr

    def evaluate(self,samples,verify):
        # Evaluate
        response_all = self.problem.evaluate(samples,return_values_of=self.results,return_as_dictionary=True) # return values doesn't work - pymoo implementatino problem
        response = np.concatenate([response_all[column] for column in response_all if column in self.results], axis=1)

        return response

class EvaluatorData(Evaluator):
    """
    Docstring
    """

    def __init__(self):
        super().__init__()
        path = os.path.join(settings["root"],"data","external")
        matching_files = [file for file in os.listdir(path) if file.startswith(settings["data"]["problem"])]

        if len(matching_files) == 0:
            raise Exception("Results file not found")
        elif len(matching_files) > 1:
            raise Exception("Ambiguous results file in folder")

        self.source_file = os.path.join(path,matching_files[0])

    def evaluate(self,samples,verify):
        """
        Dosctring
        """
        response = self.outputs[self.idx_start:self.idx_end,:]
        
        return response

    def get_info(self):
        dim_in,names,data = load_results(self.source_file)
        dim_out = len(names) - dim_in
        n_constr = dim_out-len([name for name in names if name.startswith("objective")])
        self.inputs = data[:,:dim_in]
        self.outputs = data[:,dim_in:]
        range_in = np.stack((np.min(self.inputs,0),np.max(self.inputs,0))).T

        return range_in, dim_in, dim_out, n_constr

    def get_samples(self,no_samples,no_new_samples):
        self.idx_start = no_samples
        self.idx_end = no_samples+no_new_samples

        samples = self.inputs[self.idx_start:self.idx_end,:]

        return samples

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

    def evaluate(self,samples,verify):
        """
        trick with iterations
        """
        if verify:
            self.iteration += 1
            
        self.update_journal(samples)
        while not self.can_run_ansys():
            waiting_time = 10
            print(f"Waiting for {waiting_time}s before checking licenses again")
            sleep(10)   # Wait before checking again

        # Try to perform the analysis indefinitely
        while True:
            try:
                self.call_ansys()
            except:
                print("Analysis failed, retrying")
                continue
            break        

        results = self.get_results(verify)

        if verify:
            self.iteration -= 1
        
        return results

    def update_journal(self,samples):
        # Make journal file
        points = str([list(point) for point in samples])

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

        """
        license_status = self.check_licenses()

        free_licenses = [sum([license_status[i] for i in group]) for group in self.valid_licences]

        can_run = [amount >= 1 for amount in free_licenses]
        can_run_peak = [amount >= minimal_amount for amount in free_licenses]

        now = datetime.datetime.now()

        if all(can_run) and not (now.hour in range(8,15)):
            print("Can run")
            status = True
        elif all(can_run_peak):
            print("Can run")
            status = True
        else:
            print("Not enough licences available: " + str(free_licenses))
            status = False
            status = True

        return status

    def get_results(self,verify):
        outputs = settings["data"]["output_names"]
        
        # read  results
        file = os.path.join(self.ansys_project_folder,"results",f"iteration_{self.iteration}.csv")
        # obtain all data
        names = list(np.loadtxt(file,skiprows=1,max_rows=1,dtype=str,delimiter=",")[1:])
        nodim = len(names)
        data = np.loadtxt(file,skiprows=7,delimiter=",",usecols=range(1,nodim+1))
        data = np.atleast_2d(data)

        # select only relevant outputs
        columns_output = [names.index(name) for name in names if name in outputs]
        response = data[:,columns_output]

        # undo verification trick
        if verify:
            os.rename(file,file.replace("iteration_","verification_"))

        return response


class RealoadNotAnEvaluator(Evaluator):
    """
    Just reload

    Notes:
        Just a programming convenience, doesnt evaluate anything in fact
    """

    def __init__(self):
        pass

    def get_info(self):
        range_in, dim_in, dim_out, n_constr = reload_info()
        
        return range_in, dim_in, dim_out, n_constr
