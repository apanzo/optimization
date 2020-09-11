"""
Contains the classes to evaluate the new samples, typically using an external software.
"""

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
    General evaluator class.
    
    Attributes:
        save_results (): Function to write the results into the results database.
        iteration (int): Iteration number.
    """

    def __init__(self):
        self.save_results = write_results

    def generate_results(self,samples,file,iteration,verify):
        """
        Generate the response and save the result to the database.
        
        Args:
            samples (np.array): Samples to evaluate.
            file (str): Path and name of the database file.
            iteration (int): Iteration number.
            verify (bool): Whether this is a verification evaluation.
        """
        self.iteration = iteration
        response = self.evaluate(samples,verify)
        self.save_results(file,samples,response)

class EvaluatorBenchmark(Evaluator):
    """
    Evaluate a benchmark problem on the given sample.

    Attributes:
        problem (): Benchmark problem.
        results (list): List of values requested from the problem.
    """

    def __init__(self):
        super().__init__()

    def get_info(self):
        """
        Get information about the benchmark problem.
        
        Returns:
            range_in (np.array): Input parameter allowable ranges.
            dim_in (int): Number of input dimensions.
            dim_out (int): Number of output dimensions.
            n_constr (int): Number of constraints.        

        Notes:
            return_values_of is wrongly implemented in pymoo        
        """
        self.problem, range_in, dim_in, dim_out, n_constr = load_problem(settings["data"]["problem"])
        self.results = ["F","G"] if n_constr else ["F"]
        
        return range_in, dim_in, dim_out, n_constr

    def evaluate(self,samples,verify):
        """
        Evaluate the samples.
        
        Args:
            samples (np.array): Samples to evaluate.
            verify (bool): Whether this is a verification evaluation.
            
        Returns:
            response (np.array): Output values at the samples.

        Warnings:
            Return_values_of in problem.evaluate doesn't work - Pymoo implementation problem.
        """
        # Evaluate
        response_all = self.problem.evaluate(samples,return_values_of=self.results,return_as_dictionary=True) # return_values_of doesn't work as expected
        response = np.concatenate([response_all[column] for column in response_all if column in self.results], axis=1)

        return response

class EvaluatorData(Evaluator):
    """
    Obtain the response from a data file.
    
    Attributes:
        source_file (str): Path and name of the data file.
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
        Evaluate the samples.
        
        Args:
            samples (np.array): Samples to evaluate.
            verify (bool): Whether this is a verification evaluation.            
        
        Returns:
            response (np.array): Output values at the samples.
        """
        response = self.outputs[self.idx_start:self.idx_end,:]
        
        return response

    def get_info(self):
        """
        Load information about a data-defined problem.
        
        Returns:
            range_in (np.array): Input parameter allowable ranges.
            dim_in (int): Number of input dimensions.
            dim_out (int): Number of output dimensions.
            n_constr (int): Number of constraints.        
        """
        dim_in,names,data = load_results(self.source_file)
        dim_out = len(names) - dim_in
        n_constr = dim_out-len([name for name in names if name.startswith("objective")])
        self.inputs = data[:,:dim_in]
        self.outputs = data[:,dim_in:]
        range_in = np.stack((np.min(self.inputs,0),np.max(self.inputs,0))).T

        return range_in, dim_in, dim_out, n_constr

    def get_samples(self,no_samples,no_new_samples):
        """
        Get the coordinates of the new samples.

        Returns:
            samples(np.array): Coordinates of the new samples.
        """
        self.idx_start = no_samples
        self.idx_end = no_samples+no_new_samples

        samples = self.inputs[self.idx_start:self.idx_end,:]

        return samples

class EvaluatorANSYS(Evaluator):
    """
    Evaluate the samples using ANSYS.
    
    Attributes:
        ansys_project_folder (str): Path to the folder of the ANSYS project.
        input_param_name (list): Names of the input parameters.
        setup (dict): ANSYS settings.
        valid_licences (list): Licences required for running ANSYS:
    """

    def __init__(self):
        super().__init__()
        
        self.ansys_project_folder = settings["data"]["project_folder"]
        self.input_param_name = settings["data"]["input_names"]

        self.setup = load_json(os.path.join(settings["root"],"app","config","dataconf","ansys"))
        self.valid_licences = [name for name in self.setup["licenses"].values()]

    def get_info(self):
        """
        Get information about the problem.
        
        Returns:
            range_in (np.array): Input parameter allowable ranges.
            dim_in (int): Number of input dimensions.
            dim_out (int): Number of output dimensions.
            n_constr (int): Number of constraints.        
        """
        dim_in, dim_out, n_constr = settings["data"]["dims"]
        range_in = np.array((settings["data"]["lower_limits"],settings["data"]["upper_limits"])).T

        if not all((range_in[:,0] - range_in[:,1]) < 0):
            raise Exception("Invalid bounds specified, upper bound must be higher than lower")

        return range_in, dim_in, dim_out, n_constr

    def evaluate(self,samples,verify):
        """
        Evaluate the samples.

        Args:
            samples (np.array): Samples to evaluate.
            verify (bool): Whether this is a verification evaluation.

        Returns:
            results (np.array): Output values at the samples.
        """
        self.update_input(samples)
        while not self.can_run_ansys():
            waiting_time = 10
            print(f"Waiting for {waiting_time}s before checking licenses again")
            sleep(waiting_time)

        self.call_ansys()

        results = self.get_results(verify)
        
        return results
            
    def scrape_license_info(self,line):
        """
        Extract the number of available ANSYS licenses.
        
        Args:
            line (str): Line containing the license usage information.
        
        Returns:
            license_name (str): Name of the license.
            available (int): Number of available unused licenses.

        """
        license_name = line.split()[2][:-1]
        issued, used = [int(s) for s in line.split() if s.isdigit()]
        available = issued-used
        
        return license_name, available

    def check_licenses(self):
        """
        Request the license server for infomation about license usage.
        
        Returns:
            license_status (dict): Licence names with the number of unused licenses.
        """
        directory = self.setup["license_checker"]
        file = "lmutil"
        arguments = ["lmstat","-c",self.setup["license_server"],"-f"]

        call = subprocess.run([file]+arguments,cwd=directory,shell=True,capture_output=True,text=True)

        output = call.stdout.split("\n")

        license_status = dict([self.scrape_license_info(line) for line in output if line.startswith("Users of")])

        return license_status

    def can_run_ansys(self,minimal_amount = 2):
        """
        Determine whether there is a sufficient amount of available licenses to run the simulation.
        
        Args:
            minimal_amount (int): Minimal required amount of available licenses.
            
        Returns:
            status (bool): Whether it is possible to run ANSYS or not.
            
        Notes:
            Returns True whenever at least one license is available.
        """
        license_status = self.check_licenses()

        free_licenses = [sum([license_status[i] for i in group]) for group in self.valid_licences]

        can_run = [amount >= 1 for amount in free_licenses]
        can_run_peak = [amount >= minimal_amount for amount in free_licenses]

        now = datetime.datetime.now()
        
        if not all(free_licenses):
            print("No licences available")
            status = False
        elif all(can_run) and not (now.hour in range(8,15)):
            print("Can run")
            status = True
        elif all(can_run_peak):
            print("Can run")
            status = True
        else:
            print("Not enough licences available: " + str(free_licenses))
            status = False
            status = True ##########

        return status

class EvaluatorANSYSAPDL(EvaluatorANSYS):
    """
    Evaluate the samples through ANSYS APDL.
    
    Notes:
        Not documented thorougly as it is a dev version for the particular problem.
    """

    def __init__(self):
        super().__init__()

        self.templates = [i for i in os.listdir(self.ansys_project_folder) if "_template" in i]

        self.program = "C:\\Program Files\\ANSYS Inc\\v194\\ansys\\bin\\winx64\\MAPDL.exe"
        self.input_file_x = os.path.join(self.ansys_project_folder,"ana_x.dat")
        self.input_file_y = os.path.join(self.ansys_project_folder,"ana_y.dat")
        self.out_directory = "C:\\Users\\antonin.panzo\\Downloads\\ANSYS\\APDL Work"
        self.output_file_x = os.path.join(self.out_directory,"file_x.out")
        self.output_file_y = os.path.join(self.out_directory,"file_y.out")

        self.command_x = f'"{self.program}"  -p ansys -dis -mpi INTELMPI -np 1 -lch -dir "{self.out_directory}" -j "input" -s read -l en-us -b nolist -i "{self.input_file_x}" -o "{self.output_file_x}"   '
        self.command_y = f'"{self.program}"  -p ansys -dis -mpi INTELMPI -np 1 -lch -dir "{self.out_directory}" -j "input" -s read -l en-us -b nolist -i "{self.input_file_y}" -o "{self.output_file_y}"   '

    def evaluate(self,samples,verify):
        """
        Evaluate the samples.

        Args:
            samples (np.array): Samples to evaluate.
            verify (bool): Whether this is a verification evaluation.
                    
        Returns:
            response (np.array): Output values at the samples.
        """

        results = []
        progress = 1
        for sample in samples:
            print(f"Solving {progress}/{samples.shape[0]}")
            result = super().evaluate(sample,verify)
            results.append(result)
            progress += 1
            
        return np.array(results)

    def update_input(self,samples):
        """
        Update the unput files.
        
        Args:
            samples (np.array): Input samples to evaluate.
        """
        replace = [f"*SET,T{i+1},{samples[i]:.6f}\n" for i in range(len(samples))]

        for file in self.templates:
            path = os.path.join(self.ansys_project_folder,file)
            with open(path,"r") as file:
                lines = file.readlines()
            ind = lines.index('/com,*********** Send Beam Properties ***********\n')
            lines[ind+2:ind+5] = replace
            with open(path.replace("_template",""),"w") as file:
                file.writelines(lines)

    def call_ansys(self):
        """
        Evaluate the requested input files.
        """
        subprocess.run(self.command_x)
        subprocess.run(self.command_y)

    def get_results(self,verify):
        """
        Retrieve the results from text files.
        
        Args:
            verify (bool): Whether this is a verification evaluation.

        Returns:
            response (np.array): Output values at the samples.
        """
        out = []
        for i in range(3):
            path = os.path.join(self.out_directory,f"out{i+1}.txt")
            out.append(np.loadtxt(path))
        response = np.array(out)

        return response

class EvaluatorANSYSWB(EvaluatorANSYS):
    """
    Evaluate the samples through ANSYS Workbench.
    
    Attributes:
        workbench_project (str): Path and name to the ANSYS workbench project.
        template (str): Path and name to the journal file template.
        output (str): Path and name to the newly created journal folder.
        iteration (int): Iteration number.
    """
    def __init__(self):
        super().__init__()
        
        self.workbench_project = settings["data"]["problem"]+".wbpj"
        self.template = os.path.join(settings["root"],"app","config","dataconf","ansys_journal_template.wbjn")
        self.output = os.path.join(settings["folder"],"journal.wbjn")

    def evaluate(self,samples,verify):
        """
        Evaluate the samples.
        
        Args:
            samples (np.array): Samples to evaluate.
            verify (bool): Whether this is a verification evaluation.
            
        Returns:
            results (np.array): Output values at the samples.

        Notes:
            A trick with iterations
        """
        if verify:
            self.iteration += 1

        results = super().evaluate(samples,verify)
            
        if verify:
            self.iteration -= 1
        
        return results

    def update_input(self,samples):
        """
        Create an input journal from the template.

        Args:
            samples (np.array): Samples to evaluate.
        """
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

    def call_ansys(self):
        """
        Evaluate the requested input files.
        """
        directory = self.setup["ansys_directory"]
        file = "runwb2"
        arguments = ["-B","-R"]

        while True:
            call = subprocess.run([file]+arguments+[self.output],cwd=directory,shell=True,capture_output=True,text=True)
            if call.returncode:
                print("Analysis failed, retrying")
            else:
                break

    def get_results(self,verify):
        """
        Retrieve the results from CSV files.

        Args:
            verify (bool): Whether this is a verification evaluation.
        
        Returns:
            response (np.array): Output values at the samples.
        """
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
    Just a programming convenience when reloading a surrogate, doesnt evaluate anything in fact.
    """

    def __init__(self):
        pass

    def get_info(self):
        """
        Get information about the problem.

        Returns:
            range_in (np.array): Input parameter allowable ranges.
            dim_in (int): Number of input dimensions.
            dim_out (int): Number of output dimensions.
            n_constr (int): Number of constraints.
        """
        range_in, dim_in, dim_out, n_constr = reload_info()
        
        return range_in, dim_in, dim_out, n_constr
