"""
initialize
while metric_not_met:
    generate new samples
    evaluate new samples
    retrain surrofate
    check metric
"""

from datamod.visual import ss_convergence
from model_class import Model

class make_settings:
    """
    Class that constains the definitions of the analysis.

    Attributes:
        problem: problem name
        evaluator: evaluator type: benchmark or ansys
        sampling: sampling strategy
        resampling: resampling order, linear or geometric
        default_sample_coef: default sample size multiplicator x dim_in
        resampling_param: for linear number of new samples, for geometric multiplication ratio

        surrogate: surrogaate type - ann, rbf, kriging or genn
        validation: validatino type - holdout, rlt, kfold
        validation_param: holdout ratio, for kfold number of folds

        optimization: optimization strategy
        pop_size: population size
        n_offsprings: number of offsprings
        opt_sampling: optimization sampling
        crossover: crossover strategy
        crossover_prob: crossover probability
        crossover_eta: 
        mutation: mutation strategy
        mutation_eta: 
        termination: ter,mination criterion
        termination_val: 
        opt_seed: random seed for the optimizer
        opt_verbose: output during optimization
        constrained: switch for activation of constrains

        show_raw: show raw datapoints
        show_surrogate: show surrogate response
        show_result_des: show optimization results in the design space
        show_result__obj: show optimization results in the objective space
        
    """
    def __init__(self):
        # Data settings
##        self.evaluator = "ansys"
        self.problem = "matlab_peaks"
##        self.problem = "carside"
        self.evaluator = "benchmark"
        self.sampling = "halton"
        self.resampling = "linear"
        self.default_sample_coef = 10
        self.resampling_param = 10
        
        # Surrogate settings
        self.surrogate = "kriging"
        self.validation = "kfold"
        self.validation_param = 5

        # Optimization settings
        self.optimization = "nsga2"
        self.pop_size = 40
        self.n_offsprings = 10
        self.opt_sampling = "real_random"
        self.crossover = "real_sbx"
        self.crossover_prob = 0.9
        self.crossover_eta = 15
        self.mutation = "real_pm"
        self.mutation_eta = 20
        self.termination = "n_gen"
        self.termination_val = 40
        self.opt_seed = None
        self.opt_verbose = True
        self.constrained = False
##        self.cons_min = 0.5

        # Visual
        self.show_raw = 0
        self.show_surrogate = 0
        self.show_result_des = 1
        self.show_result_obj = 1

# Initialize the model
setting = make_settings()
model = Model(setting)

while not model.optimization_converged:
    # Surrogate training loop
    while not model.trained:
        model.sample()
        model.evaluate()
        model.load_results()
        model.train()
        model.sampling_iterations += 1

        if model.sampling_iterations == 10:
            print("Surrogate converged")
            model.trained = True

    ##breakpoint()

    # Plot the sample size convergence
    ss_convergence(model)
    ##compare()

    # Solve the optimiaztion problem
    model.optimize()

    # Verify whether the optimization result agrees with original model
    model.verify()
