"""
Custom ANN definition using TensorFlow.

This module contains the definition of an ANN comptable
with the SMT Toolbox
"""
# Import native packages
import os

# Import pypi packages
from kerastuner.tuners import RandomSearch, BayesianOptimization
from kerastuner.engine.hyperparameters import HyperParameters
import matplotlib.pyplot as plt
import numpy as np
from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils.checks import check_2d_array
import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity

# Import custom packages
from settings import load_json, settings
        
class ANN(SurrogateModel):
    """
    ANN class.

    See also:
        Tensorflow documentation
        
    To do:
        * check if pretraining
        * if pretrainning:
            * select hyperparameters to optimize, keep rest default
            * take defaults from settings
            * select optimization method
            * perform pretraining
            * select best model
            * save best model
        * else:
            * load hyperparameters from optimization
    """

    def __getitem__(self,a):
        return self.options[a]

    def __init__(self,**kwargs):
        self.tbd = kwargs.keys()
        super().__init__(**kwargs)

        # Initialize model
        self.name = "ANN"
        self.log_dir = os.path.join(settings["folder"],"logs","ann")

        # Declare architecture
##        in_dim, out_dim = self["dims"]
##        self.options.declare("architecture", [in_dim]+[self["no_neurons"]]*self["no_layers"]+[out_dim], types=list, desc="default architecture")
##        self.model = self.build_model()

        self.optimized = False
        self._is_trained = False
        
    def _initialize(self):
        """
        Initialize the default hyperparameter settings.

        Parameters:
            no_layers: number of layers
            no_neurons: number of neurons per layer
            activation: activation functoin
            batch_size: batch size
            no_epochs: nu,ber of training epochs
            init: weight initialization strategy
            bias_init:  bias initialization strategy
            optimizer: optiimzer type
            loss: loss function
            kernel_regularizer: regularization paremeres
            dims: number of input and output dimension
            no_points: number of sample points
            prune: whether to use pruning
            sparsity: target network sparsity (fraction of zero weights)
            pruning_frequency: frequency of pruning
            tensorboard: whether to make tensorboard output
            stopping: use early stopping
            stopping_delta: required error delta threshold to stop training
            stopping_patience: number of iterations to wait before stopping
            plot_history: whether to plot the training history
        """
##        setup = self.configurations
##        # Set default values
        declare = self.options.declare
        for param in self.tbd:
            declare(param, None)

        pass
    
    def build_hypermodel(self,hp):
        # Initiallze model
        model = tf.keras.Sequential()

        # Initialize hyperparameters        
        neurons_hyp = hp.Fixed("no_neurons",self["no_neurons"])
        layers_hyp = hp.Fixed("no_hid_layers",self["no_layers"])
        lr_hyp = hp.Fixed("learning_rate",self["learning_rate"])
        activation_hyp = hp.Fixed("activation_function", self["activation"])
        regularization_hyp = hp.Fixed("regularization",self["kernel_regularizer"])
        sparsity_hyp = hp.Fixed("sparsity",self["sparsity"])
        # 
        kernel_regularizer = tf.keras.regularizers.l1(regularization_hyp)
        
        # Add layers        
        in_dim, out_dim = self["dims"]
        model.add(tf.keras.layers.Dense(neurons_hyp, activation=activation_hyp,
                                             kernel_initializer=self["init"],
                                             bias_initializer=self["bias_init"],
                                             kernel_regularizer=kernel_regularizer,
                                             input_shape=(in_dim,)))
        for _ in range(layers_hyp-1):
            model.add(tf.keras.layers.Dense(neurons_hyp, activation=activation_hyp,
                                                 kernel_initializer=self["init"],
                                                 bias_initializer=self["bias_init"],
                                                 kernel_regularizer=kernel_regularizer))
        model.add(tf.keras.layers.Dense(out_dim,activation="linear",use_bias=False))

        # Train
        optimizer = tf.keras.optimizers.get(self["optimizer"])
        optimizer._hyper["learning_rate"] = lr_hyp

        # Prune the model
        if self["prune"]:
            model = self.prune_model(model,sparsity_hyp)

        model.compile(optimizer,self["loss"],metrics=["mse","mae","mape"])

        return model

    def prune_model(self,model,target_sparsity):
##        end_step = np.ceil(self["no_points"]/self["batch_size"]*self["no_epochs"]
##        print(f"End step: {end_step}")
##        model = sparsity.prune_low_magnitude(model, sparsity.PolynomialSparsity(0,target_sparsity, 0, end_step, frequency=10)        )
        model = sparsity.prune_low_magnitude(model, sparsity.ConstantSparsity(target_sparsity, 0, frequency=10)        )
        return model

    def get_callbacks(self):
        callbacks = []

        if self["prune"]:
            callbacks.append(sparsity.UpdatePruningStep())
        if self["tensorboard"]:
            raise ValueError("Tensorboard not implemented yet")
##            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.log_dir,"tensorboard"), histogram_freq=20))
        if self["stopping"]:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=self["stopping_delta"], patience=self["stopping_patience"], verbose=1,restore_best_weights=True))
##            callbacks.append(tf.keras.callbacks.MyStopping(monitor='val_loss', target=5, patience=stopping_patience, verbose=1,restore_best_weights=True))

        return callbacks

    def pretrain(self):
        # Select hyperparameters to tune
        hp = HyperParameters()
##        hp.Int("no_neurons",6,40,step=2,default=self["no_neurons"])
##        hp.Int("no_hid_layers",1,3,default=self["no_layers"])
##        hp.Float("learning_rate",0.001,0.1,sampling="log",default=self["learning_rate"])
##        hp.Fixed("activation_function","tanh")
##        hp.Choice("activation_function",["sigmoid","relu","swish"],default=self["activation"])
##        hp.Float("regularization",0.001,1,sampling="log",default=self["kernel_regularizer"])
##        hp.Float("sparsity",0.3,0.9,default=self["sparsity"])

        # Load tuner
        max_trials = self["max_trials"]
        max_trials = 1
        path_tf_format = "logs"
##        tuner = BayesianOptimization(self.build_hypermodel,objective="val_mae",hyperparameters=hp,
##                             max_trials=max_trials,executions_per_trial=1,directory="logs",num_initial_points=10,
##                                     overwrite=True,tune_new_entries=False)
        
        tuner = RandomSearch(self.build_hypermodel,objective="val_mape",hyperparameters=hp,
                             max_trials=max_trials,executions_per_trial=1,directory="logs",overwrite=True)

        # Load data and callbacks
        train_in, train_out = self.training_points[None][0]
        callbacks_all = self.get_callbacks()

        # Remove early stopping
        callbacks = [call for call in callbacks_all if not isinstance(call,tf.keras.callbacks.EarlyStopping)]


        # Optimize
        tuner.search(train_in, train_out, batch_size = train_in.shape[0],
            epochs=10, validation_split = 0.2, verbose=0, shuffle=True, callbacks=callbacks)

        # Retrieve and save best model
        best_hp = tuner.get_best_hyperparameters()[0]
        untrained_model = tuner.hypermodel.build(best_hp)
        untrained_model.save(self.log_dir)

    def _train(self):
        """
        Train the ANN.

        API function: train the neural net
        """
        # Load untrained optimized model
        self.model = tf.keras.models.load_model(self.log_dir)
        
        # Load data and callbacks
        train_in, train_out = self.training_points[None][0]
        callbacks = self.get_callbacks()

        # Train the ANN
        histor = self.model.fit(train_in, train_out, batch_size = train_in.shape[0],
            epochs=self["no_epochs"], validation_split = 0.2, verbose=0, shuffle=True, callbacks=callbacks)

        # Evaluate the model
        self.validation_metric = self.evaluation(histor.history)
##        breakpoint()
        self._is_trained = True
        if self.options["plot_history"]:
            self.plot_training_history(histor,train_in,train_out,self.model.predict)
       
    def evaluation(self,history):
        """
        Evaluate the generalization error.

        Returns:
            error: validation metric
        """
        error = history["val_mean_squared_error"][-1]
        print('MSE: %.3f, RMSE: %.3f' % (error, np.sqrt(error)))
        print("MAE: %.3f" % (history["val_mean_absolute_error"][-1]))
        print("MAPE: %.0f" % (history["val_mean_absolute_percentage_error"][-1]))

        return error

    def _predict_values(self, x):
        """
        API method: predict values using appropriate methods from the neural_network.py module

        :param  x: np.ndarray[n, nx] -- Input values for the prediction points
        :return y: np.ndarray[n, ny] -- Output values at the prediction points
        """
        return self.model.predict(x)


    def plot_training_history(self,history,inputs,outputs,predict):
        """
        Plot the evolution of the training and testing error.

        Arguments:
            history: training history object

        Raises:
            ValueError: if the surrogate has not been trained yet
            
        """
        if self._is_trained:
            plt.close()
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.title('Learning Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.plot(history.history['loss'], "k-", label='train')
            plt.plot(history.history['val_loss'], "r--", label='val')
##            plt.ylim([0,.1])
            plt.legend()
            plt.ylim(bottom=0)
            
            plt.subplot(1, 2, 2)
            plt.scatter(outputs.flatten(),predict(inputs).flatten())
            plt.title('Prediction correletation')
            plt.xlabel('Data')
            plt.ylabel('Prediction')
            plt.xlim([-1,1])
            plt.ylim([-1,1])
            plt.show()
        else:
            raise ValueError("Not trained yet")

##    def build_model(self):
##        model = tf.keras.Sequential()
##        kernel_regularizer = tf.keras.regularizers.l2(self["kernel_regularizer"])
##        
##        # Add layers        
##        model.add(tf.keras.layers.Dense(self["architecture"][1], activation=self["activation"],
##                                             kernel_initializer=self["init"],
##                                             bias_initializer=self["bias_init"],
##                                             kernel_regularizer=kernel_regularizer,
##                                             input_shape=(self["architecture"][0],)))
##        for neurons in self["architecture"][2:-1]:
##            model.add(tf.keras.layers.Dense(neurons, activation=self["activation"],
##                                                 kernel_initializer=self["init"],
##                                                 bias_initializer=self["bias_init"],
##                                                 kernel_regularizer=kernel_regularizer))
##        model.add(tf.keras.layers.Dense(self["architecture"][-1],activation="sigmoid"))
##
##        # Train
##        optimizer = tf.keras.optimizers.get(self["optimizer"])
##        optimizer._hyper["learning_rate"] = self["learning_rate"]
##
##        model.compile(optimizer,self["loss"],metrics=["mape"])
##
##        return model
##
