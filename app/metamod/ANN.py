"""
Custom ANN definition using TensorFlow.

This module contains the definition of an ANN comptable
with the SMT Toolbox
"""
# Import native packages
import os

# Import pypi packages
from kerastuner.tuners import RandomSearch
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
    """

    def __getitem__(self,a):
        return self.options[a]

    def __init__(self,setup,keras_optimized,**kwargs):
        self.configurations = setup
        super().__init__(**kwargs)
        # Initialize model
        in_dim, out_dim = self["dims"]
        self.options.declare("architecture", [in_dim]+[self["no_neurons"]]*self["no_layers"]+[out_dim], types=list, desc="default architecture")
        self.model = self.build_model()

        # Calculate for pruning
        if self["prune"]:
            prune_model()

        self._is_trained = False
        self.optimized = keras_optimized
        
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
            no_points: nu,ber of sample points
            prune: whether to use pruning
            sparsity: target network sparsity (fraction of zero weights)
            pruning_frequency: frequency of pruning
            tensorboard: whether to make tensorboard output
            stopping: use early stopping
            stopping_delta: required error delta threshold to stop training
            stopping_patience: number of iterations to wait before stopping
            plot_history: whether to plot the training history
        """
        setup = self.configurations
        # Set default values
        declare = self.options.declare
        declare("no_layers", setup["no_layers"], types=int, desc="number of layers")
        declare("no_neurons", setup["no_neurons"], types=int, desc="neurons per layer")
        declare("activation", setup["activation"], types=str, desc="activation function")
        declare("batch_size", setup["batch_size"], types=int, desc="batch size")
        declare("no_epochs", setup["no_epochs"], types=int, desc="no epochs")
        declare("init", setup["init"], types=str, desc="weight initialization")
        declare("bias_init", setup["bias_init"], types=str, desc="bias initialization")
        declare("optimizer", setup["optimizer"], types=str, desc="optimizer")
        declare("learning_rate", setup["learning_rate"], types=float, desc="optimizer")
        declare("loss", setup["loss"], types=str, desc="loss function")
        declare("kernel_regularizer", setup["kernel_regularizer"], types=float, desc="regularization") # 000
        declare("prune", setup["prune"], types=bool, desc="pruning")
        declare("sparsity", setup["sparsity"], types=float, desc="target sparsity")
        declare("pruning_frequency", setup["pruning_frequency"], types=int, desc="pruning frequency")
        declare("tensorboard", setup["tensorboard"], types=bool, desc="tensorboard") ## not tested
        declare("stopping", setup["stopping"], types=bool, desc="early stopping")
        declare("stopping_delta", setup["stopping_delta"], types=float, desc="stopping delta")
        declare("stopping_patience", setup["stopping_patience"], types=int, desc="stopping patience")
        declare("plot_history", setup["plot_history"], types=bool, desc="plot training history")

        # Declare variables
        declare("dims", (None,None), types=tuple, desc="in and out dimensions")
        declare("no_points", None, types=int, desc="number of training points")

##        self.supports["derivatives"] = True
        self.name = "ANN"

    def _train(self):
        """
        Train the ANN.

        API function: train the neural net
        """
        stopping_delta = self.options["stopping_delta"]
        stopping_patience = self.options["stopping_patience"]
        
        callbacks = []

        if self.options["prune"]:
            callbacks.append(sparsity.UpdatePruningStep())
        if self.options["tensorboard"]:
            raise ValueError("Tensorboard not implemented yet")
##            log_dir = "logs\\fit\\"#+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
##            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=20))
        if self.options["stopping"]:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=stopping_delta, patience=stopping_patience, verbose=1,restore_best_weights=True))
##            callback = tf.keras.callbacks.MyStopping(monitor='val_loss', target=5, patience=3, verbose=1)

        train_in, train_out = self.training_points[None][0]

        batch_size = self.options["batch_size"]
        no_epochs = self.options["no_epochs"]

        if not self.optimized:
            print("Hello")
            log_dir = os.path.join(settings["root"],"logs")
            tuner = RandomSearch(self.build_hypermodel,objective="val_mape",max_trials=10,executions_per_trial=1,directory=log_dir)

            tuner.search(
              train_in, train_out,
              epochs=no_epochs, validation_split = 0.2, verbose=0, shuffle=True, batch_size = train_in.shape[0]//1)

        histor = self.model.fit(train_in, train_out, epochs=no_epochs, batch_size=batch_size, verbose=0, callbacks=callbacks, validation_split = 0.2)
        self.metric = self.evaluation(histor.history)

        self._is_trained = True
        if self.options["plot_history"]:
            self.plot_training_history(histor)
        

    def _predict_values(self, x):
        """
        API method: predict values using appropriate methods from the neural_network.py module

        :param  x: np.ndarray[n, nx] -- Input values for the prediction points
        :return y: np.ndarray[n, ny] -- Output values at the prediction points
        """
        return self.model.predict(x)


    def plot_training_history(self,history):
        """
        Plot the evolution of the training and testing error.

        Arguments:
            history: training history object

        Raises:
            ValueError: if the surrogate has not been trained yet
            
        """
        if self._is_trained:
            plt.title('Learning Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.plot(history.history['loss'], "k-", label='train')
            plt.plot(history.history['val_loss'], "r--", label='val')
##            plt.ylim([0,.1])
            plt.legend()
            plt.ylim(bottom=0)
            plt.show()
        else:
            ValueError("Not trained yet")

    def evaluation(self,history):
        """
        Evaluate the generalization error.

        Returns:
            error: validation metric

        Notes:
            Assumes erros is MSE
        """
        error = history["val_loss"][-1]
        print('MSE: %.3f, RMSE: %.3f' % (error, np.sqrt(error)))
        if "mean_absolute_percentage_error" in self.model.metrics_names:
            print("MAPE: %.0f" % (error["val_mape"]))

        return error

    def build_model(self):
        model = tf.keras.Sequential()
        kernel_regularizer = tf.keras.regularizers.l2(self["kernel_regularizer"])
        
        # Add layers        
        model.add(tf.keras.layers.Dense(self["architecture"][1], activation=self["activation"],
                                             kernel_initializer=self["init"],
                                             bias_initializer=self["bias_init"],
                                             kernel_regularizer=kernel_regularizer,
                                             input_shape=(self["architecture"][0],)))
        for neurons in self["architecture"][2:-1]:
            model.add(tf.keras.layers.Dense(neurons, activation=self["activation"],
                                                 kernel_initializer=self["init"],
                                                 bias_initializer=self["bias_init"],
                                                 kernel_regularizer=kernel_regularizer))
        model.add(tf.keras.layers.Dense(self["architecture"][-1],activation="sigmoid"))

        # Train
        optimizer = tf.keras.optimizers.get(self["optimizer"])
        optimizer._hyper["learning_rate"] = self["learning_rate"]

        model.compile(optimizer,self["loss"],metrics=["mape"])

        return model

    def build_hypermodel(self,hp):
        model = tf.keras.Sequential()
        kernel_regularizer = tf.keras.regularizers.l2(self["kernel_regularizer"])
        
        neurons_hyp = hp.Int("No_neur",1,20)

        # Add layers        
        model.add(tf.keras.layers.Dense(self["architecture"][1], activation=self["activation"],
                                             kernel_initializer=self["init"],
                                             bias_initializer=self["bias_init"],
                                             kernel_regularizer=kernel_regularizer,
                                             input_shape=(self["architecture"][0],)))
        for neurons in self["architecture"][2:-1]:
            model.add(tf.keras.layers.Dense(neurons_hyp, activation=self["activation"],
                                                 kernel_initializer=self["init"],
                                                 bias_initializer=self["bias_init"],
                                                 kernel_regularizer=kernel_regularizer))
        model.add(tf.keras.layers.Dense(self["architecture"][-1],activation="sigmoid"))

        # Train
        optimizer = tf.keras.optimizers.get(self["optimizer"])
        optimizer._hyper["learning_rate"] = self["learning_rate"]

        model.compile(optimizer,self["loss"],metrics=["mape"])

        model.summary()

        return model


    def prune_model(self):
        end_step = np.ceil(1.0 * self["no_points"] / self["batch_size"]).astype(np.int32) * ["no_epochs"]
        print(f"End step: {end_step}")

        # Add pruning layers
        new_pruning_params = {
              'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.0,
                                                           final_sparsity=self["sparsity"],
                                                           begin_step=0,
                                                           end_step=end_step,
                                                           frequency=self["pruning_frequency"])
        }

        self.model = sparsity.prune_low_magnitude(self.model, **new_pruning_params)

    def optimize(self):
        pass
